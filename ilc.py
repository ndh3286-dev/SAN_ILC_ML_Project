"""
SVD-filtered Iterative Learning Control (ILC) for TwoQubitModel pulses.

Structure follows the NumericalSVDILC outline:
  1. simulate_rollout   - observables y(t_k) at checkpoints (optionally with shot noise)
  2. compute_jacobian   - J = dy(t_k)/da_m(t_n), stacked over times/states/observables
  3. compute_svd_basis  - J ~ U_r S_r V_r^T  (pulse modes V_r, error modes U_r)
  4. select_measurement_times - the few checkpoint times with most Fisher information
  5. update_pulse       - mode-space ILC update from measured errors

Differences from the finite-difference outline:
  * The Jacobian is analytic. With H_k = V diag(lam) V^+ and U_k = exp(-i H_k dt),
        dU_k = V (G o (V^+ dH V)) V^+,  G_ij = (e^{-i lam_i dt} - e^{-i lam_j dt}) / (lam_i - lam_j)
    (Daleckii-Krein), and dy_k/da_n = 2 Re <psi_k| O U_k...U_{n+1} dU_n |psi_{n-1}>,
    evaluated for every (k, O) in one backward sweep. dH/dI_j uses the
    tuning-curve derivative (TwoQubitModel.detuning_jacobian).
  * Controls are normalized by per-channel `scales` (mA vs GHz) before the SVD.
  * Measurements can be restricted to a subset of checkpoint times; the
    update then fits the r mode coefficients from those rows only.

Control layout: a has shape (6, N), rows [I1, I2, ux1, ux2, uy1, uy2].
Observable vector y is flattened time-major: (n_checkpoints, n_states, n_obs).
"""

from __future__ import annotations

import numpy as np
import qutip as qt

from two_qubit_model import Controls, TwoQubitModel

CHANNELS = ["I1", "I2", "ux1", "ux2", "uy1", "uy2"]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def ctrl_to_array(ctrl: Controls) -> np.ndarray:
    return np.vstack([ctrl.I, ctrl.ux, ctrl.uy])


def array_to_ctrl(a: np.ndarray, dt: float) -> Controls:
    return Controls(dt, a[0:2], a[2:4], a[4:6])


def embedded_paulis(model: TwoQubitModel) -> dict:
    """Single-qubit Paulis on the {0,1} subspace of each transmon (zero on |2>)."""
    n = model.p.n_levels
    b0, b1 = qt.basis(n, 0), qt.basis(n, 1)
    P = {"X": b0 * b1.dag() + b1 * b0.dag(),
         "Y": -1j * b0 * b1.dag() + 1j * b1 * b0.dag(),
         "Z": b0.proj() - b1.proj()}
    Id = qt.qeye(n)
    out = {}
    for name, op in P.items():
        out[f"{name}I"] = qt.tensor(op, Id)
        out[f"I{name}"] = qt.tensor(Id, op)
    return out


def product_state(model: TwoQubitModel, s1: str, s2: str) -> qt.Qobj:
    """Product state from labels in {'0', '1', '+', '-', '+i', '-i'}."""
    n = model.p.n_levels
    b0, b1 = qt.basis(n, 0), qt.basis(n, 1)
    lut = {"0": b0, "1": b1, "+": (b0 + b1).unit(), "-": (b0 - b1).unit(),
           "+i": (b0 + 1j * b1).unit(), "-i": (b0 - 1j * b1).unit()}
    return qt.tensor(lut[s1], lut[s2])


def tracking_fidelity(model: TwoQubitModel, U: qt.Qobj, U_ref: qt.Qobj) -> float:
    """|Tr(A^+ B)|^2 / (Tr(A^+A) Tr(B^+B)) on the computational block: 1 iff U matches U_ref."""
    A, B = model.project_computational(U_ref), model.project_computational(U)
    return float(abs(np.trace(A.conj().T @ B)) ** 2
                 / (np.trace(A.conj().T @ A).real * np.trace(B.conj().T @ B).real))


def _slice_eig(model: TwoQubitModel, ctrl: Controls):
    """Eigen-decomposition and propagator of every slice Hamiltonian."""
    H = model.slice_hamiltonians(ctrl)                     # (N, d, d) rad/ns
    lam, V = np.linalg.eigh(H)
    ph = np.exp(-1j * lam * ctrl.dt)
    U = (V * ph[:, None, :]) @ V.conj().transpose(0, 2, 1)
    return lam, V, ph, U


def _propagate(U: np.ndarray, psi0: np.ndarray) -> np.ndarray:
    """States at every slice boundary, shape (n_states, N+1, d)."""
    out = np.empty((psi0.shape[0], U.shape[0] + 1, psi0.shape[1]), dtype=complex)
    psi = psi0.copy()
    out[:, 0] = psi
    for k in range(U.shape[0]):
        psi = psi @ U[k].T
        out[:, k + 1] = psi
    return out


# --------------------------------------------------------------------------- #
# SVD-ILC
# --------------------------------------------------------------------------- #
class SVDILC:
    """
    Parameters
    ----------
    model : TwoQubitModel used for the offline (nominal) Jacobian.
    initial_states : list of kets |psi_s>.
    observables : list of operators O_m.
    checkpoints : slice-boundary indices k (1..N) where observables may be measured.
    scales : per-channel normalization [I1, I2 (mA), ux1, ux2, uy1, uy2 (GHz)].
    """

    def __init__(self, model, initial_states, observables, checkpoints,
                 scales=(0.36, 0.36, 0.05, 0.05, 0.05, 0.05)):
        self.model = model
        self.psi0 = np.stack([s.full().ravel() for s in initial_states])
        self.observables = [o.full() for o in observables]
        self.checkpoints = np.asarray(checkpoints, dtype=int)
        self.scales = np.asarray(scales, dtype=float)
        self.N_states = len(initial_states)
        self.N_obs = len(observables)
        self.N_checkpoints = len(self.checkpoints)
        self.rows_per_time = self.N_states * self.N_obs
        self.dim_y = self.N_checkpoints * self.rows_per_time

    # ---- 1. rollout ------------------------------------------------------
    def simulate_rollout(self, ctrl: Controls, model=None, substeps=1,
                         shots=None, rng=None, times=None):
        """Observables at checkpoints, flattened (time-major).

        model    : override (e.g. the 'hardware' model); defaults to the nominal one.
        substeps : split each slice into sub-slices (finer integration).
        shots    : if given, Pauli expectations estimated from `shots` ±1 outcomes.
        times    : positions into self.checkpoints to return (default: all).
        Returns (y, U_total).
        """
        model = self.model if model is None else model
        if substeps > 1:
            rep = lambda x: np.repeat(x, substeps, axis=1)
            ctrl = Controls(ctrl.dt / substeps, rep(ctrl.I), rep(ctrl.ux), rep(ctrl.uy))
        _, _, _, U = _slice_eig(model, ctrl)
        states = _propagate(U, self.psi0)
        pos = np.arange(self.N_checkpoints) if times is None else np.asarray(times)
        psi = states[:, self.checkpoints[pos] * substeps, :]          # (S, K, d)
        y = np.stack([np.einsum("skd,de,ske->ks", psi.conj(), O, psi).real
                      for O in self.observables], axis=-1).ravel()     # (K*S*O,)
        if shots is not None:
            rng = np.random.default_rng() if rng is None else rng
            p = np.clip((1 + y) / 2, 0, 1)
            y = 2 * rng.binomial(shots, p) / shots - 1
        Utot = np.eye(U.shape[1], dtype=complex)
        for Uk in U:
            Utot = Uk @ Utot
        return y, qt.Qobj(Utot, dims=[model.dims, model.dims])

    # ---- 2. analytic Jacobian ------------------------------------------------
    def compute_jacobian(self, ctrl: Controls):
        """J (dim_y x 6N) w.r.t. normalized controls a/scales, and y_nom."""
        model, N, dt = self.model, ctrl.n_slices, ctrl.dt
        lam, V, ph, U = _slice_eig(model, ctrl)
        states = _propagate(U, self.psi0)
        K, S, O = self.N_checkpoints, self.N_states, self.N_obs
        d = U.shape[1]
        Os = np.stack(self.observables)
        Hz = [h.full() for h in model.H_z]
        Hxy = [h.full() for h in model.H_x] + [h.full() for h in model.H_y]
        dDelta = model.detuning_jacobian(ctrl.I)                # (2, 2, N) GHz/mA

        diff = lam[:, :, None] - lam[:, None, :]
        same = np.abs(diff) < 1e-10
        G = np.where(same, -1j * dt * ph[:, :, None],
                     (ph[:, :, None] - ph[:, None, :]) / np.where(same, 1.0, diff))
        ck_pos = {int(k): p for p, k in enumerate(self.checkpoints)}

        J = np.zeros((K, S, O, 6, N))
        for s in range(S):
            X = np.zeros((d, K * O), dtype=complex)            # costates W^+ O psi_k
            for n in range(N, 0, -1):                           # slice n: psi_{n-1} -> psi_n
                if n in ck_pos:
                    p = ck_pos[n]
                    X[:, p * O:(p + 1) * O] = (Os @ states[s, n]).T
                Vn, Gn = V[n - 1], G[n - 1]
                w = Vn.conj().T @ states[s, n - 1]
                dH = [dDelta[0, j, n - 1] * Hz[0] + dDelta[1, j, n - 1] * Hz[1] for j in range(2)] + Hxy
                for c, B in enumerate(dH):
                    g = Vn @ ((Gn * (Vn.conj().T @ B @ Vn)) @ w)
                    J[:, s, :, c, n - 1] = 2 * (X.conj().T @ g).real.reshape(K, O) * self.scales[c]
                X = U[n - 1].conj().T @ X

        y_nom = np.stack([np.einsum("skd,de,ske->ks", states[:, self.checkpoints].conj(), Ob,
                                    states[:, self.checkpoints]).real for Ob in Os], axis=-1).ravel()
        return J.reshape(self.dim_y, 6 * N), y_nom

    # ---- 3. SVD -------------------------------------------------------------
    def compute_svd_basis(self, J_stacked, r_rank=4):
        """Top-r modes: V_r (6N, r) pulse envelopes, U_r (dim_y, r) error modes, Sigma_r (r,)."""
        U, S, Vt = np.linalg.svd(J_stacked, full_matrices=False)
        r = min(r_rank, len(S))
        return Vt[:r].T, U[:, :r], S[:r]

    # ---- 4. measurement-time selection ----------------------------------------
    def select_measurement_times(self, U_r, n_times=4, y_nom=None, shots=1000):
        """Greedy D-optimal choice of checkpoint times from the error modes U_r.

        Each time t_k contributes Fisher information F_k = sum_rows u_row u_row^T / var_row
        about the r mode amplitudes (var_row = (1 - y^2)/shots, the binomial shot-noise
        variance). Times are added one at a time to maximize log det(sum F_k): the
        first pick is the time with the largest error-mode content, and later picks
        add the most new information rather than repeating neighbours.

        Returns (positions into self.checkpoints, per-time score trace(F_k)).
        """
        r = U_r.shape[1]
        w = np.ones(self.dim_y) if y_nom is None else shots / np.maximum(1 - y_nom ** 2, 1 / shots)
        Ub = (U_r * np.sqrt(w)[:, None]).reshape(self.N_checkpoints, self.rows_per_time, r)
        F_t = np.einsum("kir,kis->krs", Ub, Ub)
        chosen, F = [], 1e-9 * np.eye(r)
        for _ in range(n_times):
            k = max((k for k in range(self.N_checkpoints) if k not in chosen),
                    key=lambda k: np.linalg.slogdet(F + F_t[k])[1])
            chosen.append(k)
            F = F + F_t[k]
        return np.sort(chosen), np.einsum("krr->k", F_t)

    def rows_for_times(self, positions):
        """Flat indices into y for the given checkpoint positions."""
        R = self.rows_per_time
        return (np.asarray(positions)[:, None] * R + np.arange(R)[None, :]).ravel()

    # ---- 5. ILC update ----------------------------------------------------------
    def update_pulse(self, ctrl, y_real, y_nom, V_r, U_r, Sigma_r, rows=None,
                     reg_lambda=1e-3, gain=1.0):
        """SVD-filtered ILC step.

        y_real : measured observables at `rows` (all rows if None).
        Fits c = argmin ||U_r[rows] Sigma_r c - e_raw||^2 + reg^2 ||c||^2, then
        a_next = a - gain * V_r c. With all rows this reduces to
        c = Sigma_r/(Sigma_r^2 + reg^2) * U_r^T e_raw, and the filtered error is U_r U_r^T e_raw.
        Returns (ctrl_next, e_filtered, c).
        """
        rows = np.arange(len(y_nom)) if rows is None else np.asarray(rows)
        e_raw = y_real - y_nom[rows]
        A = U_r[rows] * Sigma_r[None, :]
        c = np.linalg.solve(A.T @ A + reg_lambda ** 2 * np.eye(len(Sigma_r)), A.T @ e_raw)
        e_filtered = A @ c
        delta_a = -(V_r @ c).reshape(6, -1) * self.scales[:, None]
        a_next = ctrl_to_array(ctrl) + gain * delta_a
        return array_to_ctrl(a_next, ctrl.dt), e_filtered, c
