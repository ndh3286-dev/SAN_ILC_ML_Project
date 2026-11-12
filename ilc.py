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

import copy
import re

import numpy as np
import qutip as qt

from two_qubit_model import Controls, SystemParams, TwoQubitModel

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


def get_param(p: SystemParams, name: str) -> float:
    """Read a parameter by name: 'M[i,j]', 'I_idle[i]', 'J', 'n_levels', or 'q1.f_max' etc."""
    m = re.fullmatch(r"(M|I_idle)\[(\d+)(?:,(\d+))?\]", name.replace(" ", ""))
    if m:
        arr = getattr(p, m.group(1))
        return float(arr[int(m.group(2)), int(m.group(3))] if m.group(3) else arr[int(m.group(2))])
    if "." in name:
        q, attr = name.split(".")
        return float(getattr(getattr(p, q), attr))
    return float(getattr(p, name))


def set_param(p: SystemParams, name: str, value: float) -> None:
    m = re.fullmatch(r"(M|I_idle)\[(\d+)(?:,(\d+))?\]", name.replace(" ", ""))
    if m:
        arr = getattr(p, m.group(1))
        if m.group(3):
            arr[int(m.group(2)), int(m.group(3))] = value
        else:
            arr[int(m.group(2))] = value
    elif "." in name:
        q, attr = name.split(".")
        setattr(getattr(p, q), attr, value)
    else:
        setattr(p, name, value)


def with_parameters(base: SystemParams, names, theta) -> TwoQubitModel:
    """TwoQubitModel built from `base` with parameters `names` set to `theta`."""
    p = copy.deepcopy(base)
    for n, v in zip(names, theta):
        set_param(p, n, float(v))
    return TwoQubitModel(p)


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
    active_channels : controls the ILC may correct, by name from CHANNELS
        (default: all six). Inactive channels get zero Jacobian columns, so the
        SVD modes and every pulse update leave them untouched.
        E.g. ["I1", "I2"] ignores the qubit drives.
    """

    def __init__(self, model, initial_states, observables, checkpoints,
                 scales=(0.36, 0.36, 0.05, 0.05, 0.05, 0.05), active_channels=None):
        self.model = model
        self.psi0 = np.stack([s.full().ravel() for s in initial_states])
        self.observables = [o.full() for o in observables]
        self.checkpoints = np.asarray(checkpoints, dtype=int)
        self.scales = np.asarray(scales, dtype=float)
        active_channels = CHANNELS if active_channels is None else list(active_channels)
        unknown = set(active_channels) - set(CHANNELS)
        if unknown:
            raise ValueError(f"unknown channels {unknown}; choose from {CHANNELS}")
        self.active = np.array([ch in active_channels for ch in CHANNELS])
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
                    if not self.active[c]:
                        continue                                # column stays zero
                    g = Vn @ ((Gn * (Vn.conj().T @ B @ Vn)) @ w)
                    J[:, s, :, c, n - 1] = 2 * (X.conj().T @ g).real.reshape(K, O) * self.scales[c]
                X = U[n - 1].conj().T @ X

        y_nom = np.stack([np.einsum("skd,de,ske->ks", states[:, self.checkpoints].conj(), Ob,
                                    states[:, self.checkpoints]).real for Ob in Os], axis=-1).ravel()
        return J.reshape(self.dim_y, 6 * N), y_nom

    # ---- 3. SVD -------------------------------------------------------------
    def compute_svd_basis(self, J_stacked, r_rank=4):
        """Top-r modes over all controls jointly: V_r (6N, r) pulse envelopes,
        U_r (dim_y, r) error modes, Sigma_r (r,).

        No channel is singled out: every mode may mix flux and drive channels,
        and the SVD ranks directions purely by their effect on the observables.
        (Channels are compared in units of `scales`, i.e. fractions of each
        control's range.)
        """
        U, S, Vt = np.linalg.svd(J_stacked, full_matrices=False)
        r = min(r_rank, len(S))
        return Vt[:r].T, U[:, :r], S[:r]

    # ---- 4. measurement-time selection ----------------------------------------
    def select_measurement_times(self, U_r, n_times=4, y_nom=None, shots=1000, required=None):
        """Greedy D-optimal choice of checkpoint times from the error modes U_r.

        Each time t_k contributes Fisher information F_k = sum_rows u_row u_row^T / var_row
        about the r mode amplitudes (var_row = (1 - y^2)/shots, the binomial shot-noise
        variance). Times are added one at a time to maximize log det(sum F_k): the
        first pick is the time with the largest error-mode content, and later picks
        add the most new information rather than repeating neighbours.

        required : checkpoint positions that are always measured (e.g. [-1] for the
            final time). They count toward n_times, and the remaining picks maximize
            the information added on top of them.

        Returns (positions into self.checkpoints, per-time score trace(F_k)).
        """
        r = U_r.shape[1]
        w = np.ones(self.dim_y) if y_nom is None else shots / np.maximum(1 - y_nom ** 2, 1 / shots)
        Ub = (U_r * np.sqrt(w)[:, None]).reshape(self.N_checkpoints, self.rows_per_time, r)
        F_t = np.einsum("kir,kis->krs", Ub, Ub)
        chosen = [] if required is None else sorted({int(k) % self.N_checkpoints for k in required})
        if len(chosen) > n_times:
            raise ValueError("more required times than n_times")
        F = 1e-9 * np.eye(r) + sum((F_t[k] for k in chosen), np.zeros((r, r)))
        for _ in range(n_times - len(chosen)):
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

    # ---- 6. adaptive model: Gauss-Newton parameter fit --------------------------
    def fit_parameters(self, base_params: SystemParams, names, theta0, history,
                       shots=1000, prior=None, prior_std=None, n_gn=5,
                       fd_step=1e-4, tol=1e-6, scan=None, scan_points=41,
                       chi2_rescan=3.0, verbose=False):
        """Fit Hamiltonian parameters to every hardware measurement so far.

        Minimizes  sum_j ||y_meas,j - y(a_j; theta)||^2_W + ||(theta - prior)/prior_std||^2
        by Gauss-Newton with step halving. W = shots / (1 - y^2) is the binomial
        shot-noise weight. The parameter Jacobian dy/dtheta uses central finite
        differences (a few parameters, so a few rollouts) and so includes every way
        a parameter enters the model (flux, idle frequencies, rotating frame).

        Grid-scan seeding: Gauss-Newton cannot leave a stationary point. A pulse
        that holds a qubit at its sweet spot gives dy/dM = 0 there, and the data
        cannot tell +phi from -phi. So when `scan` is given ({name: (lo, hi)}),
        a coordinate-wise grid scan picks the starting point (two passes,
        `scan_points` per parameter) if this is the first fit (len(history) == 1)
        or the Gauss-Newton fit from theta0 ends with chi2/meas > chi2_rescan.

        base_params : SystemParams of the offline model (other parameters kept fixed).
        names       : parameter names, e.g. ["M[1,0]", "M[0,1]"].
        theta0      : starting values (usually the current estimate).
        history     : list of (ctrl, times, y_meas) from the hardware, times = checkpoint positions.
        prior, prior_std : Gaussian prior (default: theta0 with std 1.0, i.e. weak).
        Returns (theta, fitted TwoQubitModel, info dict).
        """
        theta0 = np.asarray(theta0, dtype=float).copy()
        prior = theta0.copy() if prior is None else np.asarray(prior, dtype=float)
        prior_std = np.ones_like(theta0) if prior_std is None else np.asarray(prior_std, dtype=float)
        y_meas = np.concatenate([y for _, _, y in history])
        w = np.sqrt(shots / np.maximum(1 - y_meas ** 2, 1 / shots))

        def predict(th):
            m = with_parameters(base_params, names, th)
            return np.concatenate([self.simulate_rollout(c, model=m, times=t)[0]
                                   for c, t, _ in history])

        def cost(th, y_pred):
            return float(np.sum((w * (y_meas - y_pred)) ** 2)
                         + np.sum(((th - prior) / prior_std) ** 2))

        def chi2(y_pred):
            return float(np.sum((w * (y_meas - y_pred)) ** 2) / len(y_meas))

        def gauss_newton(theta):
            y_pred = predict(theta)
            f = cost(theta, y_pred)
            Jth = np.zeros((len(y_meas), len(theta)))
            for it in range(n_gn):
                # parameter Jacobian by central differences
                for i in range(len(theta)):
                    h = fd_step * max(1.0, abs(theta[i]))
                    tp, tm = theta.copy(), theta.copy()
                    tp[i] += h
                    tm[i] -= h
                    Jth[:, i] = (predict(tp) - predict(tm)) / (2 * h)
                # weighted Gauss-Newton step with prior rows
                A = np.vstack([w[:, None] * Jth, np.diag(1 / prior_std)])
                b = np.concatenate([w * (y_meas - y_pred), (prior - theta) / prior_std])
                step = np.linalg.lstsq(A, b, rcond=None)[0]
                # backtrack until the cost decreases
                for _ in range(8):
                    th_new = theta + step
                    y_new = predict(th_new)
                    f_new = cost(th_new, y_new)
                    if f_new < f:
                        break
                    step = step / 2
                else:
                    break
                theta, y_pred, f = th_new, y_new, f_new
                if verbose:
                    print(f"  GN {it + 1}: theta = {np.round(theta, 5)}, chi2/meas = {chi2(y_pred):.2f}")
                if np.max(np.abs(step)) < tol:
                    break
            return theta, y_pred, Jth

        def grid_scan(theta):
            theta = theta.copy()
            for _ in range(2):
                for i, n in enumerate(names):
                    if n not in scan:
                        continue
                    grid = np.linspace(*scan[n], scan_points)
                    costs = []
                    for v in grid:
                        th = theta.copy()
                        th[i] = v
                        costs.append(cost(th, predict(th)))
                    theta[i] = grid[int(np.argmin(costs))]
            if verbose:
                print(f"  scan start: theta = {np.round(theta, 4)}")
            return theta

        info = {"scanned": False}
        start = theta0
        if scan and len(history) == 1:
            start, info["scanned"] = grid_scan(theta0), True
        theta, y_pred, Jth = gauss_newton(start)
        if scan and not info["scanned"] and chi2(y_pred) > chi2_rescan:
            th2, y2, J2 = gauss_newton(grid_scan(theta))
            info["scanned"] = True
            if cost(th2, y2) < cost(theta, y_pred):
                theta, y_pred, Jth = th2, y2, J2

        # 1-sigma uncertainty from the Gauss-Newton Hessian
        A = np.vstack([w[:, None] * Jth, np.diag(1 / prior_std)])
        info["theta_std"] = np.sqrt(np.diag(np.linalg.pinv(A.T @ A)))
        info["chi2_per_meas"] = chi2(y_pred)
        return theta, with_parameters(base_params, names, theta), info

    def model_based_update(self, ctrl, y_target, V_r, U_r, Sigma_r, gain=1.0,
                           reg_lambda=1e-3, max_halvings=5):
        """ILC step using the current (fitted) model's noise-free prediction as the error.

        e = y(a; theta_hat) - y_target on all checkpoints. The step is halved until
        the model predicts a smaller trajectory error than before.
        Returns (ctrl_next, predicted rms error after the step, gain used).
        """
        y_pred, _ = self.simulate_rollout(ctrl)
        rms0 = np.sqrt(np.mean((y_pred - y_target) ** 2))
        for _ in range(max_halvings + 1):
            c_next, _, _ = self.update_pulse(ctrl, y_pred, y_target, V_r, U_r, Sigma_r,
                                             reg_lambda=reg_lambda, gain=gain)
            rms1 = np.sqrt(np.mean((self.simulate_rollout(c_next)[0] - y_target) ** 2))
            if rms1 < rms0:
                return c_next, rms1, gain
            gain /= 2
        return ctrl, rms0, 0.0


# --------------------------------------------------------------------------- #
# Adaptive SVD-ILC loop
# --------------------------------------------------------------------------- #
def run_adaptive_ilc(ilc: SVDILC, ctrl0: Controls, y_target, base_params: SystemParams,
                     hardware_model: TwoQubitModel, fit_names, scan=None, prior_std=None,
                     n_iter=12, r_rank=4, n_times=4, shots=1000,
                     substeps=4, gain=1.0, first_times=None, U_ref=None, seed=0, verbose=True,
                     forget_sigma=None, required_times=None):
    """Measure -> Gauss-Newton parameter fit -> re-linearize (SVD) -> model-based control update.

    ilc            : SVDILC whose .model is the offline model (it is replaced by the fit each step).
    y_target       : trajectory to track (nominal pulse in the offline model).
    base_params    : SystemParams of the offline model; fit_names are varied on top of it.
    hardware_model : the 'real' device, measured with shot noise at n_times checkpoints.
    U_ref          : nominal unitary for scoring (computed if None).
    required_times : checkpoint positions measured on every iteration (e.g. [-1] = final time).
    forget_sigma   : forgetting-factor ILC. None (default) accumulates corrections as usual.
                     Otherwise a scalar or one value per fit parameter (same units), and
                         a_{k+1} = a_0 + lambda_k (a_k - a_0) + delta_a_k,
                         lambda_k = exp(-sum(((theta_k - theta_{k-1}) / forget_sigma)^2)),
                     so past corrections are discounted when the fitted model moves by
                     more than ~forget_sigma between iterations (lambda -> 0) and kept when
                     it is stable (lambda -> 1). delta_a_k is the update computed at a_k.
    Returns (final Controls, log dict). Hardware truth is used only for the log, never by the learner.
    """
    rng = np.random.default_rng(seed)
    if U_ref is None:
        _, U_ref = ilc.simulate_rollout(ctrl0)
    theta_nom = np.array([get_param(base_params, n) for n in fit_names])
    theta = theta_nom.copy()
    if first_times is None:
        J0, y0 = ilc.compute_jacobian(ctrl0)
        V0, U0, S0 = ilc.compute_svd_basis(J0, r_rank)
        first_times, _ = ilc.select_measurement_times(U0, n_times, y_nom=y0, shots=shots,
                                                      required=required_times)
    pos = first_times

    log = {"rms": [], "tracking_F": [], "theta": [], "theta_std": [], "chi2": [], "times": [],
           "gain": [], "lambda": [], "fit_names": list(fit_names)}
    a0 = ctrl_to_array(ctrl0)
    c_k, hist = ctrl0, []
    for it in range(n_iter + 1):
        y_true, U_hw = ilc.simulate_rollout(c_k, model=hardware_model, substeps=substeps)
        log["rms"].append(float(np.sqrt(np.mean((y_true - y_target) ** 2))))
        log["tracking_F"].append(tracking_fidelity(ilc.model, U_hw, U_ref))
        log["theta"].append(theta.copy())
        log["times"].append(ilc.checkpoints[pos] * ctrl0.dt)
        if verbose:
            print(f"iter {it}: rms err {log['rms'][-1]:.4f}  tracking F {log['tracking_F'][-1]:.5f}  "
                  f"theta {np.round(theta, 4)}")
        if it == n_iter:
            break

        # 1. measure + Gauss-Newton parameter fit on all data so far
        y_meas, _ = ilc.simulate_rollout(c_k, model=hardware_model, substeps=substeps,
                                         shots=shots, rng=rng, times=pos)
        hist.append((c_k, pos, y_meas))
        theta_prev = theta.copy()
        theta, fit_model, info = ilc.fit_parameters(base_params, fit_names, theta, hist, shots=shots,
                                                    prior=theta_nom, prior_std=prior_std, scan=scan)
        log["theta_std"].append(info["theta_std"])
        log["chi2"].append(info["chi2_per_meas"])
        if verbose:
            print(f"   fit chi2/meas {info['chi2_per_meas']:.2f}" + (" (grid scan)" if info["scanned"] else ""))

        # 2. re-linearize at the fitted model
        ilc.model = fit_model
        J_k, y_pred = ilc.compute_jacobian(c_k)
        V_k, U_k, S_k = ilc.compute_svd_basis(J_k, r_rank)

        # 3. model-based update (+ forgetting of past corrections) and new measurement times
        c_new, _, g = ilc.model_based_update(c_k, y_target, V_k, U_k, S_k, gain=gain)
        if forget_sigma is None:
            lam = 1.0
            c_k = c_new
        else:
            lam = float(np.exp(-np.sum(((theta - theta_prev) / np.asarray(forget_sigma, dtype=float)) ** 2)))
            a_k = ctrl_to_array(c_k)
            delta_a = ctrl_to_array(c_new) - a_k
            c_k = array_to_ctrl(a0 + lam * (a_k - a0) + delta_a, ctrl0.dt)
        log["gain"].append(g)
        log["lambda"].append(lam)
        if verbose and forget_sigma is not None:
            print(f"   forgetting factor lambda = {lam:.3f}")
        pos, _ = ilc.select_measurement_times(U_k, n_times, y_nom=y_pred, shots=shots,
                                              required=required_times)
    return c_k, log


# --------------------------------------------------------------------------- #
# Final-unitary SVD-ILC: plan on the fitted model for the end map U_T
# --------------------------------------------------------------------------- #
def _aligned_cols_fidelity(Uc, T, comp):
    """tracking_fidelity computed from the computational columns (D x 4) of U_T and target."""
    A, B = T[comp], Uc[comp]
    return float(abs(np.trace(A.conj().T @ B)) ** 2
                 / (np.trace(A.conj().T @ A).real * np.trace(B.conj().T @ B).real))


def _svdilc_compute_unitary_jacobian(self, ctrl: Controls):
    """Exact d U_T[:, comp] / da for normalized controls.

    Returns (J, Ucols): J is complex with shape (D*4, 6N) (rows = vec of the D x 4
    computational columns of U_T, all D output levels so leakage counts), and
    Ucols = U_T[:, comp]. Same Daleckii-Krein slice derivative as compute_jacobian,
    with the backward product B_n = U_N ... U_{n+1}:
        dU_T[:, j] / da_n = B_n dU_n psi_{n-1}^{(j)}.
    Inactive channels (active_channels) stay zero.
    """
    model, N, dt = self.model, ctrl.n_slices, ctrl.dt
    lam, V, ph, U = _slice_eig(model, ctrl)
    comp = model.computational_indices()
    D = U.shape[1]
    states = _propagate(U, np.eye(D, dtype=complex)[comp])     # (4, N+1, D)
    Hz = [h.full() for h in model.H_z]
    Hxy = [h.full() for h in model.H_x] + [h.full() for h in model.H_y]
    dDelta = model.detuning_jacobian(ctrl.I)
    diff = lam[:, :, None] - lam[:, None, :]
    same = np.abs(diff) < 1e-10
    G = np.where(same, -1j * dt * ph[:, :, None],
                 (ph[:, :, None] - ph[:, None, :]) / np.where(same, 1.0, diff))

    J = np.zeros((D, len(comp), 6, N), dtype=complex)
    B = np.eye(D, dtype=complex)                                 # U_N ... U_{n+1}
    for n in range(N, 0, -1):
        Vn, Gn = V[n - 1], G[n - 1]
        W = Vn.conj().T @ states[:, n - 1].T                     # (D, 4)
        dH = [dDelta[0, j, n - 1] * Hz[0] + dDelta[1, j, n - 1] * Hz[1] for j in range(2)] + Hxy
        for c, Bop in enumerate(dH):
            if not self.active[c]:
                continue
            dUpsi = Vn @ ((Gn * (Vn.conj().T @ Bop @ Vn)) @ W)   # dU_n psi_{n-1}, (D, 4)
            J[:, :, c, n - 1] = (B @ dUpsi) * self.scales[c]
        B = B @ U[n - 1]
    return J.reshape(D * len(comp), 6 * N), states[:, N].T


def _svdilc_final_fidelity(self, ctrl, target_cols, model=None):
    _, Ut = self.simulate_rollout(ctrl, model=model)
    m = self.model if model is None else model
    return _aligned_cols_fidelity(Ut.full()[:, m.computational_indices()], target_cols,
                                  m.computational_indices())


def _svdilc_plan_final_unitary(self, ctrl: Controls, target, a_anchor=None, r_rank=8,
                               n_steps=10, reg=1e-3, anchor=0.0, gain=1.0, tol=1e-7,
                               verbose=False):
    """Gauss-Newton on the final-unitary error with a truncated SVD (on self.model).

    e = vec(e^{-i phi} U_T[:, comp] - T[:, comp]), phi = global phase aligning U_T to T.
    Objective (scaled control units, n = number of control values):
        C(a) = ||e(a)||^2 + anchor * mean((a - a_anchor)^2)
    J_U ~ U_r S_r V_r^T (real-stacked); the step in the top-r pulse modes solves the
    linearized objective, with mu = anchor / n:
        c = -(S_r U_r^T e + mu V_r^T (a - a_anchor)) / (S_r^2 + mu + reg^2),  da = V_r c.
    `anchor` > 0 pulls the pulse toward a_anchor (default: the starting pulse) so it
    stays where the model has been tested; 0 = pure fidelity. The anchor is a mean
    over control values, so its strength does not depend on pulse length.
    Each step is halved until C decreases. Intermediate times are free: only the
    end map is targeted.

    target : D x D Qobj/array (e.g. the nominal U_ref) or its D x 4 computational columns.
    Returns (ctrl, info) with info["fidelity"] and info["cost"] per accepted step.
    """
    comp = self.model.computational_indices()
    T = target.full() if isinstance(target, qt.Qobj) else np.asarray(target)
    if T.shape[1] != len(comp):
        T = T[:, comp]
    a0 = ctrl_to_array(ctrl if a_anchor is None else a_anchor)
    a_anchor_s = (a0 / self.scales[:, None]).ravel()
    mu = anchor / a_anchor_s.size

    def aligned_error(Uc):
        z = np.vdot(T.ravel(), Uc.ravel())
        rot = np.exp(-1j * np.angle(z)) if abs(z) > 0 else 1.0
        return rot, (rot * Uc - T).ravel()

    def evaluate(c):
        _, Ut = self.simulate_rollout(c)
        Uc = Ut.full()[:, comp]
        _, e = aligned_error(Uc)
        d = (ctrl_to_array(c) / self.scales[:, None]).ravel() - a_anchor_s
        cost = float(np.sum(np.abs(e) ** 2) + mu * np.sum(d ** 2))
        return cost, _aligned_cols_fidelity(Uc, T, comp)

    C, F = evaluate(ctrl)
    info = {"fidelity": [F], "cost": [C], "gain": []}
    for step in range(n_steps):
        Jc, Uc = self.compute_unitary_jacobian(ctrl)
        rot, e = aligned_error(Uc)
        Jr = np.vstack([(rot * Jc).real, (rot * Jc).imag])
        er = np.concatenate([e.real, e.imag])
        Ur, S, Vt = np.linalg.svd(Jr, full_matrices=False)
        r = min(r_rank, len(S))
        Ur, S, Vr = Ur[:, :r], S[:r], Vt[:r].T
        a = ctrl_to_array(ctrl)
        d_anchor = (a / self.scales[:, None]).ravel() - a_anchor_s
        c = -(S * (Ur.T @ er) + mu * (Vr.T @ d_anchor)) / (S ** 2 + mu + reg ** 2)
        da = (Vr @ c).reshape(6, -1) * self.scales[:, None]
        g = gain
        for _ in range(8):
            cand = array_to_ctrl(a + g * da, ctrl.dt)
            C_new, F_new = evaluate(cand)
            if C_new < C:
                break
            g /= 2
        else:
            break
        ctrl, C, F = cand, C_new, F_new
        info["fidelity"].append(F)
        info["cost"].append(C)
        info["gain"].append(g)
        if verbose:
            print(f"    plan step {step + 1}: model fidelity {F:.6f}, cost {C:.3e} (gain {g:.3g})")
        if abs(info["cost"][-2] - info["cost"][-1]) < tol:
            break
    return ctrl, info


def _svdilc_parameter_jacobian(self, ctrl, base_params, names, theta, fd_step=1e-4):
    """dy/dtheta at every checkpoint (central differences), shape (dim_y, p)."""
    cols = []
    for i in range(len(theta)):
        h = fd_step * max(1.0, abs(theta[i]))
        tp, tm = np.array(theta, float), np.array(theta, float)
        tp[i] += h
        tm[i] -= h
        yp, _ = self.simulate_rollout(ctrl, model=with_parameters(base_params, names, tp))
        ym, _ = self.simulate_rollout(ctrl, model=with_parameters(base_params, names, tm))
        cols.append((yp - ym) / (2 * h))
    return np.stack(cols, axis=1)


SVDILC.compute_unitary_jacobian = _svdilc_compute_unitary_jacobian
SVDILC.final_fidelity = _svdilc_final_fidelity
SVDILC.plan_final_unitary = _svdilc_plan_final_unitary
SVDILC.parameter_jacobian = _svdilc_parameter_jacobian


def run_final_unitary_ilc(ilc: SVDILC, ctrl0: Controls, U_target, base_params: SystemParams,
                          hardware_model: TwoQubitModel, fit_names, scan=None, prior_std=None,
                          n_iter=8, r_rank=8, n_plan_steps=10, anchor=1e-2, reg=1e-3, gain=1.0,
                          warm_start=False, n_times=4, required_times=None, shots=1000,
                          substeps=4, seed=0, verbose=True):
    """Hardware loop: measure -> fit theta -> plan for the final unitary on the fitted model.

    The hardware measurements (n_times checkpoints, shot noise) only identify the
    parameters theta; the pulse is then planned on the fitted model to reach
    U_target at the final time (plan_final_unitary), with no constraint on the
    intermediate trajectory. Measurement times are chosen for Fisher information
    about theta (parameter_jacobian), plus any required_times (e.g. [-1]).

    warm_start : False (default) re-plans from ctrl0 on every fitted model, so corrections
                 planned on earlier, wrong models are not carried over; True plans from
                 the current pulse (cheaper, but can stall on a plateau left by a wrong model).
    anchor     : pull toward ctrl0 during planning (0 = none).
    Returns (final Controls, log). Hardware truth is used only for the log.
    """
    rng = np.random.default_rng(seed)
    comp = ilc.model.computational_indices()
    T = U_target.full() if isinstance(U_target, qt.Qobj) else np.asarray(U_target)
    T = T[:, comp] if T.shape[1] != len(comp) else T
    theta_nom = np.array([get_param(base_params, n) for n in fit_names])
    theta = theta_nom.copy()

    def pick_times(c, model, th):
        ilc.model = model
        Jth = ilc.parameter_jacobian(c, base_params, fit_names, th)
        y, _ = ilc.simulate_rollout(c)
        pos, _ = ilc.select_measurement_times(Jth, n_times, y_nom=y, shots=shots, required=required_times)
        return pos

    model_nom = ilc.model
    pos = pick_times(ctrl0, model_nom, theta)
    log = {"F_hw": [], "F_model": [], "theta": [], "theta_std": [], "chi2": [], "times": [],
           "fit_names": list(fit_names)}
    c_k, hist = ctrl0, []
    for it in range(n_iter + 1):
        F_hw = ilc.final_fidelity(c_k, T, model=hardware_model)   # truth, for the log only
        log["F_hw"].append(F_hw)
        log["theta"].append(theta.copy())
        log["times"].append(ilc.checkpoints[pos] * ctrl0.dt)
        if verbose:
            print(f"iter {it}: hardware final-map fidelity {F_hw:.5f}  theta {np.round(theta, 4)}")
        if it == n_iter:
            break

        # 1. measure + Gauss-Newton parameter fit on all data so far
        y_meas, _ = ilc.simulate_rollout(c_k, model=hardware_model, substeps=substeps,
                                         shots=shots, rng=rng, times=pos)
        hist.append((c_k, pos, y_meas))
        theta, fit_model, info = ilc.fit_parameters(base_params, fit_names, theta, hist, shots=shots,
                                                    prior=theta_nom, prior_std=prior_std, scan=scan)
        log["theta_std"].append(info["theta_std"])
        log["chi2"].append(info["chi2_per_meas"])

        # 2. plan for the final unitary on the fitted model
        ilc.model = fit_model
        start = c_k if warm_start else ctrl0
        c_k, pinfo = ilc.plan_final_unitary(start, T, a_anchor=ctrl0, r_rank=r_rank,
                                            n_steps=n_plan_steps, reg=reg, anchor=anchor, gain=gain)
        log["F_model"].append(pinfo["fidelity"][-1])
        if verbose:
            print(f"   fit chi2/meas {info['chi2_per_meas']:.2f}"
                  + (" (grid scan)" if info["scanned"] else "")
                  + f" | planned model fidelity {pinfo['fidelity'][0]:.5f} -> {pinfo['fidelity'][-1]:.5f}"
                  + f" in {len(pinfo['fidelity']) - 1} steps")

        # 3. next measurement times: most informative about theta at the new pulse
        pos = pick_times(c_k, fit_model, theta)
    return c_k, log
