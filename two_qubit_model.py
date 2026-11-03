"""
Two flux-tunable, capacitively coupled transmons in QuTiP.

Model (frame rotating at each qubit's *idle* frequency, RWA):

    H(t)/2pi = sum_i [ Delta_i(t) n_i + (alpha_i/2) b_i^+ b_i^+ b_i b_i
                       + u_x,i(t)/2 (b_i + b_i^+) + u_y,i(t)/2 i(b_i^+ - b_i) ]
             + J ( e^{+i delta t} b_1^+ b_2 + e^{-i delta t} b_1 b_2^+ )

    Delta_i(t) = f_i(I(t)) - f_idle,i              (flux detuning, the "u_z" control)
    delta      = 2pi (f_idle,1 - f_idle,2)          (idle detuning between the qubits)

With n_levels=2 this reduces to the Pauli model:
    b -> sigma_-, (b + b^+) -> sigma_x, i(b^+ - b) -> sigma_y, and
    Delta n = (1/2) Delta sigma_z (+const) for sigma_z = |e><e| - |g><g|.
The anharmonicity term vanishes identically for 2 levels; use n_levels=3 to
capture leakage.

The coupling picks up the phase e^{i delta t} because the two qubits are in
frames rotating at *different* idle frequencies. That is what makes exchange
efficient only when flux brings them into resonance.

Flux / tuning curve
    Phi_i / Phi0 = sum_j M_ij I_j / Phi0 + phi_offset_i     (M includes crosstalk)
    f_i(phi)     = f_max,i * (cos^2(pi phi) + d_i^2 sin^2(pi phi))^(1/4)
For d = 0 (symmetric SQUID) this is f_max sqrt|cos(pi phi)|.

Units: time in ns, every frequency (f_max, alpha, J, Delta, u_x, u_y) in GHz
(cyclic; the 2pi is applied internally), currents in mA, mutual inductance in pH.
Drive check: a constant u_x of amplitude A for time T rotates by 2pi*A*T, so a
pi pulse has A*T = 0.5.

Controls are piecewise constant on N slices of width dt (GRAPE convention):
    I  : (2, N) flux-bias currents [mA]
    ux : (2, N) in-phase drive envelopes [GHz]
    uy : (2, N) quadrature drive envelopes [GHz]
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import qutip as qt
from scipy.linalg import expm

PHI0_WB = 2.067833848e-15            # flux quantum [Wb]
PHI0_PH_MA = PHI0_WB / 1e-15         # Phi0 expressed in pH*mA (= 1e-15 Wb) ~ 2.0678
TWO_PI = 2.0 * np.pi


# --------------------------------------------------------------------------- #
# Parameters and controls
# --------------------------------------------------------------------------- #
@dataclass
class QubitParams:
    f_max: float                 # sweet-spot (max) frequency [GHz]
    alpha: float = -0.200        # anharmonicity [GHz] (negative for transmons)
    asymmetry: float = 0.0       # SQUID junction asymmetry d (0 = symmetric)
    flux_offset: float = 0.0     # static flux offset [Phi0]


@dataclass
class SystemParams:
    q1: QubitParams
    q2: QubitParams
    J: float = 0.010                                  # coupling strength [GHz]
    M: np.ndarray = field(                            # mutual inductance [pH]
        default_factory=lambda: np.array([[2.0, 0.05],  # M[i, j]: flux line j -> SQUID i
                                          [0.05, 2.0]]))
    I_idle: np.ndarray = field(                       # idle bias currents [mA]
        default_factory=lambda: np.zeros(2))
    n_levels: int = 3                                 # 2 = qubits, 3+ = transmons w/ leakage

    def __post_init__(self):
        self.M = np.asarray(self.M, dtype=float).reshape(2, 2)
        self.I_idle = np.asarray(self.I_idle, dtype=float).reshape(2)
        if self.n_levels < 2:
            raise ValueError("n_levels must be >= 2")


@dataclass
class Controls:
    dt: float                    # slice width [ns]
    I: np.ndarray                # (2, N) currents [mA]
    ux: np.ndarray               # (2, N) [GHz]
    uy: np.ndarray               # (2, N) [GHz]

    def __post_init__(self):
        self.I = np.atleast_2d(np.asarray(self.I, dtype=float))
        self.ux = np.atleast_2d(np.asarray(self.ux, dtype=float))
        self.uy = np.atleast_2d(np.asarray(self.uy, dtype=float))
        shapes = {self.I.shape, self.ux.shape, self.uy.shape}
        if len(shapes) != 1 or self.I.shape[0] != 2:
            raise ValueError(f"I, ux, uy must all be shape (2, N); got {shapes}")

    @property
    def n_slices(self) -> int:
        return self.I.shape[1]

    @property
    def duration(self) -> float:
        return self.n_slices * self.dt

    @property
    def t_edges(self) -> np.ndarray:
        """Slice boundaries, length N+1."""
        return np.arange(self.n_slices + 1) * self.dt

    @property
    def t_mid(self) -> np.ndarray:
        """Slice midpoints, length N."""
        return (np.arange(self.n_slices) + 0.5) * self.dt

    @classmethod
    def idle(cls, model: "TwoQubitModel", n_slices: int, dt: float) -> "Controls":
        """All drives off, flux parked at the idle currents."""
        z = np.zeros((2, n_slices))
        return cls(dt, np.repeat(model.p.I_idle[:, None], n_slices, axis=1), z, z.copy())


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class TwoQubitModel:
    def __init__(self, params: SystemParams):
        self.p = params
        n = params.n_levels
        self.dims = [n, n]
        Id = qt.qeye(n)
        b = qt.destroy(n)
        self.b = [qt.tensor(b, Id), qt.tensor(Id, b)]
        self.n = [bi.dag() * bi for bi in self.b]

        qs = [params.q1, params.q2]
        self.f_max = np.array([q.f_max for q in qs])
        self.alpha = np.array([q.alpha for q in qs])
        self.asym = np.array([q.asymmetry for q in qs])
        self.phi_off = np.array([q.flux_offset for q in qs])

        self.f_idle = self.frequency(params.I_idle)
        self.delta = TWO_PI * (self.f_idle[0] - self.f_idle[1])   # rad/ns

        # Operators (all carry the 2pi so coefficients stay in GHz)
        self.H_anharm = sum(TWO_PI * a / 2 * bi.dag() * bi.dag() * bi * bi
                            for a, bi in zip(self.alpha, self.b))
        self.H_ex = TWO_PI * params.J * self.b[0].dag() * self.b[1]   # x e^{+i delta t}, + h.c.
        self.H_z = [TWO_PI * ni for ni in self.n]
        self.H_x = [TWO_PI * 0.5 * (bi + bi.dag()) for bi in self.b]
        self.H_y = [TWO_PI * 0.5 * 1j * (bi.dag() - bi) for bi in self.b]

    # ---- flux -> frequency -------------------------------------------------
    def flux(self, I) -> np.ndarray:
        """Reduced flux phi_i = Phi_i/Phi0 for currents I of shape (2,) or (2, N)."""
        I = np.asarray(I, dtype=float)
        phi = self.p.M @ I / PHI0_PH_MA
        return phi + (self.phi_off[:, None] if phi.ndim == 2 else self.phi_off)

    def _g(self, phi):
        d = self.asym[:, None] if np.ndim(phi) == 2 else self.asym
        return np.cos(np.pi * phi) ** 2 + d**2 * np.sin(np.pi * phi) ** 2

    def frequency(self, I) -> np.ndarray:
        """Qubit 0->1 frequencies [GHz] at currents I."""
        phi = self.flux(I)
        fmax = self.f_max[:, None] if phi.ndim == 2 else self.f_max
        return fmax * self._g(phi) ** 0.25

    def detuning(self, I) -> np.ndarray:
        """Delta_i = f_i(I) - f_idle,i [GHz]."""
        f = self.frequency(I)
        return f - (self.f_idle[:, None] if f.ndim == 2 else self.f_idle)

    def detuning_jacobian(self, I) -> np.ndarray:
        """dDelta_i/dI_j [GHz/mA]; shape (2, 2) or (2, 2, N).

        Chain rule for optimizers that work in current space. Note it is zero
        at a sweet spot (phi = 0), so a gradient in I vanishes when parked there.
        """
        phi = self.flux(I)
        two_d = phi.ndim == 2
        fmax = self.f_max[:, None] if two_d else self.f_max
        d = self.asym[:, None] if two_d else self.asym
        g = self._g(phi)
        dg = np.pi * np.sin(2 * np.pi * phi) * (d**2 - 1)
        df_dphi = fmax * 0.25 * g ** (-0.75) * dg                  # (2,) or (2,N)
        dphi_dI = self.p.M / PHI0_PH_MA                           # (2,2)
        if two_d:
            return df_dphi[:, None, :] * dphi_dI[:, :, None]
        return df_dphi[:, None] * dphi_dI

    def currents_for_frequencies(self, f_target, branch=None) -> np.ndarray:
        """Invert the tuning curves (with crosstalk) -> currents [mA] giving f_target.

        branch: sign of phi to pick for each qubit (defaults to the idle side).
        """
        f_target = np.asarray(f_target, dtype=float)
        d2 = self.asym**2
        g = (f_target / self.f_max) ** 4
        cos2 = (g - d2) / (1 - d2)
        if np.any((cos2 < 0) | (cos2 > 1)):
            raise ValueError("target frequency outside tuning range")
        phi_mag = np.arccos(np.sqrt(cos2)) / np.pi
        if branch is None:
            branch = np.sign(self.flux(self.p.I_idle))
            branch[branch == 0] = 1.0
        phi = np.asarray(branch) * phi_mag
        return np.linalg.solve(self.p.M / PHI0_PH_MA, phi - self.phi_off)

    # ---- Hamiltonian construction ----------------------------------------
    def hamiltonian(self, ctrl: Controls) -> qt.QobjEvo:
        """Time-dependent H(t) as a QobjEvo with piecewise-constant controls."""
        Delta = self.detuning(ctrl.I)
        N, dt = ctrl.n_slices, ctrl.dt

        def pwc(values):
            v = np.asarray(values)
            return lambda t: v[min(max(int(t // dt), 0), N - 1)]

        terms = [self.H_anharm]
        if self.p.J != 0:
            if self.delta == 0:
                terms.append(self.H_ex + self.H_ex.dag())
            else:
                dlt = self.delta
                terms.append([self.H_ex, lambda t: np.exp(1j * dlt * t)])
                terms.append([self.H_ex.dag(), lambda t: np.exp(-1j * dlt * t)])
        for i in range(2):
            terms.append([self.H_z[i], pwc(Delta[i])])
            terms.append([self.H_x[i], pwc(ctrl.ux[i])])
            terms.append([self.H_y[i], pwc(ctrl.uy[i])])
        return qt.QobjEvo(terms)

    def slice_hamiltonians(self, ctrl: Controls) -> np.ndarray:
        """Dense H_k for each slice, shape (N, D, D) [rad/ns].

        The coupling phase is evaluated at each slice midpoint. That is exact
        when delta = 0 and accurate when |delta| * dt << 1 (a warning is issued
        otherwise). Use this for fast GRAPE rollouts and gradients.
        """
        if abs(self.delta) * ctrl.dt > 0.1:
            import warnings
            warnings.warn(f"|delta|*dt = {abs(self.delta) * ctrl.dt:.2f}; "
                          "use smaller dt or evolve() for accuracy")
        Delta = self.detuning(ctrl.I)
        A = self.H_anharm.full()
        Ex = self.H_ex.full()
        ops = [(self.H_z[i].full(), self.H_x[i].full(), self.H_y[i].full()) for i in range(2)]
        ph = np.exp(1j * self.delta * ctrl.t_mid)
        H = np.empty((ctrl.n_slices, *A.shape), dtype=complex)
        for k in range(ctrl.n_slices):
            Hk = A + ph[k] * Ex + np.conj(ph[k]) * Ex.conj().T
            for i, (Hz, Hx, Hy) in enumerate(ops):
                Hk = Hk + Delta[i, k] * Hz + ctrl.ux[i, k] * Hx + ctrl.uy[i, k] * Hy
            H[k] = Hk
        return H

    def control_operators(self) -> dict[str, list[qt.Qobj]]:
        """Control Hamiltonians, each multiplying a GHz-valued coefficient.

        For GRAPE: dH_k/du_{x,i} = H_x[i], dH_k/du_{y,i} = H_y[i],
        dH_k/dI_j = sum_i dDelta_i/dI_j * H_z[i]  (see detuning_jacobian).
        """
        return {"z": self.H_z, "x": self.H_x, "y": self.H_y}

    # ---- rollout ---------------------------------------------------------
    def slice_propagators(self, ctrl: Controls) -> np.ndarray:
        """U_k = expm(-i H_k dt), shape (N, D, D)."""
        return np.stack([expm(-1j * Hk * ctrl.dt) for Hk in self.slice_hamiltonians(ctrl)])

    def propagator(self, ctrl: Controls, method: str = "pwc") -> qt.Qobj:
        """Total unitary U(T). method='pwc' (fast slice product) or 'ode' (QuTiP)."""
        if method == "pwc":
            U = np.eye(np.prod(self.dims), dtype=complex)
            for Uk in self.slice_propagators(ctrl):
                U = Uk @ U
            return qt.Qobj(U, dims=[self.dims, self.dims])
        if method == "ode":
            return qt.propagator(self.hamiltonian(ctrl), [0, ctrl.duration],
                                 options={"max_step": ctrl.dt})[-1]
        raise ValueError("method must be 'pwc' or 'ode'")

    def evolve(self, ctrl: Controls, state0: qt.Qobj, tlist=None, c_ops=None,
               e_ops=None, **options):
        """Integrate with sesolve (no c_ops, ket) or mesolve. Returns a QuTiP Result."""
        tlist = ctrl.t_edges if tlist is None else tlist
        opts = {"max_step": ctrl.dt, "store_states": True, **options}
        H = self.hamiltonian(ctrl)
        if not c_ops and state0.isket:
            return qt.sesolve(H, state0, tlist, e_ops=e_ops, options=opts)
        return qt.mesolve(H, state0, tlist, c_ops=c_ops or [], e_ops=e_ops, options=opts)

    def collapse_ops(self, T1=(np.inf, np.inf), Tphi=(np.inf, np.inf)) -> list[qt.Qobj]:
        """Amplitude damping and pure dephasing [ns]. Tphi enters as sqrt(2/Tphi) n."""
        c = []
        for i in range(2):
            if np.isfinite(T1[i]):
                c.append(np.sqrt(1 / T1[i]) * self.b[i])
            if np.isfinite(Tphi[i]):
                c.append(np.sqrt(2 / Tphi[i]) * self.n[i])
        return c

    # ---- states and figures of merit --------------------------------------
    def basis(self, n1: int, n2: int) -> qt.Qobj:
        return qt.tensor(qt.basis(self.p.n_levels, n1), qt.basis(self.p.n_levels, n2))

    def computational_indices(self) -> list[int]:
        n = self.p.n_levels
        return [i * n + j for i in (0, 1) for j in (0, 1)]

    def project_computational(self, U: qt.Qobj) -> np.ndarray:
        """4x4 block of U in the {00, 01, 10, 11} subspace (non-unitary if leaking)."""
        idx = self.computational_indices()
        return U.full()[np.ix_(idx, idx)]

    def gate_fidelity(self, U: qt.Qobj, target) -> float:
        """Average gate fidelity on the computational subspace, leakage-aware.

        F = (Tr(M M^+) + |Tr M|^2) / (d (d + 1)),  M = target^+ U_sub,  d = 4.
        Note: phases are in the idle rotating frame; virtual-Z corrections are
        not applied here.
        """
        Ut = target.full() if isinstance(target, qt.Qobj) else np.asarray(target)
        Mm = Ut.conj().T @ self.project_computational(U)
        d = 4
        return float((np.trace(Mm @ Mm.conj().T).real + abs(np.trace(Mm)) ** 2) / (d * (d + 1)))
