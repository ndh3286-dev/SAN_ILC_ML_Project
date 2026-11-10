# Optimize a CZ gate for the two-tunable-transmon model with Piccolo.jl.
#
#   julia --project=piccolo piccolo/solve_cz.jl [system.json] [out.json] [key=value ...]
#
# system.json is written by TwoQubitModel.export_piccolo_system() so the
# Hamiltonian here is *identical* to two_qubit_model.py (frame rotating at each
# qubit's idle frequency, time in ns, H in rad/ns):
#
#   H(u, t) = H_anharm + cos(δt)(H_ex + H_ex†) + sin(δt) i(H_ex - H_ex†)
#           + Σ_i Δ_i(I1, I2) H_z[i] + ux_i H_x[i] + uy_i H_y[i]
#
# Controls u = [I1, I2, ux1, ux2, uy1, uy2]: the flux-bias currents [mA] enter
# through the tuning curve (NonlinearDrive, crosstalk included), and the XY
# envelopes [GHz] enter linearly. The result is written as JSON for
# TwoQubitModel.load_piccolo_result().
#
# Options (key=value): T (ns), N (knots), ramp (ns), max_iter, Q, R, ddu_bound,
# I_frac (max |flux| in Phi0), u_max (GHz drive bound), seed.

using Piccolo
using JSON3
using LinearAlgebra
using SparseArrays
using Random

const ROOT = normpath(joinpath(@__DIR__, ".."))
pos = filter(a -> !occursin('=', a), ARGS)
kw = Dict(Pair(split(a, '=', limit = 2)...) for a in ARGS if occursin('=', a))
opt(k, default) = haskey(kw, k) ? parse(typeof(default), kw[k]) : default

sysfile = length(pos) ≥ 1 ? pos[1] : joinpath(ROOT, "data", "system.json")
outfile = length(pos) ≥ 2 ? pos[2] : joinpath(ROOT, "data", "cz_pulse.json")

# --------------------------------------------------------------------------- #
# Load the model exported from Python
# --------------------------------------------------------------------------- #
d = JSON3.read(read(sysfile, String))
tomat(m) = ComplexF64[m.re[i][j] + im * m.im[i][j] for i = 1:length(m.re), j = 1:length(m.re[1])]

H_anharm = tomat(d.ops.H_anharm)
H_ex = tomat(d.ops.H_ex)
H_z = [tomat(m) for m in d.ops.H_z]
H_x = [tomat(m) for m in d.ops.H_x]
H_y = [tomat(m) for m in d.ops.H_y]

nlev = Int(d.n_levels)
f_max = Float64.(d.f_max)
alpha = Float64.(d.alpha)
asym = Float64.(d.asymmetry)
phi_off = Float64.(d.flux_offset)
Minv_phi = Float64.(reduce(hcat, d.M_pH)') ./ Float64(d.phi0_pH_mA)  # dphi_i/dI_j [1/mA]
f_idle = Float64.(d.f_idle)
δ = Float64(d.delta)
J = Float64(d.J)

# Tuning curve: Delta_i(I) = f_i(I) - f_idle,i  [GHz]
function detuning(u, i)
    phi = Minv_phi[i, 1] * u[1] + Minv_phi[i, 2] * u[2] + phi_off[i]
    g = cos(π * phi)^2 + asym[i]^2 * sin(π * phi)^2
    return f_max[i] * g^(1 / 4) - f_idle[i]
end

function currents_for_frequencies(f_target)
    cos2 = ((f_target ./ f_max) .^ 4 .- asym .^ 2) ./ (1 .- asym .^ 2)
    phi = acos.(sqrt.(cos2)) ./ π
    return Minv_phi \ (phi .- phi_off)
end

# --------------------------------------------------------------------------- #
# Quantum system
# --------------------------------------------------------------------------- #
drift = δ == 0 ? Any[H_anharm + H_ex + H_ex'] :
        Any[H_anharm, (H_ex + H_ex') => t -> cos(δ * t), (im * (H_ex - H_ex')) => t -> sin(δ * t)]

drives = AbstractDrive[
    NonlinearDrive(H_z[1], u -> detuning(u, 1); active_controls = [1, 2]),
    NonlinearDrive(H_z[2], u -> detuning(u, 2); active_controls = [1, 2]),
    LinearDrive(sparse(H_x[1]), 3),
    LinearDrive(sparse(H_x[2]), 4),
    LinearDrive(sparse(H_y[1]), 5),
    LinearDrive(sparse(H_y[2]), 6),
]

I_frac = opt("I_frac", 0.35)                       # keep |phi| ≤ I_frac (away from 0.5 Phi0)
I_max = I_frac / maximum(abs.(diag(Minv_phi)))
u_max = opt("u_max", 0.05)
bounds = [(-I_max, I_max), (-I_max, I_max), fill((-u_max, u_max), 4)...]
sys = QuantumSystem(drift, drives, bounds)

# CZ on the {0,1}⊗{0,1} subspace of the (nlev × nlev) system
U_goal = EmbeddedOperator(GATES[:CZ], [1, 2], [1:2, 1:2], [nlev, nlev])

# --------------------------------------------------------------------------- #
# Initial guess: tune qubit 1 to the |11>-|20> resonance (f1 = f2 - alpha1)
# with sin² ramps, hold for one full 2π cycle at the √2 J exchange rate.
# --------------------------------------------------------------------------- #
Random.seed!(opt("seed", 1234))
t_hold = 1 / (2 * sqrt(2) * J)
ramp = opt("ramp", 5.0)
T = opt("T", t_hold + 2 * ramp)
N = opt("N", 451)
times = collect(range(0.0, T, length = N))

I_on = currents_for_frequencies([f_idle[2] - alpha[1], f_idle[2]])
I_off = Float64.(d.I_idle)
env(t) = t < ramp ? sin(π * t / (2ramp))^2 : t > T - ramp ? sin(π * (T - t) / (2ramp))^2 : 1.0
u0 = zeros(6, N)
for (k, t) in enumerate(times)
    u0[1:2, k] .= I_off .+ env(t) .* (I_on .- I_off)
end
u0[3:6, :] .= 1e-3 .* randn(4, N)

pulse = ZeroOrderPulse(u0, times)
qtraj = UnitaryTrajectory(sys, pulse, U_goal)

println("T = $(round(T, digits=2)) ns, N = $N knots, dt = $(round(T/(N-1), digits=4)) ns")
println("I_on = $(round.(I_on, digits=4)) mA,  |I| ≤ $(round(I_max, digits=3)) mA,  |u| ≤ $u_max GHz")

qcp = SmoothPulseProblem(
    qtraj,
    N;
    Q = opt("Q", 100.0),
    R = opt("R", 1e-2),
    ddu_bound = opt("ddu_bound", 1e-3),
    piccolo_options = PiccoloOptions(timesteps_all_equal = true),
)
solve!(qcp; max_iter = opt("max_iter", 200))

F = fidelity(qcp)
println("Piccolo fidelity: $F")

# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #
traj = get_trajectory(qcp)
u = Matrix(traj[:u])
ts = collect(get_times(traj))
U_final = iso_vec_to_operator(traj[:Ũ⃗][:, end])

mkpath(dirname(outfile))
open(outfile, "w") do io
    JSON3.write(io, Dict(
        "control_names" => ["I1", "I2", "ux1", "ux2", "uy1", "uy2"],
        "times" => ts,
        "u" => [u[i, :] for i = 1:size(u, 1)],
        "fidelity" => F,
        "gate" => "CZ",
        "U_final" => Dict("re" => [real.(U_final[i, :]) for i = 1:size(U_final, 1)],
                          "im" => [imag.(U_final[i, :]) for i = 1:size(U_final, 1)]),
        "piccolo_version" => string(pkgversion(Piccolo)),
    ))
end
println("wrote $outfile")
