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
#
# Everything lives inside functions: closures over non-const globals are
# type-unstable in Julia and make the ForwardDiff'd NonlinearDrive very slow.

using Piccolo
using JSON3
using LinearAlgebra
using SparseArrays
using Random

const ROOT = normpath(joinpath(@__DIR__, ".."))
const CONTROL_NAMES = ["I1", "I2", "ux1", "ux2", "uy1", "uy2"]

tomat(m) = ComplexF64[m.re[i][j] + im * m.im[i][j] for i = 1:length(m.re), j = 1:length(m.re[1])]

"""Tuning-curve parameters, captured by value in the drive closures."""
struct Tuning
    f_max::Vector{Float64}
    asym::Vector{Float64}
    phi_off::Vector{Float64}
    dphi_dI::Matrix{Float64}      # [1/mA], = M / Phi0
    f_idle::Vector{Float64}
end

# Delta_i(I) = f_i(I) - f_idle,i  [GHz]
function detuning(tc::Tuning, u, i)
    phi = tc.dphi_dI[i, 1] * u[1] + tc.dphi_dI[i, 2] * u[2] + tc.phi_off[i]
    g = cos(π * phi)^2 + tc.asym[i]^2 * sin(π * phi)^2
    return tc.f_max[i] * g^(1 / 4) - tc.f_idle[i]
end

function currents_for_frequencies(tc::Tuning, f_target)
    cos2 = ((f_target ./ tc.f_max) .^ 4 .- tc.asym .^ 2) ./ (1 .- tc.asym .^ 2)
    phi = acos.(sqrt.(cos2)) ./ π
    return tc.dphi_dI \ (phi .- tc.phi_off)
end

function build_system(d; I_frac, u_max)
    H_anharm = tomat(d.ops.H_anharm)
    H_ex = tomat(d.ops.H_ex)
    H_z = [tomat(m) for m in d.ops.H_z]
    H_x = [tomat(m) for m in d.ops.H_x]
    H_y = [tomat(m) for m in d.ops.H_y]

    tc = Tuning(
        Float64.(d.f_max),
        Float64.(d.asymmetry),
        Float64.(d.flux_offset),
        Matrix(Float64.(reduce(hcat, d.M_pH))') ./ Float64(d.phi0_pH_mA),
        Float64.(d.f_idle),
    )
    δ = Float64(d.delta)

    drift = δ == 0 ? Any[H_anharm + H_ex + H_ex'] :
            Any[H_anharm,
                (H_ex + H_ex') => t -> cos(δ * t),
                (im * (H_ex - H_ex')) => t -> sin(δ * t)]

    drives = AbstractDrive[
        NonlinearDrive(H_z[1], u -> detuning(tc, u, 1); active_controls = [1, 2]),
        NonlinearDrive(H_z[2], u -> detuning(tc, u, 2); active_controls = [1, 2]),
        LinearDrive(sparse(H_x[1]), 3),
        LinearDrive(sparse(H_x[2]), 4),
        LinearDrive(sparse(H_y[1]), 5),
        LinearDrive(sparse(H_y[2]), 6),
    ]

    I_max = I_frac / maximum(abs.(diag(tc.dphi_dI)))   # keep |phi| ≤ I_frac
    bounds = [(-I_max, I_max), (-I_max, I_max), fill((-u_max, u_max), 4)...]
    return QuantumSystem(drift, drives, bounds), tc, I_max
end

"""Flux pulse to the |11>-|20> resonance (f1 = f2 - alpha1), sin² ramps, one 2π cycle."""
function initial_guess(d, tc, times; ramp)
    T = times[end]
    alpha = Float64.(d.alpha)
    I_on = currents_for_frequencies(tc, [tc.f_idle[2] - alpha[1], tc.f_idle[2]])
    I_off = Float64.(d.I_idle)
    env(t) = t < ramp ? sin(π * t / (2ramp))^2 : t > T - ramp ? sin(π * (T - t) / (2ramp))^2 : 1.0
    u0 = zeros(6, length(times))
    for (k, t) in enumerate(times)
        u0[1:2, k] .= I_off .+ env(t) .* (I_on .- I_off)
    end
    u0[3:6, :] .= 1e-3 .* randn(4, length(times))
    return u0, I_on
end

function write_result(outfile, qcp, F)
    traj = get_trajectory(qcp)
    u = Matrix(traj[:u])
    ts = collect(get_times(traj))
    U = iso_vec_to_operator(traj[:Ũ⃗][:, end])
    mkpath(dirname(outfile))
    open(outfile, "w") do io
        JSON3.write(io, Dict(
            "control_names" => CONTROL_NAMES,
            "times" => ts,
            "u" => [u[i, :] for i = 1:size(u, 1)],
            "fidelity" => F,
            "gate" => "CZ",
            "U_final" => Dict("re" => [real.(U[i, :]) for i = 1:size(U, 1)],
                              "im" => [imag.(U[i, :]) for i = 1:size(U, 1)]),
            "piccolo_version" => string(pkgversion(Piccolo)),
        ))
    end
end

function main(args)
    pos = filter(a -> !occursin('=', a), args)
    kw = Dict(Pair(split(a, '=', limit = 2)...) for a in args if occursin('=', a))
    opt(k, default) = haskey(kw, k) ? parse(typeof(default), kw[k]) : default

    sysfile = length(pos) ≥ 1 ? pos[1] : joinpath(ROOT, "data", "system.json")
    outfile = length(pos) ≥ 2 ? pos[2] : joinpath(ROOT, "data", "cz_pulse.json")

    d = JSON3.read(read(sysfile, String))
    nlev = Int(d.n_levels)
    J = Float64(d.J)
    Random.seed!(opt("seed", 1234))

    sys, tc, I_max = build_system(d; I_frac = opt("I_frac", 0.35), u_max = opt("u_max", 0.05))
    U_goal = EmbeddedOperator(GATES[:CZ], [1, 2], [1:2, 1:2], [nlev, nlev])

    ramp = opt("ramp", 5.0)
    T = opt("T", 1 / (2 * sqrt(2) * J) + 2 * ramp)
    N = opt("N", 451)
    times = collect(range(0.0, T, length = N))
    u0, I_on = initial_guess(d, tc, times; ramp)

    println("T = $(round(T, digits=2)) ns, N = $N knots, dt = $(round(T/(N-1), digits=4)) ns")
    println("I_on = $(round.(I_on, digits=4)) mA,  |I| ≤ $(round(I_max, digits=3)) mA")

    qtraj = UnitaryTrajectory(sys, ZeroOrderPulse(u0, times), U_goal)
    qcp = SmoothPulseProblem(
        qtraj,
        N;
        Q = opt("Q", 100.0),
        R = opt("R", 1e-2),
        ddu_bound = opt("ddu_bound", 1e-3),
        piccolo_options = PiccoloOptions(timesteps_all_equal = true),
    )
    t0 = time()
    solve!(qcp; max_iter = opt("max_iter", 200))
    println("solve wall time: $(round(time() - t0, digits=1)) s")

    F = fidelity(qcp)
    println("Piccolo fidelity (ODE re-rollout): $F")
    write_result(outfile, qcp, F)
    println("wrote $outfile")
end

abspath(PROGRAM_FILE) == @__FILE__ && main(ARGS)
