# ---------------------------------------------------------------------------
# s3_inclass_solution.jl — Session 3, Phases 1 to 3. Complete version.
#
# The same reservoir, and the stock is now a continuous quantity. Three parts.
#
#   Phase 1. Thirty years written out as one optimization problem, solved in
#            JuMP with Ipopt. The shadow price of water comes out of the duals.
#   Phase 2. The infinite horizon on a fine grid, with linear interpolation
#            between the nodes and Optim's one-line optimizer at each node.
#   Phase 3. Random rainfall, ε ∈ {1, 3} with equal odds. Three lines change.
#
# Model and calibration from Miranda and Fackler, Sections 7.2.5 and 7.6.5; the
# steady state conditions are from Section 8.4.4. Section 6.8 solves this model
# by collocation; we use linear interpolation instead for simplicity.
# ---------------------------------------------------------------------------
using JuMP, Ipopt, Optim, Plots

# ── Setup (given) ────────────────────────────────────────────────────────
α_1 = 14.0      # farmer benefit scale
β_1 = 0.8       # farmer benefit curvature
α_2 = 10.0      # recreation benefit scale
β_2 = 0.4       # recreation benefit curvature
δ   = 0.9       # discount factor
M   = 30.0      # reservoir capacity, units of water
ε_rain = 2.0    # rainfall, the same every year until Phase 3

F(x)  = α_1 * x^β_1
U(z)  = α_2 * z^β_2
dF(x) = α_1 * β_1 * x^(β_1 - 1)       # F'(x)
dU(z) = α_2 * β_2 * z^(β_2 - 1)       # U'(z)

# The analytic steady state of this model (Section 8.4.4): the release equals the
# rain, the shadow price is F' there, and the stock solves U'(s - x) = (1 - δ) F'(x).
x_star = ε_rain
λ_star = dF(x_star)
s_star = x_star + ((1 - δ) * λ_star / (α_2 * β_2))^(1 / (β_2 - 1))
println("analytic steady state: x* = ", x_star, "   λ* = ", round(λ_star, digits = 6),
        "   s* = ", round(s_star, digits = 6))

# ═══ PHASE 1. Finite horizon in JuMP, and the shadow price ═══════════════

# ── 1.1 The whole horizon as one problem ─────────────────────────────────
# Choose x_1, ..., x_T and s_1, ..., s_{T+1} to maximize
#     Σ_t δ^(t-1) [ F(x_t) + U(z_t) ]     with  z_t = s_t - x_t,  s_{t+1} = z_t + ε,
#     s_1 given, 0 ≤ s_t ≤ M, and nothing after T (V_{T+1} = 0).
# z_t is declared as a VARIABLE with a positive lower bound rather than written as
# the expression s_t - x_t. Ipopt evaluates the objective at points that violate
# constraints but never at points that violate bounds, and z^0.4 is not defined
# for z < 0. With z bounded, the objective only ever sees z inside its bounds.
function solve_finite_horizon(T, s_1)
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    @variable(model, x[1:T] >= 0, start = 2.0)          # releases
    @variable(model, z[1:T] >= 0, start = 2.0)          # water left for recreation
    @variable(model, 0 <= s[1:T+1] <= M, start = 8.0)      # stocks
    @constraint(model, initial, s[1] == s_1)
    # TODO 1. Two lines. The split of the stock, z[t] == s[t] - x[t], as a
    #         constraint named split, one per year. Then the objective: the
    #         discounted sum of F(x[t]) + U(z[t]) over t = 1:T, written with
    #         α_1, β_1, α_2, β_2 directly.
    #         @constraint(model, split[t = 1:T], ...)
    #         @objective(model, Max, sum(... for t in 1:T))
    @constraint(model, split[t = 1:T], z[t] == s[t] - x[t])
    @objective(model, Max, sum(δ^(t - 1) * (α_1 * x[t]^β_1 + α_2 * z[t]^β_2) for t in 1:T))
    @constraint(model, transition[t = 1:T], s[t + 1] == z[t] + ε_rain)
    optimize!(model)
    @assert is_solved_and_feasible(model)
    # The multiplier on each stock: the initial condition pins s_1, the transition
    # constraint of year t pins s_{t+1}. We collect them in one vector of T + 1.
    μ = zeros(T + 1)
    μ[1] = dual(initial)
    for t in 1:T
        μ[t + 1] = dual(transition[t])
    end
    return value.(x), value.(s), objective_value(model), μ
end

T = 30
s_1 = 4.0
x_path, s_path, W_T, μ = solve_finite_horizon(T, s_1)
println("T = ", T, ", s_1 = ", s_1, ":  objective = ", round(W_T, digits = 4))
println("x_t = ", round.(x_path, digits = 3))
println("s_t = ", round.(s_path, digits = 3))

# ── 1.2 Plot (given) ─────────────────────────────────────────────────────
plot(1:T, x_path; xlabel = "year", ylabel = "x_t, released", label = false, marker = :circle)
savefig("s3_finite_x.png")
plot(1:T+1, s_path; xlabel = "year", ylabel = "s_t, water in the reservoir", label = "finite-horizon solution", marker = :circle)
hline!([s_star]; linestyle = :dash, label = "infinite-horizon steady state")
savefig("s3_finite_s.png")

# ── 1.3 The shadow price, read off the duals (given) ─────────────────────
# λ_t is the value of one more unit of water at the start of year t, in year-t
# units. JuMP reports the multiplier in the sign convention of its own standard
# form and discounted to year 1, so we undo that with: λ_t = -μ_t / δ^(t-1).
# Check it: the Euler equation says λ_t = F'(x_t).
λ_path = zeros(T + 1)
for t in 1:T+1
    λ_path[t] = -μ[t] / δ^(t - 1)
end
println("λ_1 = ", round(λ_path[1], digits = 6), "   F'(x_1) = ", round(dF(x_path[1]), digits = 6))
println("largest |λ_t - F'(x_t)| over t = 1..T: ", maximum(abs.(λ_path[1:T] .- dF.(x_path))))

# ── 1.4 The last year is Session 1 (given) ───────────────────────────────
# Nothing after T, so λ_{T+1} = 0 and the last year's condition is the static
# one: F'(x_T) = U'(s_T - x_T). This is the drawdown in Session 1, but 30 years late.
println("λ_{T+1} = ", λ_path[T + 1], "   (should be zero)")
println("F'(x_T) = ", round(dF(x_path[T]), digits = 6),
        "   U'(s_T - x_T) = ", round(dU(s_path[T] - x_path[T]), digits = 6))
println("x_1 = ", round(x_path[1], digits = 3), " here, against x*(4) = 1 on the integer grid of Session 2")

# ═══ PHASE 2. Infinite horizon, continuous state ═════════════════════════

# ── 2.1 The grid, and a function between the nodes ───────────────────────
# V is a vector again, one number per node, but now it stands for a function on
# [0, M]: between two nodes we draw a straight line. Section 6.8 of the book does
# this properly with a basis of smooth functions (collocation); we do not have
# time to cover it, but we can measure the error of using a straight line.
function interp(v, Δ, s)
    i = floor(Int, s / Δ) + 1             # the node just below s (Julia counts from 1)
    if i >= length(v)
        return v[end]
    end
    # TODO 2. Draw the straight line through the nodes i and i + 1, point and slope:
    #         s_i is the stock at node i, the slope is rise over run between the
    #         two nodes, and the answer is the point on that line at s.
    #         s_i = ...
    #         slope = ...
    #         return ...
    s_i = (i - 1) * Δ                     # the stock at node i
    slope = (v[i + 1] - v[i]) / Δ         # rise over run between the two nodes
    return v[i] + slope * (s - s_i)
end

# ── 2.2 The continuation value, deterministic (given) ────────────────────
# What is left after the release, z = s - x, plus the rain, capped at M.
continuation_certain(v, Δ, z) = interp(v, Δ, min(z + ε_rain, M))

# ── 2.3 One node: the Bellman update with Optim (given) ──────────────────
# At stock s, choose x in (0, s) to maximize F(x) + U(s - x) + δ E V(s'). This is
# Session 1's one-line optimizer with one more term in the objective.
function solve_node(s, v, Δ, continuation, δ)
    if s < 1e-8
        return 0.0, δ * continuation(v, Δ, 0.0)          # nothing to release
    end
    result = optimize(x -> -(F(x) + U(s - x) + δ * continuation(v, Δ, s - x)), 1e-9, s - 1e-9)
    return Optim.minimizer(result), -Optim.minimum(result)
end

# ── 2.4 Function iteration on the grid (given) ───────────────────────────
function value_iteration(continuation, Δ; δ = δ, M = M, tol = 1e-9, maxit = 2000)
    grid = collect(0.0:Δ:M)
    n = length(grid)
    v_old = zeros(n)
    v_new = zeros(n)
    x_pol = zeros(n)
    for it in 1:maxit
        for i in 1:n
            x_pol[i], v_new[i] = solve_node(grid[i], v_old, Δ, continuation, δ)
        end
        change = maximum(abs.(v_new .- v_old))
        v_old = copy(v_new)
        if change < tol
            return v_new, x_pol, grid, it
        end
    end
    println("value_iteration did not converge in ", maxit, " iterations")
    return v_new, x_pol, grid, maxit
end

Δ = 0.01
println("Function iteration on ", length(0.0:Δ:M), " nodes, about 15 seconds...")
v_det, x_det, grid, it_det = value_iteration(continuation_certain, Δ)
println("converged in ", it_det, " iterations")

# ── 2.5 Verify against the analytic steady state (given) ─────────────────
node(s, Δ) = round(Int, s / Δ) + 1                    # index of the node nearest s
x_at_star = interp(x_det, Δ, s_star)
println("x(s*) = ", round(x_at_star, digits = 6), "   analytic x* = ", x_star)
println("λ(s*) = F'(x(s*)) = ", round(dF(x_at_star), digits = 6), "   analytic λ* = ", round(λ_star, digits = 6))
println("V(s*) = ", round(interp(v_det, Δ, s_star), digits = 2),
        "   analytic V(s*) ≈ (F(x*) + U(s* - x*)) / (1 - δ) = ", round((F(x_star) + U(s_star - x_star)) / (1 - δ), digits = 2))

# ── 2.6 Grid error (given) ───────────────────────────────────────────────
# Follow the policy from s_1 = 4 until the stock stops moving. On the grid the
# resting point is not exactly s*: the gap is smaller than one grid step, and it
# halves when the step halves. (Δ = 0.005 gives 12.515 and takes about 30 seconds.)
function simulate_continuous(x_pol, Δ, s_1, T, continuation_rain)
    path = zeros(T)
    path[1] = s_1
    for t in 1:(T - 1)
        x = interp(x_pol, Δ, path[t])
        path[t + 1] = min(path[t] - x + continuation_rain(), M)
    end
    return path
end
rain_certain() = ε_rain
long_run_01 = simulate_continuous(x_det, Δ, s_1, 200, rain_certain)[end]
v_coarse, x_coarse, grid_coarse, it_coarse = value_iteration(continuation_certain, 0.02)
long_run_02 = simulate_continuous(x_coarse, 0.02, s_1, 200, rain_certain)[end]
println("long-run stock:  Δ = 0.02 gives ", round(long_run_02, digits = 4),
        "   Δ = 0.01 gives ", round(long_run_01, digits = 4),
        "   analytic ", round(s_star, digits = 4))

# ── 2.7 Overlay Session 2's staircase (given) ────────────────────────────
policy_s2 = [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5]
plot(grid, x_det; xlabel = "s, water in the reservoir", ylabel = "x(s), released",
     label = "continuous state", linewidth = 2)
scatter!(0:30, policy_s2; label = "Session 2, integer grid", marker = :circle)
vline!([s_star]; linestyle = :dash, label = false)
savefig("s3_policy_overlay.png")

# ═══ PHASE 3. Uncertainty ════════════════════════════════════════════════

# ── 3.1 Two-state rainfall (blank 3) ─────────────────────────────────────
# Rain is 1 or 3, each with probability one half. Same mean as before. The only
# change to Phase 2 is the continuation value, which becomes an expectation.
ε_values = [1.0, 3.0]
p = [0.5, 0.5]
function continuation_random(v, Δ, z)
    # TODO 3. E V(s') = Σ_q p[q] V(min(z + ε_values[q], M)). Loop over the two
    #         rainfall states, interpolate V at each next stock, weight by p.
    expected = 0.0
    for q in 1:length(ε_values)
        expected = expected + p[q] * interp(v, Δ, min(z + ε_values[q], M))
    end
    return expected
end

println("Function iteration with random rain, about 15 seconds...")
v_sto, x_sto, grid, it_sto = value_iteration(continuation_random, Δ)
println("converged in ", it_sto, " iterations")

# ── 3.2 Compare the policies (given) ─────────────────────────────────────
for s in (4.0, 8.0, 12.0, s_star, 20.0, 30.0)
    println("  x(", rpad(round(s, digits = 3), 7), ")  deterministic ", rpad(round(interp(x_det, Δ, s), digits = 4), 7),
            "  random rain ", round(interp(x_sto, Δ, s), digits = 4))
end
plot(grid, x_det; xlabel = "s, water in the reservoir", ylabel = "x(s), released",
     label = "rain = 2 for sure", linewidth = 2)
plot!(grid, x_sto; label = "rain = 1 or 3", linewidth = 2, linestyle = :dash)
savefig("s3_policy_random.png")

# ── 3.3 Simulate the long run with random rain (given) ───────────────────
rain_random() = ε_values[rand() < p[1] ? 1 : 2]
using Random
Random.seed!(1)
path_random = simulate_continuous(x_sto, Δ, 12.0, 402_000, rain_random)
mean_random = sum(path_random[2001:end]) / (length(path_random) - 2000)
println("long-run mean stock with random rain: ", round(mean_random, digits = 3),
        "   deterministic steady state: ", round(s_star, digits = 3))

# ── 3.4 What to make of it (read, do not run) ────────────────────────────
# Uncertainty of this size barely moves anything: the release falls by about 1%
# and the average stock rises by 0.12 units. Deciding as if the rain were its
# mean, which is what Session 2 did, is a good approximation for this model.
# When would it stop being one? Bigger shocks, a capacity that binds, more
# curvature in F or U. The homework tries the first two.
