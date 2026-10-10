# ---------------------------------------------------------------------------
# s3_homework_solution.jl — Session 3, take it further. Complete version.
#
# Four parts, each a variation on the code from class. The functions from class
# are repeated below so this file runs on its own.
#
#   1. Make capacity bind. M = 12, just below the steady state of 12.51.
#   2. Patience. δ = 0.8 and δ = 0.95: the release and the shadow price at the
#      steady state do not depend on δ, the stock does (Section 8.4.4).
#   3. Five-point rain. The book's distribution instead of our two-point one.
#   4. Stretch. The finite-horizon problem at T = 10, 20, 40.
# ---------------------------------------------------------------------------
using JuMP, Ipopt, Optim, Plots

# ── Setup (given) ────────────────────────────────────────────────────────
α_1 = 14.0
β_1 = 0.8
α_2 = 10.0
β_2 = 0.4
δ   = 0.9
M   = 30.0
ε_rain = 2.0

F(x)  = α_1 * x^β_1
U(z)  = α_2 * z^β_2
dF(x) = α_1 * β_1 * x^(β_1 - 1)
dU(z) = α_2 * β_2 * z^(β_2 - 1)

# From class (given): interpolation, one node, the iteration, the simulation.
function interp(v, Δ, s)
    i = floor(Int, s / Δ) + 1
    if i >= length(v)
        return v[end]
    end
    s_i = (i - 1) * Δ
    slope = (v[i + 1] - v[i]) / Δ
    return v[i] + slope * (s - s_i)
end

function solve_node(s, v, Δ, continuation, δ)
    if s < 1e-8
        return 0.0, δ * continuation(v, Δ, 0.0)
    end
    result = optimize(x -> -(F(x) + U(s - x) + δ * continuation(v, Δ, s - x)), 1e-9, s - 1e-9)
    return Optim.minimizer(result), -Optim.minimum(result)
end

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
    return v_new, x_pol, maxit
end

function simulate_continuous(x_pol, Δ, s_1, T, rain, M)
    path = zeros(T)
    path[1] = s_1
    for t in 1:(T - 1)
        x = interp(x_pol, Δ, path[t])
        path[t + 1] = min(path[t] - x + rain(), M)
    end
    return path
end
rain_certain() = ε_rain

Δ = 0.01
continuation_certain(v, Δ, z) = interp(v, Δ, min(z + ε_rain, M))
println("Reference solve at M = 30, about 15 seconds...")
v_30, x_30, grid_30, it_30 = value_iteration(continuation_certain, Δ)

# ═══ 1. Make capacity bind ═══════════════════════════════════════════════

# ── 1.1 In the finite-horizon JuMP model (blank 1) ───────────────────────
# In class the capacity was a bound on s[t]. Written as a named constraint it
# gets a dual: zero when it is slack, and the value of one more unit of
# capacity when it binds.
function solve_finite_horizon_capped(T, s_1, cap)
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    @variable(model, x[1:T] >= 0, start = 2.0)
    @variable(model, z[1:T] >= 0, start = 2.0)
    @variable(model, s[1:T+1] >= 0, start = 8.0)
    @constraint(model, initial, s[1] == s_1)
    @constraint(model, split[t = 1:T], z[t] == s[t] - x[t])
    @constraint(model, transition[t = 1:T], s[t + 1] == z[t] + ε_rain)
    # TODO 1. The reservoir cannot hold more than cap in any year. One
    #         constraint per year, named capacity, over t = 1:T+1.
    #         @constraint(model, capacity[t = 1:T+1], ...)
    @constraint(model, capacity[t = 1:T+1], s[t] <= cap)
    @objective(model, Max, sum(δ^(t - 1) * (α_1 * x[t]^β_1 + α_2 * z[t]^β_2) for t in 1:T))
    optimize!(model)
    @assert is_solved_and_feasible(model)
    μ_cap = zeros(T + 1)
    for t in 1:T+1
        μ_cap[t] = dual(capacity[t])
    end
    return value.(x), value.(s), objective_value(model), μ_cap
end

x_fh30, s_fh30, W_30, μ_cap30 = solve_finite_horizon_capped(30, 4.0, 30.0)
x_fh12, s_fh12, W_12, μ_cap12 = solve_finite_horizon_capped(30, 4.0, 12.0)
println("Capacity 30: objective ", round(W_30, digits = 4), ", largest capacity dual ", maximum(abs.(μ_cap30)))
println("Capacity 12: objective ", round(W_12, digits = 4), ", capacity binds in years ",
        [t for t in 1:31 if abs(μ_cap12[t]) > 1e-6])
println("  s_t with M = 12: ", round.(s_fh12, digits = 3))
plot(1:31, s_fh30; xlabel = "year", ylabel = "s_t", label = "M = 30", marker = :circle)
plot!(1:31, s_fh12; label = "M = 12", marker = :circle)
savefig("s3_homework_capacity.png")

# ── 1.2 In the infinite-horizon model (given) ────────────────────────────
# The cap enters through the transition: min(z + ε, 12).
continuation_capped(v, Δ, z) = interp(v, Δ, min(z + ε_rain, 12.0))
println("Solve at M = 12, about 7 seconds...")
v_12, x_12, grid_12, it_12 = value_iteration(continuation_capped, Δ; M = 12.0)
println("  s      x(s), M = 30   x(s), M = 12   λ(s), M = 30   λ(s), M = 12")
for s in (4.0, 8.0, 10.0, 11.0, 11.5, 12.0)
    println("  ", rpad(s, 6), rpad(round(interp(x_30, Δ, s), digits = 4), 15),
            rpad(round(interp(x_12, Δ, s), digits = 4), 15),
            rpad(round(dF(interp(x_30, Δ, s)), digits = 4), 15),
            round(dF(interp(x_12, Δ, s)), digits = 4))
end
long_run_12 = simulate_continuous(x_12, Δ, 4.0, 200, rain_certain, 12.0)[end]
println("  long-run stock with M = 12: ", round(long_run_12, digits = 4),
        "   release there: ", round(interp(x_12, Δ, long_run_12), digits = 4))
# Below the cap the policy barely moves: the planner cannot store the extra
# water, so the value of holding it falls only where the cap is within reach.
# At the cap the reservoir stays full and the release equals the rain.

# ═══ 2. Patience ═════════════════════════════════════════════════════════
# Section 8.4.4: at the steady state x* = ε and λ* = F'(ε) whatever δ is, while
# the stock solves U'(s* - x*) = (1 - δ) F'(x*), so it rises with δ.
function analytic_steady_state(δ_value)
    x = ε_rain
    λ = dF(x)
    # TODO 2. Invert U'(z) = (1 - δ) λ for z, then s = x + z.
    #         U'(z) = α_2 β_2 z^(β_2 - 1), so z = ((1 - δ) λ / (α_2 β_2))^(1 / (β_2 - 1)).
    z = ((1 - δ_value) * λ / (α_2 * β_2))^(1 / (β_2 - 1))
    return x, λ, x + z
end

for δ_value in (0.8, 0.9, 0.95)
    x_a, λ_a, s_a = analytic_steady_state(δ_value)
    println("δ = ", δ_value, ":  analytic x* = ", x_a, "  λ* = ", round(λ_a, digits = 4), "  s* = ", round(s_a, digits = 3))
end
println("Solve at δ = 0.8 (about 7 seconds) and δ = 0.95 (about 40 seconds)...")
v_08, x_08, grid_08, it_08 = value_iteration(continuation_certain, Δ; δ = 0.8)
v_95, x_95, grid_95, it_95 = value_iteration(continuation_certain, Δ; δ = 0.95)
long_run_08 = simulate_continuous(x_08, Δ, 4.0, 400, rain_certain, M)[end]
long_run_95 = simulate_continuous(x_95, Δ, 4.0, 400, rain_certain, M)[end]
println("δ = 0.8:  ", it_08, " iterations, long-run stock ", round(long_run_08, digits = 3),
        ", release there ", round(interp(x_08, Δ, long_run_08), digits = 4),
        ", λ ", round(dF(interp(x_08, Δ, long_run_08)), digits = 4))
println("δ = 0.95: ", it_95, " iterations, long-run stock ", round(long_run_95, digits = 3),
        ", release there ", round(interp(x_95, Δ, long_run_95), digits = 4),
        ", λ ", round(dF(interp(x_95, Δ, long_run_95)), digits = 4))
# At δ = 0.8 the numbers check: the stock settles at 5.31, the release at 2 and
# λ at 9.75. At δ = 0.95 the formula asks for s* = 35.4, more than the reservoir
# holds, so the capacity binds instead: the stock settles at 30, and the release
# still equals the rain because a full reservoir cannot grow.

# ═══ 3. Five-point rain ══════════════════════════════════════════════════
k_five = [0.0, 1.0, 2.0, 3.0, 4.0]
p_five = [0.1, 0.2, 0.4, 0.2, 0.1]
function continuation_five(v, Δ, z)
    expected = 0.0
    for q in 1:length(k_five)
        expected = expected + p_five[q] * interp(v, Δ, min(z + k_five[q], M))
    end
    return expected
end
k_two = [1.0, 3.0]
p_two = [0.5, 0.5]
function continuation_two(v, Δ, z)
    expected = 0.0
    for q in 1:length(k_two)
        expected = expected + p_two[q] * interp(v, Δ, min(z + k_two[q], M))
    end
    return expected
end
println("Solve with two-point and five-point rain")
v_two, x_two, grid_two, it_two = value_iteration(continuation_two, Δ)
v_five, x_five, grid_five, it_five = value_iteration(continuation_five, Δ)
s_star = analytic_steady_state(δ)[3]
println("  s        rain = 2     two-point    five-point")
for s in (4.0, 8.0, 12.0, s_star, 20.0, 30.0)
    println("  ", rpad(round(s, digits = 3), 8), rpad(round(interp(x_30, Δ, s), digits = 4), 13),
            rpad(round(interp(x_two, Δ, s), digits = 4), 13), round(interp(x_five, Δ, s), digits = 4))
end
# The five-point rain has a standard deviation of 1.10 against 1.00 for the
# two-point one. The policy moves by about 1/1000.

# ═══ 4. Stretch: horizons of 10, 20 and 40 years ═════════════════════════
function solve_finite_horizon(T, s_1)
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    @variable(model, x[1:T] >= 0, start = 2.0)
    @variable(model, z[1:T] >= 0, start = 2.0)
    @variable(model, 0 <= s[1:T+1] <= M, start = 8.0)
    @constraint(model, initial, s[1] == s_1)
    @constraint(model, split[t = 1:T], z[t] == s[t] - x[t])
    @constraint(model, transition[t = 1:T], s[t + 1] == z[t] + ε_rain)
    @objective(model, Max, sum(δ^(t - 1) * (α_1 * x[t]^β_1 + α_2 * z[t]^β_2) for t in 1:T))
    optimize!(model)
    @assert is_solved_and_feasible(model)
    return value.(x), value.(s), objective_value(model)
end

x_10, s_10, W_10 = solve_finite_horizon(10, 4.0)
x_20, s_20, W_20 = solve_finite_horizon(20, 4.0)
x_40, s_40, W_40 = solve_finite_horizon(40, 4.0)
plot(1:10, x_10; xlabel = "year", ylabel = "x_t, released", label = "T = 10", marker = :circle)
plot!(1:20, x_20; label = "T = 20", marker = :circle)
plot!(1:40, x_40; label = "T = 40", marker = :circle)
hline!([ε_rain]; linestyle = :dash, label = "steady-state release")
savefig("s3_homework_horizons.png")
println("x_1 at T = 10, 20, 40: ", round(x_10[1], digits = 4), "  ", round(x_20[1], digits = 4), "  ", round(x_40[1], digits = 4))
println("x_5 at T = 10, 20, 40: ", round(x_10[5], digits = 4), "  ", round(x_20[5], digits = 4), "  ", round(x_40[5], digits = 4))
# The early years sit on top of each other and the drawdown slides right as T
# grows. The infinite horizon is the limit of the finite one.
