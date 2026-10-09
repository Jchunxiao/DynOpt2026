# ---------------------------------------------------------------------------
# s2_inclass.jl — Session 2, Phases 1 to 3. Fill in the three TODO blanks.
#
# The same reservoir, now year after year. What you do not release remains
# for the next year, plus rainfall. Everything is deterministic and integer.
#
#     state       s_t ∈ {0, 1, ..., 30}    water at the start of year t
#     control     x_t ∈ {0, 1, ..., s_t}   released for irrigation
#     payoff      f(s, x) = F(x) + U(s - x)
#     transition  s_{t+1} = min(s_t - x_t + k, M),  k = 2 every year, M = 30
#     Bellman     V(s) = max_x { f(s, x) + δ V(min(s - x + k, M)) }
#
# Model and calibration from Miranda and Fackler, "Applied Computational
# Economics and Finance," Sections 7.2.5 and 7.6.5. Two changes we make:
# rainfall is fixed at its mean of 2 instead of random (Session 3 adds the
# randomness back; Section 7.3 of the book says the algorithms are unchanged),
# and we keep the capacity M = 30 even though it never binds.
# ---------------------------------------------------------------------------
using Plots, LinearAlgebra

# ── Setup (given) ────────────────────────────────────────────────────────
α_1 = 14.0      # farmer benefit scale
β_1 = 0.8       # farmer benefit curvature
α_2 = 10.0      # recreation benefit scale
β_2 = 0.4       # recreation benefit curvature
δ   = 0.9       # discount factor
M   = 30        # reservoir capacity, units of water
k_rain = 2      # rainfall, the same every year

F(x) = α_1 * x^β_1                    # farmers' benefit from x units of irrigation
U(z) = α_2 * z^β_2                    # recreation benefit from z units left in place

S = 0:M         # 31 possible stocks
X = 0:M         # 31 possible releases
n = length(S)
m = length(X)
# Julia counts from 1, so the stock S[i] is i - 1 and the release X[k] is k - 1.
# A stock s sits at index s + 1. Keep this in mind to avoid bookkeeping traps.

# ═══ PHASE 1. From one season to many ════════════════════════════════════

# ── 1.1 Two pieces: reward matrix and transition index ───────────────────
# f[i, k] is the payoff from releasing X[k] when the stock is S[i].
# g[i, k] is the INDEX of next year's stock after that release.
function build_pieces()
    f = zeros(n, m)
    g = zeros(Int, n, m)
    for i in 1:n
        for k in 1:m
            if X[k] <= S[i]
                f[i, k] = F(X[k]) + U(S[i] - X[k])
                g[i, k] = min(S[i] - X[k] + ε_rain, M) + 1
                # TODO 1. The payoff is F(X[k]) + U(S[i] - X[k]) when the release
                #         is feasible, and -Inf when it is not, so that an
                #         impossible release never wins the max. Fill both branches.
                # TODO 2. Next year's stock is min(S[i] - X[k] + k_rain, M).
                #         Store its index in S, which is the stock plus one.
            else
                f[i, k] = -Inf
                g[i, k] = 1           # never used, because f[i, k] = -Inf
            end
        end
    end
    return f, g
end
f, g = build_pieces()

# Look at one row (given): the stock is 4, so releases above 4 are -Inf
println("f[5, :] at s = 4: ", round.(f[5, :], digits = 2))
println("g[5, :] at s = 4: ", g[5, :], "   (next stock = index - 1)")

# ── 1.2 Backward recursion, finite horizon ───────────────────────────────
# V[i, t] is the value of holding stock S[i] at the start of year t, with T
# years to go and nothing after: V[:, T + 1] = 0. Work backward from T.
#     V_t(s) = max_x { f(s, x) + δ V_{t+1}(s') }
function backward_recursion(f, g, T)
    V = zeros(n, T + 1)
    x_pol = zeros(Int, n, T)
    for t in T:-1:1
        for i in 1:n
            V[i, t], k_best = findmax(f[i, :] .+ δ .* V[g[i, :], t + 1])
            # TODO 3. For every release k, the candidate value is
            #         f[i, k] + δ * V[g[i, k], t + 1]. findmax over the vector
            #         of candidates returns the best value and its index.
            x_pol[i, t] = X[k_best]
        end
    end
    return V, x_pol
end

T = 5
V_T, x_T = backward_recursion(f, g, T)
println("Finite horizon, T = ", T, ". Release x_t(s) for s = 0, 1, ..., 30:")
for t in 1:T
    println("  t = ", t, ": ", x_T[:, t])
end
println("At s = 4: release ", x_T[5, 1], " in year 1, and ", x_T[5, T], " in year ", T)

# ═══ PHASE 2. Infinite horizon by function iteration ═════════════════════

# ── 2.1 Drop the time subscript (given) ──────────────────────────────────
# The same update as in TODO 3, applied to one vector until it stops changing.
#     v_new(s) = max_x { f(s, x) + δ v_old(s') }
# Stopping rule from the book, Section 7.3.2:  ‖v_k − v*‖ ≤ δ/(1−δ) ‖v_k − v_{k−1}‖
function value_iteration(f, g; tol = 1e-10, maxit = 1000)
    v_old = zeros(n)
    v_new = zeros(n)
    x_pol = zeros(Int, n)
    for it in 1:maxit
        for i in 1:n
            v_new[i], k_best = findmax(f[i, :] .+ δ .* v_old[g[i, :]])
            x_pol[i] = X[k_best]
        end
        change = maximum(abs.(v_new .- v_old))
        if it == 1 || it % 32 == 0
            println("  iteration ", it, "   error bound ", δ / (1 - δ) * change)
        end
        v_old = copy(v_new)
        if change < tol
            return v_new, x_pol, it
        end
    end
    println("value_iteration did not converge in ", maxit, " iterations")
    return v_new, x_pol, maxit
end

println("Function iteration:")
v_vfi, x_vfi, it_vfi = value_iteration(f, g)
println("Converged in ", it_vfi, " iterations")
println("Policy x*(s), s = 0, 1, ..., 30: ", x_vfi)
println("V(12) = ", round(v_vfi[13], digits = 2), "   V(13) = ", round(v_vfi[14], digits = 2),
        "   V(30) = ", round(v_vfi[31], digits = 2))

# ── 2.2 Plot the optimal policy and the value function (given) ───────────
bar(S, x_vfi; xlabel = "s, water in the reservoir", ylabel = "x*(s), released",
    label = false, bar_width = 0.8)
savefig("s2_policy.png")
plot(S, v_vfi; xlabel = "s, water in the reservoir", ylabel = "V(s)", label = false)
savefig("s2_value.png")

# ═══ PHASE 3. Policy iteration and simulation ════════════════════════════

# ── 3.1 Policy iteration (given) ─────────────────────────────────────────
# For a FIXED policy, the value solves a linear system: v = f_star + δ P_star v,
# where P_star[i, j] = 1 if the policy moves stock i to stock j. Solve it, then
# improve the policy, then repeat. Newton's method on the Bellman equation.
function policy_iteration(f, g; maxit = 100)
    x_pol = zeros(Int, n)             # starting guess: release nothing
    v = zeros(n)
    for it in 1:maxit
        P_star = zeros(n, n)
        f_star = zeros(n)
        for i in 1:n
            k = x_pol[i] + 1          # index of the release chosen at S[i]
            P_star[i, g[i, k]] = 1.0  # mark the transition to next year's stock
            f_star[i] = f[i, k]
        end
        v = (I - δ * P_star) \ f_star
        x_new = zeros(Int, n)
        for i in 1:n
            best, k_best = findmax(f[i, :] .+ δ .* v[g[i, :]])
            x_new[i] = X[k_best]
        end
        println("  iteration ", it, "   states where the release changed: ", sum(x_new .!= x_pol))
        if x_new == x_pol
            return v, x_pol, it
        end
        x_pol = x_new
    end
    println("policy_iteration did not converge in ", maxit, " iterations")
    return v, x_pol, maxit
end

println("Policy iteration:")
v_pi, x_pi, it_pi = policy_iteration(f, g)
println("Converged in ", it_pi, " iterations, against ", it_vfi, " for function iteration")
println("Same policy: ", x_pi == x_vfi,
        "   largest difference in V: ", maximum(abs.(v_pi .- v_vfi)))

# ── 3.2 Simulate forward (given) ─────────────────────────────────────────
function simulate(x_pol, s_1, T)
    path = zeros(Int, T)
    path[1] = s_1
    for t in 1:(T - 1)
        x = x_pol[path[t] + 1]                         # the release at stock path[t]
        path[t + 1] = min(path[t] - x + k_rain, M)
    end
    return path
end
path_4  = simulate(x_vfi, 4, 20)
path_30 = simulate(x_vfi, 30, 20)
println("Stock from s_1 = 4:  ", path_4)
println("Stock from s_1 = 30: ", path_30)

s_star = 12.513482907674891       # analytic steady state of the continuous model
plot(1:20, path_4; xlabel = "year", ylabel = "s_t, water in the reservoir",
     label = "from s_1 = 4", marker = :circle)
plot!(1:20, path_30; label = "from s_1 = 30", marker = :circle)
hline!([s_star]; linestyle = :dash, label = "analytic steady state 12.51")
savefig("s2_paths.png")

# ── 3.3 Stationary states (given) ────────────────────────────────────────
# A stock is stationary when following the policy brings you back to it.
function stationary_states(x_pol)
    result = zeros(Int, 0)                             # empty, we add to it
    for i in 1:n
        if min(S[i] - x_pol[i] + k_rain, M) == S[i]
            push!(result, S[i])
        end
    end
    return result
end
stationary = stationary_states(x_vfi)
println("Stationary stocks: ", stationary, "   (the release there is ", x_vfi[stationary .+ 1], ")")

# ── 3.4 The missing term, computed (given) ───────────────────────────────
# Last session we promised the value of one more unit of water at the long-run
# stock: λ = 9.750. On a grid, the derivative of V is a difference.
λ_grid = v_vfi[14] - v_vfi[13]                         # V(13) - V(12)
println("V(13) - V(12) = ", round(λ_grid, digits = 3), "   analytic λ* = 9.750")

# ── 3.5 What to make of the plateau (read, do not run) ───────────────────
# The continuous model has one steady state, s* = 12.513, where the release
# equals the rainfall, x* = 2. Here the release must be an integer, and x*(s) = 2
# for s = 11, 12 and 13, so all three stay put. Where you end up depends on
# where you start: 11 from below, 13 from above. Session 3 makes the state
# continuous and the rainfall random, and either change removes it.
