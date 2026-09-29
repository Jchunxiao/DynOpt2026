# ---------------------------------------------------------------------------
# s2_homework_solution.jl — Session 2, take it further. Complete version.
#
# Put the rain back. Rainfall k is now random, independent across years, so
# next year's stock is random too and the Bellman equation takes an expectation:
#
#     V(s) = max_x { f(s, x) + δ Σ_k p(k) V(min(s - x + k, M)) }
#
# Three parts. (1) The book's five-point rainfall, then our two-point version.
# (2) Where does the plateau go? (3) Stretch: the ergodic distribution as the
# stationary vector of the transition matrix under the optimal policy.
#
# Model and calibration from Miranda and Fackler, "Applied Computational 
# Economics and Finance," Sections 7.2.5 and 7.6.5. The five-point rainfall is
# from the book. The two-point rainfall k ∈ {1, 3} with equal probability
# has the same mean of 2 but standard deviation of 1.00 against 1.10.
# ---------------------------------------------------------------------------
using Plots, LinearAlgebra, Random

# ── Setup (given) ────────────────────────────────────────────────────────
α_1 = 14.0
β_1 = 0.8
α_2 = 10.0
β_2 = 0.4
δ   = 0.9
M   = 30

F(x) = α_1 * x^β_1
U(z) = α_2 * z^β_2

S = 0:M
X = 0:M
n = length(S)
m = length(X)

function build_reward()                 # from class, the deterministic part
    f = zeros(n, m)
    for i in 1:n
        for k in 1:m
            if X[k] <= S[i]
                f[i, k] = F(X[k]) + U(S[i] - X[k])
            else
                f[i, k] = -Inf
            end
        end
    end
    return f
end
f = build_reward()

# ═══ 1. Put the rain back ════════════════════════════════════════════════

# ── 1.1 The transition probability array ─────────────────────────────────
# P[k, i, j] is the probability of going from stock S[i] to stock S[j] when the
# release is X[k]. Rainfall takes the value k_values[q] with probability p[q].
function build_P(k_values, p)
    P = zeros(m, n, n)
    for k in 1:m
        for i in 1:n
            if X[k] <= S[i]
                for q in 1:length(k_values)
                    # TODO 1. After releasing X[k] and receiving rain k_values[q],
                    #         the stock is min(S[i] - X[k] + k_values[q], M). Find its
                    #         index j and ADD p[q] to P[k, i, j]. Add, do not assign:
                    #         two rainfalls can land on the same stock when M binds.
                    j = min(S[i] - X[k] + k_values[q], M) + 1
                    P[k, i, j] = P[k, i, j] + p[q]
                end
            end
        end
    end
    return P
end

# ── 1.2 Function iteration with an expectation ───────────────────────────
function value_iteration(f, P; tol = 1e-10, maxit = 1000)
    v_old = zeros(n)
    v_new = zeros(n)
    x_pol = zeros(Int, n)
    for it in 1:maxit
        for i in 1:n
            # TODO 2. The expected value of next year's stock, one number per
            #         release k, is Σ_j P[k, i, j] v_old[j]. That is the matrix
            #         P[:, i, :] times the vector v_old. Then the same findmax as
            #         in class.
            v_new[i], k_best = findmax(f[i, :] .+ δ .* (P[:, i, :] * v_old))
            x_pol[i] = X[k_best]
        end
        change = maximum(abs.(v_new .- v_old))
        v_old = copy(v_new)
        if change < tol
            return v_new, x_pol, it
        end
    end
    println("value_iteration did not converge in ", maxit, " iterations")
    return v_new, x_pol, maxit
end

# The book's rainfall (given)
k_book = [0, 1, 2, 3, 4]
p_book = [0.1, 0.2, 0.4, 0.2, 0.1]
P_book = build_P(k_book, p_book)
v_book, x_book, it_book = value_iteration(f, P_book)
println("Five-point rainfall: ", it_book, " iterations")
println("  policy x*(s): ", x_book)
published = [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5]
println("  matches the policy in the book: ", x_book == published)
println("  V(0) = ", round(v_book[1], digits = 2), "   V(30) = ", round(v_book[31], digits = 2))

# Our two-point rainfall (given)
k_two = [1, 3]
p_two = [0.5, 0.5]
P_two = build_P(k_two, p_two)
v_two, x_two, it_two = value_iteration(f, P_two)
println("Two-point rainfall: ", it_two, " iterations")
println("  policy x*(s): ", x_two)
println("  identical to the five-point policy: ", x_two == x_book)

# ── 1.3 Simulate with random rain (given) ────────────────────────────────
# Draw one rainfall: u is uniform on (0, 1), walk up the cumulative probabilities.
function draw_rain(k_values, p)
    u = rand()
    cumulative = 0.0
    for q in 1:length(p)
        cumulative = cumulative + p[q]
        if u <= cumulative
            return k_values[q]
        end
    end
    return k_values[end]
end

function simulate(x_pol, k_values, p, s_1, T)
    path = zeros(Int, T)
    path[1] = s_1
    for t in 1:(T - 1)
        x = x_pol[path[t] + 1]
        path[t + 1] = min(path[t] - x + draw_rain(k_values, p), M)
    end
    return path
end

Random.seed!(1) # We do this for reproducibility
T_long = 200_000
path_book = simulate(x_book, k_book, p_book, 4, T_long)
mean_book = sum(path_book) / T_long
println("Long-run mean stock, five-point rain, ", T_long, " years: ", round(mean_book, digits = 3),
        "   (book: 12.5)")
path_two = simulate(x_two, k_two, p_two, 4, T_long)
mean_two = sum(path_two) / T_long
println("Long-run mean stock, two-point rain:  ", round(mean_two, digits = 3))

# ═══ 2. Where does the plateau go? ═══════════════════════════════════════
# In class, three stocks were stationary: 11, 12 and 13, and the reservoir stayed
# wherever it first landed. With random rain there is no stock that maps back to
# itself for sure, so the question changes from "where does the stock settle" to
# "how often is the stock at each level". Count the years spent at each stock,
# skipping the first 1000 so that the climb from s_1 = 4 does not show up.
function frequencies(path, burn_in)
    freq = zeros(n)
    for t in (burn_in + 1):length(path)
        freq[path[t] + 1] = freq[path[t] + 1] + 1
    end
    return freq ./ (length(path) - burn_in)
end
freq_book = frequencies(path_book, 1000)
freq_two  = frequencies(path_two, 1000)
bar(S, freq_book; xlabel = "s, water in the reservoir", ylabel = "share of years",
    label = "five-point rain", alpha = 0.6, bar_width = 0.8)
bar!(S, freq_two; label = "two-point rain", alpha = 0.6, bar_width = 0.8)
vline!([12.513]; linestyle = :dash, label = "analytic steady state")
savefig("s2_homework_ergodic.png")
println("Stocks visited, five-point rain: ", S[freq_book .> 0])
println("Stocks visited, two-point rain:  ", S[freq_two .> 0])

# ═══ 3. Stretch: the ergodic distribution without simulating ═════════════
# Under the optimal policy the stock is a Markov chain with transition matrix
# P_star[i, j] = P[x*(S[i]) + 1, i, j]. Its long-run distribution π solves
# π = P_star' π. Start with an initial guess and apply P_star' until π stops changing.
function ergodic_distribution(P, x_pol; tol = 1e-14, maxit = 100_000)
    P_star = zeros(n, n) 
    for i in 1:n
        P_star[i, :] = P[x_pol[i] + 1, i, :] 
    end
    π = fill(1 / n, n)                   # Initial guess is the uniform distribution
    for it in 1:maxit
        π_new = P_star' * π
        if maximum(abs.(π_new .- π)) < tol
            return π_new
        end
        π = π_new
    end
    println("ergodic_distribution did not converge")
    return π
end
π_book = ergodic_distribution(P_book, x_book)
ergodic_mean_book = sum(π_book .* S)
π_two = ergodic_distribution(P_two, x_two)
ergodic_mean_two = sum(π_two .* S)
println("Ergodic mean stock from the stationary vector: five-point ", round(ergodic_mean_book, digits = 4),
        "   two-point ", round(ergodic_mean_two, digits = 4))
println("Monte Carlo above: five-point ", round(mean_book, digits = 4),
        "   two-point ", round(mean_two, digits = 4))
