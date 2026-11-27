"""
Upper bounds on the ℓ¹ norms of the powers of an operator on the zero-sum vectors `V_h`, by
iterating every column (`norms_of_powers_columns`), the bound behind the l1_mixing records.
"""

using BallArithmetic: add_up, mul_up, sub_up, sqrt_up, gamma_bound

export norms_of_powers_columns, norms_of_powers_sparse, refine_norms_of_powers

"""
    norms_of_powers_columns(blockmul!, n, m; η1, eta1A = 0.0, normP = 1.0, u,
                            block = 64, verbose = false) -> C

Upper bounds `C[j] ≥ ‖P^j|_{V_h}‖₁`, `j = 1, …, m`, where `V_h = {v ∈ ℝⁿ : Σ vᵢ = 0}`, for a linear
operator `P` known through a computed product that obeys, column by column,

    ‖fl(P y) − P y‖₁ ≤ η1 ‖y‖₁ + eta1A,      ‖P‖₁ ≤ normP.

**The bound.** For `v ∈ V_h` and any fixed `u ∈ ℝⁿ`, `v = Σᵢ vᵢ (eᵢ − u)` because `Σᵢ vᵢ = 0`, so
`P^j v = Σᵢ vᵢ (P^j eᵢ − u)` and

    ‖P^j|_{V_h}‖₁ ≤ maxᵢ ‖P^j eᵢ − u‖₁.

No property of `P` beyond linearity is used; `u` is a free choice, and near the fixed point of a
mixing `P` it makes the bound decay with `j`.

**The rounding.** Each column is iterated in floating point from `x₀ = eᵢ`, exact; with
`ε_j ≥ ‖x_j − P^j eᵢ‖₁`, `ε₀ = 0`, and `x_{j+1} − P^{j+1} eᵢ = (fl(P x_j) − P x_j) + P (x_j − P^j eᵢ)`,

    ε_{j+1} = η1 ‖x_j‖₁ + eta1A + normP ε_j,

so `C[j] = maxᵢ (‖x_j − u‖₁ + ε_j)`, every operation rounded upward.

`blockmul!(Y, X)` must set every column of `Y` to the computed product of `P` with the same column
of `X` (`n × b` matrices), under the model above; it is called on disjoint blocks of `block`
columns from several threads at once, so it must be thread safe.  Cost: `n m` column products.

With `cols`, only the columns `i ∈ cols` are iterated and `C[j]` is the maximum over them; the
entrywise maximum of the results for ranges covering `1:n` is the `C` of the whole run, bit for
bit, so a sweep can be split between machines (`bench/l1_mixing.jl` with `COLS`,
`bench/l1_mixing_merge.jl`).
"""
function norms_of_powers_columns(blockmul!, n::Int, m::Int; η1::Real, eta1A::Real = 0.0,
        normP::Real = 1.0, u::AbstractVector{Float64}, block::Int = 64, verbose::Bool = false,
        cols::AbstractUnitRange{<:Integer} = 1:n)
    length(u) == n || throw(DimensionMismatch("u has length $(length(u)), expected $n"))
    m >= 1 && block >= 1 || throw(ArgumentError("m and block must be positive"))
    1 <= first(cols) <= last(cols) <= n || throw(ArgumentError("cols = $cols is not a nonempty range in 1:$n"))
    η1, eta1A, normP = Float64(η1), Float64(eta1A), Float64(normP)
    starts = collect(first(cols):block:last(cols))
    nc = length(cols)
    C = zeros(m)
    Clock = ReentrantLock()
    done = Threads.Atomic{Int}(0)
    t0 = time()
    Threads.@threads :dynamic for s in starts
        cs = s:min(s + block - 1, last(cols))
        b = length(cs)
        X = zeros(n, b); Y = zeros(n, b)
        for (c, i) in enumerate(cs)
            X[i, c] = 1.0
        end
        nrm = ones(b)                       # ‖x_j‖₁, upward
        ε = zeros(b)
        C_t = zeros(m)                      # this block's maxima, merged under the lock
        for j in 1:m
            blockmul!(Y, X)
            X, Y = Y, X
            @inbounds for c in 1:b
                ε[c] = add_up(add_up(mul_up(η1, nrm[c]), eta1A), mul_up(normP, ε[c]))
                s1 = 0.0; s2 = 0.0
                for r in 1:n
                    x = X[r, c]
                    s1 = add_up(s1, abs(x))
                    s2 = add_up(s2, max(sub_up(x, u[r]), sub_up(u[r], x)))
                end
                nrm[c] = s1
                C_t[j] = max(C_t[j], add_up(s2, ε[c]))
            end
        end
        lock(Clock) do
            for j in 1:m
                C[j] = max(C[j], C_t[j])
            end
        end
        k = Threads.atomic_add!(done, b) + b
        verbose && k % max(block, nc ÷ 100) < b &&               # about every 1% of the columns
            (println("  columns $k / $nc, $(round(time() - t0, digits = 1)) s"); flush(stdout))
    end
    if verbose
        for j in 1:m
            println("  j = $j: ‖P^j|_V‖₁ ≤ $(C[j])")
        end
    end
    return C
end

"""
    norms_of_powers_sparse(P, m; rigorous = true, u = nothing, block = 64, verbose = false) -> C

[`norms_of_powers_columns`](@ref) for an explicit matrix `P`, taken as the operator itself.  With
`w` the largest number of nonzeros in a row or a column of `P`, an entry of `P y` is an inner product of at
most `w` terms, so `|fl(P y) − P y| ≤ γ_w |P| |y| + (underflow)` entrywise [Higham, Accuracy and
Stability of Numerical Algorithms, 2nd ed., (3.11)]; whence `η1 = γ_w ‖|P|‖₁`, `eta1A = √n η_A`
with the underflow term `η_A` of `_eta_stored` for `w` products, and
`normP = ‖|P|‖₁`, all upward.  `rigorous = false` sets `η1 = eta1A = 0`.  `u` defaults to the power
iteration `ulam_fixed_point` of `P`.
"""
function norms_of_powers_sparse(P::AbstractMatrix, m::Int; rigorous::Bool = true, u = nothing,
        block::Int = 64, verbose::Bool = false)
    n = size(P, 1)
    size(P, 2) == n || throw(DimensionMismatch("P must be square"))
    Ps = P isa SparseMatrixCSC ? SparseMatrixCSC{Float64,Int}(P) : sparse(Matrix{Float64}(P))
    w = max(1, _stencil_width(Ps))
    normP = _max_colsum_up(Ps)
    η1 = rigorous ? mul_up(gamma_bound(w, Float64), normP) : 0.0
    eta1A = rigorous ? mul_up(sqrt_up(Float64(n)), _eta_stored(n, w)) : 0.0
    uu = u === nothing ? ulam_fixed_point(v -> Ps * v, n).v : Vector{Float64}(u)
    return norms_of_powers_columns((Y, X) -> mul!(Y, Ps, X), n, m; η1, eta1A, normP, u = uu,
        block, verbose)
end

# `refine_norms_of_powers` is the one of RigorousInvariantMeasures (C_{i+j} ≤ C_i C_j, upward).
using RigorousInvariantMeasures: refine_norms_of_powers
