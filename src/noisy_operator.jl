"""
The noisy Ulam operator on the square grid, `P_h = π N_ξ π L π`, from the deterministic block
`L_lo ≤ L ≤ L_lo + U` of `ulam_matrices` and the Ulam projection of the uniform noise, applied
matrix-free as `y ↦ fl(N fl(L̂ y))` with `L̂ = fl(L_lo + U/2)`, together with constants `η₁`,
`η₁ᴬ` such that

    ‖fl(N L̂ y) − P_h y‖₁ ≤ η₁ ‖y‖₁ + η₁ᴬ

for every `y` and every matrix `L` of the enclosure.

`N` is the product `N_y ⊗ N_x` of two `UlamProjectedUniformKernel` of
`RigorousInvariantMeasures`, the operator `π N_ξ π` of one axis (periodic or reflecting), which
is the Ulam projection of the convolution with the uniform density on the rectangle
`[-a, a] × [-b, b]`, the Ulam basis of the square grid being the product of the two
one-dimensional ones.  Vectors are in grid order, `v[(ix - 1) k + iy]` for the cell `(ix, iy)`
of the `k × k` grid, `k = 2^depth`, so that `reshape(v, k, k)[iy, ix]` is the cell; `grid_order`
is the permutation from the order of `build_uniform_partition`.

The constants.  With `w` the largest number of nonzeros in a row or a column of `L̂`, `γ_w` the
a priori bound of an inner product of `w` terms [Higham, Accuracy and Stability of Numerical
Algorithms, 2nd ed., (3.11)], `E_ij = max(L̂ − L_lo, L_lo + U − L̂)_ij ≥ |L̂ − L|_ij` for every
`L` of the enclosure, and `γ_N` the relative rounding of the product by `N`,
`|fl(Nv) − Nv| ≤ γ_N N|v|` entrywise: `|fl(L̂ y) − L̂ y| ≤ γ_w |L̂| |y|` entrywise, so with
`fl(L̂ y) = L̂ y + e₁` and `fl(N v) = N v + e₂` for `v = fl(L̂ y)`,

    ‖e₁‖₁ ≤ γ_w ‖L̂‖₁ ‖y‖₁,   ‖v‖₁ ≤ (1 + γ_w) ‖L̂‖₁ ‖y‖₁,   ‖e₂‖₁ ≤ γ_N ‖N‖₁ ‖v‖₁ = γ_N ‖v‖₁,
    fl(N L̂ y) − N L y = N (L̂ − L) y + N e₁ + e₂,   ‖N (L̂ − L) y‖₁ ≤ ‖E‖₁ ‖y‖₁,

whence `η₁ = ‖E‖₁ + ‖L̂‖₁ (γ_w + γ_N (1 + γ_w))`, `N` being nonnegative with unit column sums.  The
product by `N` is the one-dimensional kernel along `y` and then along `x`: with
`|fl(K v) − K v| ≤ γ_K K|v|` for each kernel (`RigorousInvariantMeasures`,
`UlamProjectedUniformKernel`), the two passes give `γ_N = (1 + γ_x)(1 + γ_y) − 1`.  The absolute
(underflow) term is `η₁ᴬ = √n η_A`, the ℓ¹ size of the bound `η_A` on the underflow of `n` entries
of `w + nnz_per_row(N)` products each (`_eta_stored`).  Every operation on the constants is
rounded upward.
"""

using RigorousInvariantMeasures: RigorousInvariantMeasures
using BallArithmetic: gamma_bound, mul_up, add_up, sub_up, sqrt_up, div_up, sum_up, subnormal_min
const RIM = RigorousInvariantMeasures

export SquareNoise, noisy_operator

"""
    SquareNoise(domain, depth, a; bc = :periodic, slots = Threads.nthreads())

The Ulam projection of the uniform noise on `[-a[1], a[1]] × [-a[2], a[2]]` (a number `a` is
the same half-width on both axes) on the grid of depth `depth` of `domain`, with the boundary
condition `bc`, as an operator applied by `apply!` or `*` to vectors in grid order.  The widths
`2a[i]` are turned into fractions of the sides of the domain exactly, so `a` should be dyadic.
`slots` kernels are kept, so that as many applications may run at once.
"""
struct SquareNoise{BC}
    k::Int
    ξ::NTuple{2,Float64}            # widths of the support as fractions of the sides
    kx::Vector{RIM.UlamProjectedUniformKernel{BC,Float64}}
    ky::Vector{RIM.UlamProjectedUniformKernel{BC,Float64}}
    bufs::Vector{Vector{Float64}}
    free::Channel{Int}
    γ::Float64                      # γ_N
    nnz_per_row::Int                # terms of one entry of N, for the underflow term
end

function SquareNoise(domain::IntervalBox{2}, depth::Int, a; bc::Symbol = :periodic,
        slots::Int = Threads.nthreads())
    ax, ay = a isa Number ? (a, a) : (a[1], a[2])
    k = 2^depth
    B = RIM.Ulam(k)
    ξ = (Rational{BigInt}(2ax) / Rational{BigInt}(diam(domain[1])),
         Rational{BigInt}(2ay) / Rational{BigInt}(diam(domain[2])))
    kx = [RIM.UlamProjectedUniformKernel(Val(bc), Float64, B, ξ[1]) for _ in 1:slots]
    ky = [RIM.UlamProjectedUniformKernel(Val(bc), Float64, B, ξ[2]) for _ in 1:slots]
    bufs = [zeros(k) for _ in 1:slots]
    free = Channel{Int}(slots)
    foreach(s -> put!(free, s), 1:slots)
    γx = gamma_bound(RIM.nonzero_per_row(kx[1]), Float64)
    γy = gamma_bound(RIM.nonzero_per_row(ky[1]), Float64)
    γ = add_up(add_up(γx, γy), mul_up(γx, γy))
    return SquareNoise{bc}(k, (Float64(ξ[1]), Float64(ξ[2])), kx, ky, bufs, free, γ,
        RIM.nonzero_per_row(kx[1]) + RIM.nonzero_per_row(ky[1]))
end

"""
    apply!(N::SquareNoise, v::Vector{Float64}) -> v

`fl(N v)` in place, the kernel along `y` and then along `x`.
"""
function apply!(N::SquareNoise, v::Vector{Float64})
    k = N.k
    length(v) == k * k || throw(DimensionMismatch("expected a vector of length $(k * k), got $(length(v))"))
    s = take!(N.free)
    try
        Kx, Ky, buf = N.kx[s], N.ky[s], N.bufs[s]
        V = reshape(v, k, k)
        for ix in 1:k                       # along y: the columns of V
            col = view(V, :, ix)
            copyto!(buf, col); RIM.mul!(Ky, buf); copyto!(col, buf)
        end
        for iy in 1:k                       # along x: the rows of V
            row = view(V, iy, :)
            copyto!(buf, row); RIM.mul!(Kx, buf); copyto!(row, buf)
        end
    finally
        put!(N.free, s)
    end
    return v
end

Base.:*(N::SquareNoise, v::AbstractVector) = apply!(N, Vector{Float64}(v))

"""
    noisy_operator(L_lo, U, N::SquareNoise) -> info

The operator `P_h = N L` of the file docstring, for `L_lo` and `U` in grid order: `info` holds
`L̂ = fl(L_lo + U/2)` (`Lhat`), the product `Amul(x) = fl(N fl(L̂ x))` on a vector, the product
column by column of a matrix, `Bmul!(Y, X)` setting `Y = fl(N fl(L̂ X))` with the same arithmetic
per column as `Amul` and so the same rounding model, `w` and `γ_w` (`γw`), `γ_N` (`γN`),
`colL ≥ ‖L̂‖₁`, `colE ≥ ‖E‖₁`, the number of nonzeros of `L̂` (`nnzL`), and the constants `η1`,
`eta1A` of the file docstring.
"""
function noisy_operator(L_lo::SparseMatrixCSC{Float64,Int}, U::SparseMatrixCSC{Float64,Int},
        N::SquareNoise)
    n = size(L_lo, 1)
    size(L_lo, 2) == n && size(U) == (n, n) || throw(DimensionMismatch("L_lo and U must be square of the same size"))
    n == N.k^2 || throw(DimensionMismatch("the matrices are $n × $n and the noise is on $(N.k^2) cells"))
    Lhat = dropzeros!(L_lo + U / 2)
    E = dropzeros!(max.(sub_up.(Lhat, L_lo), sub_up.(add_up.(L_lo, U), Lhat)))
    w = max(1, _stencil_width(Lhat))
    γw = gamma_bound(w, Float64)
    colL, colE = _max_colsum_up(Lhat), _max_colsum_up(E)
    γN = N.γ
    η1 = add_up(colE, mul_up(colL, add_up(γw, mul_up(γN, add_up(1.0, γw)))))
    Amul(x) = apply!(N, Lhat * Vector{Float64}(x))
    function Bmul!(Y::Matrix{Float64}, X::Matrix{Float64})
        mul!(Y, Lhat, X)
        buf = Vector{Float64}(undef, n)
        for c in axes(Y, 2)
            col = view(Y, :, c)
            copyto!(buf, col); apply!(N, buf); copyto!(col, buf)
        end
        return Y
    end
    etaA = _eta_stored(n, w + N.nnz_per_row)
    eta1A = mul_up(sqrt_up(Float64(n)), etaA)
    return (; Lhat, Amul, Bmul!, w, γw, γN, nnzL = nnz(Lhat), colL, colE, η1, eta1A)
end

# The three helpers below are those of RigorousPseudospectra (src/operator.jl at 283c1b2, where
# they are private), moved here unchanged when this package stopped depending on it, so that the
# constants of the records stay bit for bit (test/test_records.jl).
# checked: BallArithmetic.upper_bound_L1_opnorm -- takes a dense BallMatrix, not a SparseMatrixCSC

"The largest number of nonzeros in a row or a column of `A`."
function _stencil_width(A::SparseMatrixCSC)
    wc = maximum(diff(A.colptr); init = 0)
    rows = zeros(Int, size(A, 1))
    for r in rowvals(A)
        rows[r] += 1
    end
    return max(wc, maximum(rows; init = 0))
end

"An upper bound on the largest column sum of `|A|`, summed upward over the stored entries."
function _max_colsum_up(A::SparseMatrixCSC)
    vals = nonzeros(A)
    return maximum(j -> sum_up(abs(vals[p]) for p in nzrange(A, j)), 1:size(A, 2); init = 0.0)
end

"""
    _eta_stored(n, w) -> η_A

An upper bound on the ℓ² norm of the underflow error of a product with `n` entries of `w` terms
each, `√n · 4w · η` with `η` the smallest subnormal double, upward (the factor 4 is the one of
RigorousPseudospectra, which also covers complex data).
"""
function _eta_stored(n, w)
    η = subnormal_min(Float64)
    c = 4.0
    return mul_up(sqrt_up(Float64(n)), mul_up(c * w, η))
end
