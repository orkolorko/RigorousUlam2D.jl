"""
The noisy Ulam operator on the square grid, `P_δ = N L`, from the deterministic block
`L_lo ≤ L ≤ L_lo + U` of `ulam_matrices` and the Ulam projection of the uniform noise, applied
matrix-free as `y ↦ fl(N fl(L̂ y))` with `L̂ = fl(L_lo + U/2)`, together with a constant `η` such
that

    ‖fl(N L̂ y) − P_δ y‖₂ ≤ η ‖y‖₂,   ‖fl(L̂ᵀ N y) − P_δᵀ y‖₂ ≤ η ‖y‖₂

for every `y` and every matrix `L` of the enclosure.  This is the rounding model of
`RigorousPseudospectra.LinOp` with `P_δ` itself as the operator being certified and the two
products as its inexact evaluations, so a certificate on the `LinOp` of `noisy_operator` is a
certificate on `P_δ`.

`N` is the product `N_y ⊗ N_x` of two `UlamProjectedUniformKernel` of
`RigorousInvariantMeasures`, the operator `π N_ξ π` of one axis (periodic or reflecting), which
is the Ulam projection of the convolution with the uniform density on the rectangle
`[-a, a] × [-b, b]`, the Ulam basis of the square grid being the product of the two
one-dimensional ones.  Vectors are in grid order, `v[(ix - 1) k + iy]` for the cell `(ix, iy)`
of the `k × k` grid, `k = 2^depth`, so that `reshape(v, k, k)[iy, ix]` is the cell; `grid_order`
is the permutation from the order of `build_uniform_partition`.

The constant.  With `w` the largest number of nonzeros in a row or a column of `L̂`,
`ε_L = √w γ_w` [Higham, Accuracy and Stability of Numerical Algorithms, 2nd ed., (3.11)-(3.12)],
`‖L̂‖₂ ≤ ‖L̂‖ := √(‖L̂‖₁ ‖L̂‖_∞)`, `‖N‖₂ ≤ √(‖N‖₁ ‖N‖_∞) = 1` since `N` is nonnegative with unit row
and column sums, `λ := √(‖E‖₁ ‖E‖_∞) ≥ ‖L̂ − L‖₂` for every `L` of the enclosure, where
`E_ij = max(L̂ − L_lo, L_lo + U − L̂)_ij ≥ |L̂ − L|_ij`, and `γ_N` the relative rounding of the
product by `N`, `|fl(Nv) − Nv| ≤ γ_N N|v|` entrywise, whence `‖fl(Nv) − Nv‖₂ ≤ γ_N ‖v‖₂`:

    fl(L̂ y) = L̂ y + e₁,   ‖e₁‖ ≤ ε_L ‖L̂‖ ‖y‖,
    fl(N v) = N v + e₂,    ‖e₂‖ ≤ γ_N ‖v‖ ≤ γ_N (1 + ε_L) ‖L̂‖ ‖y‖   for v = fl(L̂ y),
    fl(N L̂ y) − N L y = N (L̂ − L) y + N e₁ + e₂,

so `η = λ + ‖L̂‖ (ε_L + γ_N (1 + ε_L))`; the adjoint product `fl(L̂ᵀ fl(N y))` gives
`λ + ‖L̂‖ (γ_N + ε_L (1 + γ_N))`, the same number, `N` being symmetric.  The product by `N` is the
one-dimensional kernel along `y` and then along `x`: with `|fl(K v) − K v| ≤ γ_K K|v|` for each
kernel (`RigorousInvariantMeasures`, `UlamProjectedUniformKernel`), the two passes give
`γ_N = (1 + γ_x)(1 + γ_y) − 1`.  Every operation on the constants is rounded upward.
"""

using RigorousInvariantMeasures: RigorousInvariantMeasures
using RigorousPseudospectra: RigorousPseudospectra, LinOp, norm_lower
using BallArithmetic: gamma_bound, mul_up, add_up, sub_up, sqrt_up, div_up
const RIM = RigorousInvariantMeasures
const RP = RigorousPseudospectra

export SquareNoise, grid_order, noisy_operator

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
    grid_order(domain, depth) -> perm

The permutation from the order of `build_uniform_partition(domain, depth)` to the grid order of
this file: `v[perm]` and `L[perm, perm]` are in grid order when `v` and `L` are indexed as the
partition (`perm[(ix - 1) k + iy]` is the index in the partition of the cell `(ix, iy)`).
"""
function grid_order(domain::IntervalBox{2}, depth::Int)
    k = 2^depth
    _, grid_to_box = build_grid_index_maps(build_uniform_partition(domain, depth), (k, k))
    return vec(grid_to_box)                 # grid_to_box[iy, ix], column major
end

"""
    noisy_operator(L_lo, U, N::SquareNoise) -> (op::LinOp, info)

The operator `P_δ = N L` of the file docstring as a `LinOp` of `RigorousPseudospectra`, for
`L_lo` and `U` in grid order: the products `y ↦ fl(N fl(L̂ y))` and `y ↦ fl(L̂ᵀ fl(N y))` with
`L̂ = fl(L_lo + U/2)`, the constant `η` of the file docstring turned into the relative `ε_A = η/a_lo`
of the `LinOp` model with `a_lo ≤ ‖P_δ‖₂` from `norm_lower`, and the underflow term `η_A` of
`w + nnz_per_row(N)` products per entry.  `info` holds `L̂`, the two products as functions of a
vector (`Amul`, `Aadj`) and the pieces of `η`.
"""
function noisy_operator(L_lo::SparseMatrixCSC{Float64,Int}, U::SparseMatrixCSC{Float64,Int},
        N::SquareNoise)
    n = size(L_lo, 1)
    size(L_lo, 2) == n && size(U) == (n, n) || throw(DimensionMismatch("L_lo and U must be square of the same size"))
    n == N.k^2 || throw(DimensionMismatch("the matrices are $n × $n and the noise is on $(N.k^2) cells"))
    Lhat = dropzeros!(L_lo + U / 2)
    E = dropzeros!(max.(sub_up.(Lhat, L_lo), sub_up.(add_up.(L_lo, U), Lhat)))
    Lt = sparse(transpose(Lhat))
    w = max(1, RP._stencil_width(Lhat))
    εL = mul_up(sqrt_up(Float64(w)), gamma_bound(w, Float64))
    normL = sqrt_up(mul_up(RP._max_colsum_up(Lhat), RP._max_colsum_up(Lt)))
    λ = sqrt_up(mul_up(RP._max_colsum_up(E), RP._max_colsum_up(sparse(transpose(E)))))
    γN = N.γ
    η = add_up(λ, mul_up(normL, add_up(εL, mul_up(γN, add_up(1.0, εL)))))
    Amul(x) = apply!(N, Lhat * Vector{Float64}(x))
    Aadj(x) = Lt * apply!(N, Vector{Float64}(x))
    etaA = RP._eta_stored(n, w + N.nnz_per_row, Float64)
    a_lo = norm_lower(Amul, n, η, etaA)
    epsA = div_up(η, a_lo)
    op = LinOp(Amul, Aadj, n; epsA, etaA)
    return op, (; Lhat, Amul, Aadj, w, εL, normL, λ, γN, η, etaA, a_lo, epsA, nnzL = nnz(Lhat))
end
