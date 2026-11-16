# The ℓ¹ norms of the powers of the noisy Ulam operator P_h = π N_ξ π L π on the zero-sum vectors
# V_h, by columns (`norms_of_powers_columns`): C_j ≥ ‖P_h^j|_{V_h}‖₁ ≤ max_i ‖P_h^j e_i − u‖₁ + ε_j
# for j = 1, …, M, every column iterated in floating point with the ℓ¹ rounding model η₁ of
# `noisy_operator` (which covers the rounding of both products and the enclosure of L) and
# ‖P_h‖₁ ≤ ‖N‖₁ ‖L_lo + U‖₁ = ‖L_lo + U‖₁. The operator is the one of bench/certified_gap.jl and
# bench/stationary_density.jl (block tree_deepen.<map>_d<d>_level14_area30_xinv_t12, midpoint
# L̂ = L_lo + U/2, Ulam-projected uniform noise on [-a, a]², periodic). u is the computed density
# of the stationary_density record of the same (map, depth, a), or the power iteration if there
# is none; it is a free choice in the bound.
#
# From the C_j, refined by submultiplicativity (`refine_norms_of_powers`): Σ_grid ≥
# ‖(I − P_h)⁻¹|_{V_h}‖₁ (`sigma_from_norms`), Σ_V = 1 + Σ_grid, and with the stationary_density
# record the bounds of src/l1_error.jl: ‖f_ξ − f_h‖₁ ≤ Σ_V · 2δ var(ρ_ξ) and
# ‖f_ξ − f̃‖₁ ≤ that + ‖f̃ − f_h‖₁ (`fixed_point_distance`).
#
# Env: MAP, DEPTH, A (rational such as 51//512), M (largest power, 60), BLOCK (columns per block,
# 64), LEVEL (14), BISECT (30), TAG (_xinv_t12).  Output: one line per quantity on stdout and
#     ~/ResearchData/RigorousUlam2D_Data/data/l1_mixing.<map>_d<depth>_a<p>-<q>.<host>.jls
# (an existing file is not rewritten).
#
#   julia --project=. -t 16 bench/l1_mixing.jl

using RigorousUlam2D, IntervalArithmetic, IntervalBoxes
using LinearAlgebra, SparseArrays, Printf, Dates
using BallArithmetic: div_up, mul_up, sqrt_up, add_up
using RigorousPseudospectra: RigorousPseudospectra
import Serialization

const HOST = chomp(read(`hostname`, String))
const SCRIPT = splitext(basename(@__FILE__))[1]
const DATADIR = expanduser(get(ENV, "DATADIR", "~/ResearchData/RigorousUlam2D_Data/data"))
const MAP = get(ENV, "MAP", "henon")
const DEPTH = parse(Int, get(ENV, "DEPTH", "6"))
const A = let s = get(ENV, "A", "51//512"); p, q = parse.(Int, split(s, "//")); p // q end
const M = parse(Int, get(ENV, "M", "60"))
const BLOCK = parse(Int, get(ENV, "BLOCK", "64"))
const LEVEL = get(ENV, "LEVEL", "14")
const AREATAG = "area$(get(ENV, "BISECT", "30"))$(get(ENV, "TAG", "_xinv_t12"))"

gitrev(dir) = try chomp(read(`git -C $dir rev-parse --short HEAD`, String)) catch; "?" end
const COMMITS = (; RigorousUlam2D = gitrev(dirname(dirname(pathof(RigorousUlam2D)))),
    RigorousInvariantMeasures = gitrev(dirname(dirname(pathof(RigorousUlam2D.RIM)))),
    RigorousPseudospectra = gitrev(dirname(dirname(pathof(RigorousPseudospectra)))))

tag = "$(MAP)_d$(DEPTH)_a$(numerator(A))-$(denominator(A))"
out = joinpath(DATADIR, "$SCRIPT.$tag.$HOST.jls")
isfile(out) && error("$out exists")

dom = MAP == "henon" ? IntervalBox(interval(-1.0, 1.0), interval(-1.0, 1.0)) :
                       IntervalBox(interval(-0.5, 0.5), interval(-0.5, 0.5))
side = diam(dom[1])
k = 2^DEPTH
h = side / k                              # exact: a dyadic fraction
δ = mul_up(sqrt_up(2.0), h)              # the diameter of a cell, upward
varρ = div_up(2.0 * denominator(A), Float64(numerator(A)))   # var(ρ_ξ) = 2/a, upward

file = joinpath(DATADIR, "tree_deepen.$(MAP)_d$(DEPTH)_level$(LEVEL)_$(AREATAG).$HOST.jls")
t0 = time()
blk = Serialization.deserialize(file)
perm = grid_order(dom, DEPTH)
L_lo, U = blk.L_lo[perm, perm], blk.U[perm, perm]
blk = nothing
n = size(L_lo, 1)
N = SquareNoise(dom, DEPTH, Float64(A); bc = :periodic, slots = Threads.nthreads())
op, info = noisy_operator(L_lo, U, N)
normP = RigorousPseudospectra._max_colsum_up(dropzeros!(add_up.(L_lo, U)))
defect = defect_apriori(δ, varρ, varρ)

densfile = joinpath(DATADIR, "stationary_density.$tag.$HOST.jls")
dens = isfile(densfile) ? Serialization.deserialize(densfile) : nothing
dens === nothing || dens.n == n || error("$(basename(densfile)) is for n = $(dens.n), the operator has n = $n")
u = dens === nothing ? ulam_fixed_point(info.Amul, n).v : dens.density

println("$SCRIPT on $HOST, $(Dates.now()), julia $VERSION, threads $(Threads.nthreads()), commits $COMMITS")
@printf("%s depth %d n = %d, a = %s, h = %.6g, δ = %.6g, var(ρ_ξ) = %.6g, block %s\n", MAP, DEPTH, n, A, h, δ, varρ, basename(file))
@printf("operator: w %d, ‖L̂‖₁ ≤ %.6g, ‖E‖₁ ≤ %.3e, η₁ = %.3e, η₁ᴬ = %.3e, ‖P_h‖₁ ≤ %.12g; u from %s; M = %d, columns per block %d (%.1f s)\n",
    info.w, info.colL, info.colE, info.η1, info.eta1A, normP,
    dens === nothing ? "the power iteration" : basename(densfile), M, BLOCK, time() - t0)
flush(stdout)

tc = @elapsed C = norms_of_powers_columns(info.Bmul!, n, M; η1 = info.η1, eta1A = info.eta1A, normP, u, block = BLOCK)
Cr = refine_norms_of_powers(C, M)
@printf("columns: %d × %d products in %.1f s\n", n, M, tc)
for j in 1:M
    @printf("  j = %3d: C_j ≤ %.6g   refined %.6g\n", j, C[j], Cr[j])
end
s = any(<(1), Cr) ? sigma_from_norms(Cr) : nothing
if s === nothing
    println("no C_j below 1 up to j = $M")
    dist = err_apriori = err = NaN
else
    err_apriori = mul_up(s.V, defect)
    dist = dens === nothing ? NaN : fixed_point_distance(s.grid, dens.ε1, dens.ε2)
    err = dens === nothing ? NaN : add_up(err_apriori, dist)
    @printf("Σ_grid ≤ %.6g at k = %d (α = %.6g), Σ_V ≤ %.6g; 2δ var(ρ_ξ) = %.6g, ‖f_ξ − f_h‖₁ ≤ %.6g, ‖f̃ − f_h‖₁ ≤ %.3e, ‖f_ξ − f̃‖₁ ≤ %.6g\n",
        s.grid, s.k, s.α, s.V, defect, err_apriori, dist, err)
end
flush(stdout)

Serialization.serialize(out, (; map = MAP, depth = DEPTH, a = A, ξ = N.ξ, n, k, h, δ, varρ, block = basename(file),
    operator = (; info.w, info.colL, info.colE, info.γN, info.η1, info.eta1A, normP),
    u_from = dens === nothing ? "power iteration" : basename(densfile), M, columns_per_block = BLOCK,
    C, C_refined = Cr, sigma = s, defect, err_apriori, dist, err, time_columns = tc,
    host = HOST, script = SCRIPT, commits = COMMITS, julia = string(VERSION), threads = Threads.nthreads(), time = time() - t0))
println("written $(basename(out)), $(round(time() - t0; digits = 1)) s")
