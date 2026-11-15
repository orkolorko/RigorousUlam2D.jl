# The computed stationary density of the noisy Ulam operator P_h = π N_ξ π L π of one map at one
# depth and one noise amplitude, with its a posteriori quantities: the operator is the one of
# bench/certified_gap.jl (deterministic block tree_deepen.<map>_d<d>_level14_area30_xinv_t12,
# midpoint L̂ = L_lo + U/2 in the products, Ulam-projected uniform noise on [-a, a]², periodic);
# the density f̃ is the power iteration `ulam_fixed_point` on the masses; then
#
# - ε₁ ≥ ‖P_h f̃ − f̃‖₁ and ε₂ ≥ |Σ f̃ − 1| (`fixed_point_residual`, with the ℓ¹ rounding model
#   η₁ of `noisy_operator`, which covers the rounding of the products and the enclosure of L),
# - var(f̃), the variation on the torus of the piecewise-constant density (`grid_variation`),
# - the diameter δ = √2 h of a cell and the a priori defect 2δ var(ρ_ξ) of l1_error.jl,
#
# and, when the certified_gap record of the same (map, depth, a) exists, the bounds Σ on the
# resolvent at 1 (`sigma_from_gap` on every certified radius, the smallest kept, and
# `sigma_from_circle` on every certified circle), the distance ‖f̃ − f_h‖₁
# (`fixed_point_distance`) and the L¹ error ‖f_ξ − f̃‖₁ ≤ Σ_V · 2δ var(ρ_ξ) + ‖f̃ − f_h‖₁.
#
# Env: MAP, DEPTH, A (rational such as 51//512), LEVEL (14), BISECT (30), TAG (_xinv_t12),
# TOL (1e-14), MAXITER (20000).  Output: one line per quantity on stdout and
#     ~/ResearchData/RigorousUlam2D_Data/data/stationary_density.<map>_d<depth>_a<p>-<q>.<host>.jls
# with the density and every number (an existing file is not rewritten).
#
#   julia --project=. -t 16 bench/stationary_density.jl

using RigorousUlam2D, IntervalArithmetic, IntervalBoxes
using LinearAlgebra, SparseArrays, Printf, Dates
using BallArithmetic: div_up, mul_up, sqrt_up, add_up
import Serialization

const HOST = chomp(read(`hostname`, String))
const SCRIPT = splitext(basename(@__FILE__))[1]
const DATADIR = expanduser(get(ENV, "DATADIR", "~/ResearchData/RigorousUlam2D_Data/data"))
const MAP = get(ENV, "MAP", "henon")
const DEPTH = parse(Int, get(ENV, "DEPTH", "6"))
const A = let s = get(ENV, "A", "51//512"); p, q = parse.(Int, split(s, "//")); p // q end
const LEVEL = get(ENV, "LEVEL", "14")
const AREATAG = "area$(get(ENV, "BISECT", "30"))$(get(ENV, "TAG", "_xinv_t12"))"
const TOL = parse(Float64, get(ENV, "TOL", "1e-14"))
const MAXITER = parse(Int, get(ENV, "MAXITER", "20000"))

gitrev(dir) = try chomp(read(`git -C $dir rev-parse --short HEAD`, String)) catch; "?" end
const COMMITS = (; RigorousUlam2D = gitrev(dirname(dirname(pathof(RigorousUlam2D)))),
    RigorousInvariantMeasures = gitrev(dirname(dirname(pathof(RigorousUlam2D.RIM)))))

tag = "$(MAP)_d$(DEPTH)_a$(numerator(A))-$(denominator(A))"
out = joinpath(DATADIR, "$SCRIPT.$tag.$HOST.jls")
isfile(out) && error("$out exists")

dom = MAP == "henon" ? IntervalBox(interval(-1.0, 1.0), interval(-1.0, 1.0)) :
                       IntervalBox(interval(-0.5, 0.5), interval(-0.5, 0.5))
side = diam(dom[1])                       # 2 for Hénon, 1 for Lorenz; |M| = side²
area = side * side
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
println("$SCRIPT on $HOST, $(Dates.now()), julia $VERSION, threads $(Threads.nthreads()), commits $COMMITS")
@printf("%s depth %d n = %d, a = %s, h = %.6g, δ = %.6g, var(ρ_ξ) = %.6g, block %s\n", MAP, DEPTH, n, A, h, δ, varρ, basename(file))
@printf("operator: nnz(L̂) %d, w %d, ‖L̂‖₁ ≤ %.6g, ‖E‖₁ ≤ %.3e, γ_N = %.3e, η₁ = %.3e, η₁ᴬ = %.3e (%.1f s)\n",
    info.nnzL, info.w, info.colL, info.colE, info.γN, info.η1, info.eta1A, time() - t0)
flush(stdout)

tp = @elapsed fp = ulam_fixed_point(info.Amul, n; tol = TOL, maxiter = MAXITER)
@printf("power iteration: %d iterations, last ℓ¹ change %.3e, converged %s, min %.3e, max %.3e (%.1f s)\n",
    fp.iterations, fp.diff, fp.converged, minimum(fp.v), maximum(fp.v), tp)
res = fixed_point_residual(info.Amul, fp.v, info.η1, info.eta1A)
varf = grid_variation(fp.v, k, h)
defect = defect_apriori(δ, varρ, varρ)
@printf("residual ε₁ ≤ %.3e, mass defect ε₂ ≤ %.3e, var(f̃) ≤ %.6g (var(ρ_ξ) = %.6g), a priori defect 2δ var(ρ_ξ) = %.6g\n",
    res.ε1, res.ε2, varf, varρ, defect)
flush(stdout)

# the certified spectral data of the same operator, when they exist
gapfile = joinpath(DATADIR, "certified_gap.$tag.$HOST.jls")
sigmas = nothing
if isfile(gapfile)
    g = Serialization.deserialize(gapfile)
    g.n == n || error("$(basename(gapfile)) is for n = $(g.n), the operator has n = $n")
    gap = NamedTuple[]
    for x in g.radii
        (x.ok && isfinite(x.floor) && x.floor > 0 && x.R < 1) || continue
        K = mul_up(mul_up(mul_up(sqrt_up(area), RigorousUlam2D.C_M_UP), varρ), div_up(1.0, x.floor))
        s = sigma_from_gap(K, x.R)
        push!(gap, (; R = x.R, ks = x.ks, dimQ = x.dimQ, floor = x.floor, K, k = s.k, α = s.α, grid = s.grid, V = s.V))
    end
    circ = NamedTuple[]
    for c in g.circles
        (c.ok && c.proj_ok && c.proj_count == 1 && isfinite(c.R1)) || continue
        s = sigma_from_circle(area, varρ, c.R1)
        push!(circ, (; ρ = c.ρ, R1 = c.R1, grid = s.grid, V = s.V))
    end
    bg = isempty(gap) ? nothing : gap[argmin([x.grid for x in gap])]
    bc = isempty(circ) ? nothing : circ[argmin([x.grid for x in circ])]
    best = bg === nothing ? bc : (bc === nothing || bg.grid <= bc.grid ? bg : bc)
    if best !== nothing
        dist = fixed_point_distance(best.grid, res.ε1, res.ε2)
        err_apriori = mul_up(best.V, defect)
        err = add_up(err_apriori, dist)
        global sigmas = (; gap, circles = circ, best, dist, err_apriori, err)
        bg === nothing || @printf("Σ from the gap: r = %.6g, R_Γ = %.6g, K = %.6g, k = %d, α = %.3g: Σ_grid ≤ %.6g, Σ_V ≤ %.6g\n",
            bg.R, 1 / bg.floor, bg.K, bg.k, bg.α, bg.grid, bg.V)
        bc === nothing || @printf("Σ from the circle |z − 1| = %.3g, R₁ = %.6g: Σ ≤ %.6g\n", bc.ρ, bc.R1, bc.grid)
        @printf("‖f̃ − f_h‖₁ ≤ %.3e, ‖f_ξ − f_h‖₁ ≤ Σ_V · 2δ var(ρ_ξ) = %.6g, ‖f_ξ − f̃‖₁ ≤ %.6g\n", dist, err_apriori, err)
    else
        println("$(basename(gapfile)) holds no certified radius or circle")
    end
else
    println("no certified_gap record for $tag on $HOST: Σ not available at this depth")
end
flush(stdout)

Serialization.serialize(out, (; map = MAP, depth = DEPTH, a = A, ξ = N.ξ, n, k, h, δ, area, varρ, block = basename(file),
    operator = (; info.w, info.colL, info.colE, info.γN, info.η1, info.eta1A, info.η, info.nnzL),
    density = fp.v, iterations = fp.iterations, diff = fp.diff, converged = fp.converged, tol = TOL,
    ε1 = res.ε1, ε2 = res.ε2, varf, defect, sigmas, gapfile = isfile(gapfile) ? basename(gapfile) : nothing,
    host = HOST, script = SCRIPT, commits = COMMITS, julia = string(VERSION), threads = Threads.nthreads(), time = time() - t0))
println("written $(basename(out)), $(round(time() - t0; digits = 1)) s")
