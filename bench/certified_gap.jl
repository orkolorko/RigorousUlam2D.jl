# The certified spectral gap and resolvent of the noisy Ulam operator P_δ = π N_ξ π L π of one map
# at one depth and one noise amplitude: the deterministic block L_lo ≤ L ≤ L_lo + U read from the
# tree_deepen files, the Ulam projection of the uniform noise on the square [-a, a]² as the product
# of RIM's one-dimensional kernels, the two assembled as the LinOp of `noisy_operator` (whose
# rounding model has P_δ itself as the certified operator), and RigorousPseudospectra on it:
#
# - the eigenvalues that place the radii, in floating point and no part of the certificate: dense
#   when n ≤ DENSEMAX, else Ritz values of a short Arnoldi;
# - for R = f |λ₂| over the factors f of RLIST, largest first, the two-sided frame of def:Q with ks
#   singular vectors per side (ks = 0, 8, 16, … up to KSMAX, then bisected back to the first that
#   passes, as bench/bz/exp5_bz.jl of RigorousPseudospectra does), `certify_split` at order 8, and
#   `count_exterior`: the exterior cover of |z| = R with count 1, i.e. the only eigenvalue of P_δ
#   of modulus > R is 1, so |λ₂| ≤ R; the ladder stops at the first radius that fails;
# - on the frame of the smallest certified R: `certify_exterior`, the isolation disc of the
#   eigenvalue outside R (compared with the dense eigenvalues when there are any); `check_circle`
#   on |z − 1| = ρ for every ρ of RHO (region :interior, which needs 1 − ρ > ḡ), whose `floor` is a
#   certified lower bound on 1/‖(P_δ − z)⁻¹‖₂ on the whole circle, so R₁ = 1/floor; and
#   `certify_projector` there, with the number of eigenvalues inside the circle.
#
# Env: MAP (henon|lorenz), DEPTH, A (half-width of the noise box, a rational such as 51//512),
# LEVEL (14), BISECT (30), TAG (_xinv_t12): the tree_deepen file read; RLIST, RHO, KSMAX (256),
# LRITZ (300), NTHETA (64), MAXDEPTH (64), DENSEMAX (4096), NRADII (12).  Output: one line per step on stdout and
#     ~/ResearchData/RigorousUlam2D_Data/data/certified_gap.<map>_d<depth>_a<p>-<q>.<host>.jls
# with everything computed (an existing file is not rewritten).
#
#   julia --project=. -t 32 bench/certified_gap.jl

using RigorousUlam2D, RigorousPseudospectra, IntervalArithmetic, IntervalBoxes
using LinearAlgebra, SparseArrays, Printf, Dates
import Serialization
const RP = RigorousPseudospectra

const HOST = chomp(read(`hostname`, String))
const SCRIPT = splitext(basename(@__FILE__))[1]
const DATADIR = expanduser(get(ENV, "DATADIR", "~/ResearchData/RigorousUlam2D_Data/data"))
const MAP = get(ENV, "MAP", "henon")
const DEPTH = parse(Int, get(ENV, "DEPTH", "6"))
const A = let s = get(ENV, "A", "51//512"); p, q = parse.(Int, split(s, "//")); p // q end
const LEVEL = get(ENV, "LEVEL", "14")
const AREATAG = "area$(get(ENV, "BISECT", "30"))$(get(ENV, "TAG", "_xinv_t12"))"
const DENSEMAX = parse(Int, get(ENV, "DENSEMAX", "4096"))
const NRADII = parse(Int, get(ENV, "NRADII", "12"))
const RLIST = parse.(Float64, split(get(ENV, "RLIST", "2,1.5,1.25,1.1,1.05,1.02"), ","))
const RHO = parse.(Float64, split(get(ENV, "RHO", "0.5,0.4,0.3,0.25,0.2,0.15,0.1"), ","))
const KSMAX = parse(Int, get(ENV, "KSMAX", "256"))
const LRITZ = parse(Int, get(ENV, "LRITZ", "300"))
const NTHETA = parse(Int, get(ENV, "NTHETA", "64"))
const MAXDEPTH = parse(Int, get(ENV, "MAXDEPTH", "64"))

gitrev(dir) = try chomp(read(`git -C $dir rev-parse --short HEAD`, String)) catch; "?" end
const COMMITS = (; RigorousUlam2D = gitrev(dirname(dirname(pathof(RigorousUlam2D)))),
    RigorousPseudospectra = gitrev(dirname(dirname(pathof(RigorousPseudospectra)))),
    RigorousInvariantMeasures = gitrev(dirname(dirname(pathof(RigorousUlam2D.RIM)))))

out = joinpath(DATADIR, "$SCRIPT.$(MAP)_d$(DEPTH)_a$(numerator(A))-$(denominator(A)).$HOST.jls")
isfile(out) && error("$out exists")

dom = MAP == "henon" ? IntervalBox(interval(-1.0, 1.0), interval(-1.0, 1.0)) :
                       IntervalBox(interval(-0.5, 0.5), interval(-0.5, 0.5))
file = joinpath(DATADIR, "tree_deepen.$(MAP)_d$(DEPTH)_level$(LEVEL)_$(AREATAG).$HOST.jls")
t0 = time()
blk = Serialization.deserialize(file)
perm = grid_order(dom, DEPTH)
L_lo, U = blk.L_lo[perm, perm], blk.U[perm, perm]
n = size(L_lo, 1)
N = SquareNoise(dom, DEPTH, Float64(A); bc = :periodic, slots = Threads.nthreads())
op, info = noisy_operator(L_lo, U, N)
Amul, Aadj = info.Amul, info.Aadj
println("$SCRIPT on $HOST, $(Dates.now()), julia $VERSION, threads $(Threads.nthreads()), commits $COMMITS")
@printf("%s depth %d n = %d, a = %s = %.6f, ξ = %s of the sides (s = %.4g cells), block %s (‖U‖₁ %.3e ‖U‖_∞ %.3e)\n",
    MAP, DEPTH, n, A, Float64(A), N.ξ, N.ξ[1] * 2^DEPTH, basename(file),
    maximum(sum(U; dims = 1)), maximum(sum(U; dims = 2)))
@printf("operator: nnz(L̂) %d, w %d, ‖L̂‖ ≤ %.6g, λ = %.3e, ε_L = %.3e, γ_N = %.3e, η = %.3e, a_lo = %.6g, ε_A = %.3e, η_A = %.3e (%.1f s)\n",
    info.nnzL, info.w, info.normL, info.λ, info.εL, info.γN, info.η, info.a_lo, info.epsA, info.etaA, time() - t0)
flush(stdout)

# the eigenvalues that place the radii: dense when the operator is small, else Ritz values
const dense = n <= DENSEMAX
tr = @elapsed lam = if dense
    Ad = zeros(n, n)
    for j in 1:n
        e = zeros(n); e[j] = 1.0
        Ad[:, j] = Amul(e)
    end
    λ = eigvals(Ad)
    λ[sortperm(abs.(λ); rev = true)]
else
    RP._ritz_leading(Amul, n, Float64; L = LRITZ)
end
@printf("%s moduli (%s, %.1f s): %s\n", dense ? "dense" : "Ritz", dense ? "n = $n" : "L = $LRITZ", tr,
    join((@sprintf("%.6f", abs(z)) for z in lam[1:min(8, end)]), " "))
flush(stdout)
λ2 = abs(lam[2])

# One frame and one exterior cover at the radius R with ks singular vectors per side.
function cover_at(R, ks)
    t = @elapsed begin
        Q, _ = RP.two_sided_frame(Amul, Aadj, n, R; ks, Leig = 200, Lsv = min(n, 2max(ks, 1) + 60), T = Float64)
        sc = RP.certify_split(op, Q; order = 8, target = R, compression = :full)
        if sc.gbar < R
            cnt, cok, e = RP.count_exterior(sc, R; nθ = NTHETA, maxdepth = MAXDEPTH, method = :svd)
            ok = e.ok && e.region_ok && cok && cnt == 1
        else
            cnt, cok, e, ok = -1, false, nothing, false
        end
    end
    @printf("   R = %.6f ks = %d dim Q = %d ḡ = %.6g γ1γ2 = %.3e count %d gap certified %s%s (%.1f s)\n",
        R, ks, sc.k, sc.gbar, sc.γ1 * sc.γ2, cnt, ok, e === nothing ? "" : " " * string(e.reason), t)
    flush(stdout)
    return (; R, ks, sc, ok, cnt, e, t)
end

# The smallest ks that passes at R, by doubling then bisection; `nothing` if none up to KSMAX.
function frame_search(R)
    cache = Dict{Int,Any}()
    f(ks) = get!(() -> cover_at(R, ks), cache, ks)
    lo, hi = -1, 0
    while !f(hi).ok && hi < KSMAX
        lo = hi
        hi = hi == 0 ? 8 : min(2hi, KSMAX)
    end
    f(hi).ok || return nothing, collect(values(cache))
    if lo >= 0
        while hi - lo > 1
            mid = (lo + hi) ÷ 2
            f(mid).ok ? (hi = mid) : (lo = mid)
        end
    end
    return f(hi), collect(values(cache))
end

radii = NamedTuple[]
best = nothing
for f in RLIST
    R = f * λ2
    R < 1 || continue
    @printf("R = %.6f (%.3g |λ₂|)\n", R, f)
    b, tried = frame_search(R)
    for v in tried
        push!(radii, (; R = v.R, ks = v.ks, dimQ = v.sc.k, gbar = v.sc.gbar, γ1 = v.sc.γ1, γ2 = v.sc.γ2,
            count = v.cnt, ok = v.ok, reason = v.e === nothing ? :gbar : v.e.reason,
            floor = v.e === nothing ? NaN : v.e.floor, t = v.t))
    end
    b === nothing && break
    global best = b
end

circles = NamedTuple[]
discs = NamedTuple[]
ext = nothing
if best !== nothing
    sc = best.sc
    @printf("certified: the only eigenvalue of modulus > %.6f is 1 (frame ks = %d, dim Q = %d, ḡ = %.6g)\n",
        best.R, best.ks, sc.k, sc.gbar)
    # the isolation discs of the eigenvalues outside R (here the eigenvalue 1), and the dense check
    td = @elapsed ext = RP.certify_exterior(sc, best.R; nθ = NTHETA, nradii = NRADII, maxdepth = MAXDEPTH, method = :svd)
    outside = dense ? [z for z in lam if abs(z) > best.R] : ComplexF64[]
    for dsc in ext.discs
        inside = dense && isfinite(dsc.radius) ? count(z -> abs(z - dsc.center) < dsc.radius, outside) : -1
        push!(discs, (; center = dsc.center, radius = dsc.radius, count = dsc.count, dense_inside = inside))
    end
    @printf("   exterior: count %d, %d isolation disc(s) %s%s (%.1f s)\n", ext.count, length(ext.discs),
        join((@sprintf("|z − (%.6f%+.6fi)| ≤ %.3e holding %d", real(d.center), imag(d.center), d.radius, d.count) for d in discs), "; "),
        dense ? ", dense eigenvalues inside as counted: $(all(d -> d.dense_inside == d.count, discs))" : "", td)
    flush(stdout)
    for ρ in RHO
        t = @elapsed chk = RP.check_circle(sc, 1.0, ρ; region = :interior, nθ = NTHETA, maxdepth = MAXDEPTH, method = :svd)
        R1 = chk.ok && chk.floor > 0 ? 1 / chk.floor : Inf
        tp = @elapsed pc = chk.ok ? RP.certify_projector(sc, 1.0, ρ; region = :interior, nθ = NTHETA, maxdepth = MAXDEPTH) : nothing
        @printf("   |z − 1| = %.3f: %s %s floor %.6g  R₁ ≤ %.6g  min σ11 %.6g min σ22 %.6g; projector %s count %s bound %s (%.1f s)\n",
            ρ, chk.ok, chk.reason, chk.floor, R1, chk.minσ11, chk.minσ22,
            pc === nothing ? "-" : string(pc.ok), pc === nothing ? "-" : string(pc.count),
            pc === nothing ? "-" : @sprintf("%.3e", pc.bound), t + tp)
        flush(stdout)
        push!(circles, (; ρ, ok = chk.ok, region_ok = chk.region_ok, reason = chk.reason, floor = chk.floor, R1,
            minσ11 = chk.minσ11, minσ22 = chk.minσ22, nevals = chk.nevals,
            proj_ok = pc === nothing ? false : pc.ok, proj_count = pc === nothing ? -1 : pc.count,
            proj_bound = pc === nothing ? NaN : pc.bound, t = t + tp))
    end
else
    println("no radius of the ladder was certified")
end

Serialization.serialize(out, (; map = MAP, depth = DEPTH, a = A, ξ = N.ξ, level = LEVEL, areatag = AREATAG, n,
    block = basename(file), operator = (; info.w, info.normL, info.λ, info.εL, info.γN, info.η, info.a_lo, info.epsA, info.etaA, info.nnzL),
    dense, eigenvalues = lam[1:min(20, end)], radii, best = best === nothing ? nothing : (; best.R, best.ks, dimQ = best.sc.k, best.sc.gbar, best.sc.γ1, best.sc.γ2),
    exterior = ext === nothing ? nothing : (; ext.ok, ext.count, discs), circles, host = HOST, script = SCRIPT, commits = COMMITS, julia = string(VERSION), threads = Threads.nthreads(),
    time = time() - t0))
println("written $(basename(out)), $(round(time() - t0; digits = 1)) s")
