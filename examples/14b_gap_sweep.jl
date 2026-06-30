# Gap-only sweep: certify a circle that leaves the leading `nsep` eigenvalues OUTSIDE,
# for nsep ∈ {1,5,10}. No eigenvalue re-cert, no plotting. Answers: can we certify an
# exclusion circle SMALLER than all 10 leading eigenvalues (nsep=10)?
#   CACHED=1 DEPTH=6 EX=henon julia --project=examples --threads=auto examples/14b_gap_sweep.jl
import Pkg
Pkg.activate(@__DIR__)
include(joinpath(@__DIR__, "ulam_matvec.jl"))
using RigorousPseudospectra, Printf

DEPTH = get(ENV, "DEPTH", "") == "" ? 6 : parse(Int, ENV["DEPTH"])
BITS = get(ENV, "BITS", "") == "" ? 192 : parse(Int, ENV["BITS"]); setprecision(BigFloat, BITS)
EX = get(ENV, "EX", "henon")
build = get(ENV, "CACHED", "") == "1" ? build_matvec_cached : build_matvec
NOISE = uniform_noise_box(0.1, 2)
NBLOCK = get(ENV, "NBLOCK", "") == "" ? 256 : parse(Int, ENV["NBLOCK"])

if EX == "henon"
    d = IntervalBox(interval(-1.0, 1.0), interval(-1.0, 1.0))
    L, La, n, _ = build(henon_rescaled, d, NOISE, DEPTH; periodic = true, enclosure_method = first_order_enclosure)
    prec = Float64
else
    d = IntervalBox(interval(-0.5, 0.5), interval(-0.5, 0.5))
    L, La, n, _ = build(contracting_lorenz, d, NOISE, DEPTH; periodic = false, enclosure_method = zeroth_order_enclosure)
    prec = BigFloat
end

println("="^70); println("$EX  depth=$DEPTH  n=$n  prec=$prec  nblock=$NBLOCK"); println("="^70)
for nsep in (1, 5, 10)
    fl, info = certify_gap(L, La, n; nsep = nsep, nblock = min(NBLOCK, n - 2), nθ = 64, prec = prec)
    @printf("  nsep=%2d  radius=%.5f  σ_min≥%+.4e  %s  (resolvent ≤ %.1f)\n",
        nsep, info.radius, fl, info.certified ? "✓" : "✗", fl > 0 ? 1 / fl : Inf)
    flush(stdout)
end
