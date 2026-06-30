# Compare the LOCKED gap certificate (singular-direction lock) vs the committed block-assembly,
# on the Hénon gap circle, at nsep ∈ {1,5,10}. The locked variant captures the near-circle band
# H_kk exactly (small svdbox) and certifies the deflated complement soundly.
#   CACHED=1 DEPTH=6 julia --project=examples examples/14d_locked_gap.jl
import Pkg
Pkg.activate(@__DIR__)
include(joinpath(@__DIR__, "ulam_matvec.jl"))
using RigorousPseudospectra, Printf

DEPTH = get(ENV, "DEPTH", "") == "" ? 6 : parse(Int, ENV["DEPTH"])
BAND = get(ENV, "BAND", "") == "" ? 0.2 : parse(Float64, ENV["BAND"])
build = get(ENV, "CACHED", "") == "1" ? build_matvec_cached : build_matvec
NOISE = uniform_noise_box(0.1, 2)

d = IntervalBox(interval(-1.0, 1.0), interval(-1.0, 1.0))
L, La, n, _ = build(henon_rescaled, d, NOISE, DEPTH; periodic = true, enclosure_method = first_order_enclosure)
println("="^70); @printf("Hénon depth=%d n=%d  band=%.2f\n", DEPTH, n, BAND); println("="^70)

for nsep in (1, 5, 10)
    flb, ib = certify_gap(L, La, n; nsep = nsep, nblock = 256, nθ = 64, prec = Float64)
    fll, il = certify_gap_locked(L, La, n; nsep = nsep, nblock = 256, band = BAND, nθ = 64, prec = Float64)
    @printf("nsep=%2d r=%.4f | block-assembly σ≥%+.4e %s | LOCKED σ≥%+.4e %s (m=%d σ11=%.3e σ22=%.3e γ=%.2e R2=%.2e)\n",
        nsep, ib.radius, flb, ib.certified ? "✓" : "✗",
        fll, il.certified ? "✓" : "✗", il.m, il.σ11min, il.σ22min, il.γ, il.R2)
    flush(stdout)
end
