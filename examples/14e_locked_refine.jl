# Push the locked gap certificate: (a) nsep=1 with finer angular sampling nθ (the arc-cover
# correction 2r·sin(π/2nθ) is what pinned the floor negative; the pointwise floor is positive);
# (b) small circles with a WIDER band so the deflated complement is emptied of the minimizer.
#   CACHED=1 DEPTH=6 julia --project=examples examples/14e_locked_refine.jl
import Pkg
Pkg.activate(@__DIR__)
include(joinpath(@__DIR__, "ulam_matvec.jl"))
using RigorousPseudospectra, Printf

DEPTH = get(ENV, "DEPTH", "") == "" ? 6 : parse(Int, ENV["DEPTH"])
build = get(ENV, "CACHED", "") == "1" ? build_matvec_cached : build_matvec
NOISE = uniform_noise_box(0.1, 2)
d = IntervalBox(interval(-1.0, 1.0), interval(-1.0, 1.0))
L, La, n, _ = build(henon_rescaled, d, NOISE, DEPTH; periodic = true, enclosure_method = first_order_enclosure)
println("="^70); @printf("Hénon depth=%d n=%d\n", DEPTH, n); println("="^70)

KMAX = get(ENV, "KMAX", "") == "" ? 120 : parse(Int, ENV["KMAX"])
println("(a) nsep=1, vary nθ (band=0.2, kmax=$KMAX):")
for nθ in (64, 128, 256, 512)
    fl, i = certify_gap_locked(L, La, n; nsep = 1, nblock = 256, band = 0.2, nθ = nθ, kmax = KMAX, prec = Float64)
    @printf("   nθ=%4d  r=%.4f  σ≥%+.4e %s  (m=%d σ11=%.3e σ22=%.3e γ=%.2e)\n",
        nθ, i.radius, fl, i.certified ? "✓" : "✗", i.m, i.σ11min, i.σ22min, i.γ)
    flush(stdout)
end

println("(b) nsep=10 (r≈0.38), vary band (nθ=256, kmax=$KMAX):")
for band in (0.2, 0.4, 0.6, 0.8)
    fl, i = certify_gap_locked(L, La, n; nsep = 10, nblock = 256, band = band, nθ = 256, kmax = KMAX, prec = Float64)
    @printf("   band=%.2f  σ≥%+.4e %s  (m=%d σ11=%.3e σ22=%.3e γ=%.2e R2=%.2e)\n",
        band, fl, i.certified ? "✓" : "✗", i.m, i.σ11min, i.σ22min, i.γ, i.R2)
    flush(stdout)
end
