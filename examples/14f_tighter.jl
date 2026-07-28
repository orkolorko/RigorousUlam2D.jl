# Tighter Hénon gap circles to sharpen the certified mixing rate.
#  (a) nsep=1 with hand-picked radius r just above |λ₂|=0.695 → sharp rate bound.
#  (b) nsep=3 (peel λ₁ + the 0.695 conjugate pair), radius in the (0.675,0.695) annulus
#      → sub-leading decay rate (remainder after the top-3 modes).
#   CACHED=1 DEPTH=6 julia --project=examples examples/14f_tighter.jl
import Pkg
Pkg.activate(@__DIR__)
include(joinpath(@__DIR__, "ulam_matvec.jl"))
using RigorousPseudospectra, Printf
build = get(ENV, "CACHED", "") == "1" ? build_matvec_cached : build_matvec
NOISE = uniform_noise_box(0.1, 2)
DEPTH = get(ENV, "DEPTH", "") == "" ? 6 : parse(Int, ENV["DEPTH"])
d = IntervalBox(interval(-1.0, 1.0), interval(-1.0, 1.0))
L, La, n, _ = build(henon_rescaled, d, NOISE, DEPTH; periodic = true, enclosure_method = first_order_enclosure)
println("="^70); @printf("Hénon depth=%d n=%d — tighter mixing-rate circles\n", DEPTH, n); println("="^70)

println("(a) sharp rate: nsep=1, radius just above |λ₂|≈0.695:")
for (r, nθ) in ((0.75, 512), (0.72, 768), (0.71, 1024))
    fl, i = certify_gap_locked(L, La, n; nsep = 1, nblock = 256, radius = r, band = 0.25,
        nθ = nθ, kmax = 120, prec = Float64)
    @printf("   r=%.3f nθ=%4d  σ≥%+.4e %s  rate≤%.3f  (m=%d σ11=%.3e σ22=%.3e γ=%.2e)\n",
        r, nθ, fl, i.certified ? "✓" : "✗", r, i.m, i.σ11min, i.σ22min, i.γ)
    flush(stdout)
end

println("(b) sub-leading: nsep=3 (peel the 0.695 pair), radius in (0.675,0.695):")
for (r, nθ) in ((0.688, 1024), (0.685, 1536))
    fl, i = certify_gap_locked(L, La, n; nsep = 3, nblock = 256, radius = r, band = 0.25,
        nθ = nθ, kmax = 120, prec = Float64)
    @printf("   r=%.3f nθ=%4d  σ≥%+.4e %s  (m=%d σ11=%.3e σ22=%.3e γ=%.2e)\n",
        r, nθ, fl, i.certified ? "✓" : "✗", i.m, i.σ11min, i.σ22min, i.γ)
    flush(stdout)
end
