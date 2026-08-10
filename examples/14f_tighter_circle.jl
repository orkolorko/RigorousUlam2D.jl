# Sharpen the certified mixing rate: sweep the gap-circle radius between |λ₂|≈0.695 and the
# certified 0.834, find the smallest radius that still certifies σ_min>0 (⇒ tighter rate bound).
# nsep is fixed at 1 (λ₁ the only eigenvalue outside); radius passed explicitly.
#   CACHED=1 DEPTH=6 julia --project=examples examples/14f_tighter_circle.jl
import Pkg
Pkg.activate(@__DIR__)
include(joinpath(@__DIR__, "ulam_matvec.jl"))
using RigorousPseudospectra, Printf

DEPTH = get(ENV, "DEPTH", "") == "" ? 6 : parse(Int, ENV["DEPTH"])
NTHETA = get(ENV, "NTHETA", "") == "" ? 512 : parse(Int, ENV["NTHETA"])
build = get(ENV, "CACHED", "") == "1" ? build_matvec_cached : build_matvec
NOISE = uniform_noise_box(0.1, 2)
d = IntervalBox(interval(-1.0, 1.0), interval(-1.0, 1.0))
L, La, n, _ = build(henon_rescaled, d, NOISE, DEPTH; periodic = true, enclosure_method = first_order_enclosure)
println("="^70); @printf("Hénon depth=%d n=%d — tighter gap circle (nθ=%d)\n", DEPTH, n, NTHETA); println("="^70)
println("|λ₂|≈0.695 sets the true mixing rate; the smallest certifying radius bounds it from above.")
for r in (0.834, 0.80, 0.78, 0.76, 0.74, 0.72, 0.70)
    fl, i = certify_gap_locked(L, La, n; nsep = 1, nblock = 256, radius = r,
        band = 0.2, nθ = NTHETA, kmax = 120, prec = Float64)
    @printf("  r=%.3f  σ_min≥%+.4e %s  (m=%d σ11=%.3e σ22=%.3e γ=%.2e)  ⇒ rate ≤ %.3f\n",
        r, fl, i.certified ? "✓" : "✗", i.m, i.σ11min, i.σ22min, i.γ, i.certified ? r : NaN)
    flush(stdout)
end
