# Circle optimization around z=1 for Hénon (depth 6): sweep the radius ρ₁ of Γ₁={|z-1|=ρ₁},
# certify σ_min on it → R₁(ρ₁)=1/σ_min. Find the radius minimizing (a) R₁ for the gate
# R₁·δ_h<1, and (b) ρ₁·R₁² for the projector/error prefactor.
# ρ₁ must stay below the gap 1-|λ₂|≈0.305 so Γ₁ encloses only λ₁=1.
import Pkg; Pkg.activate(@__DIR__)
include(joinpath(@__DIR__, "ulam_matvec.jl"))
using RigorousPseudospectra, Printf
using RigorousUlam2D: first_order_enclosure

NOISE = uniform_noise_box(0.1, 2)                 # box [-0.1,0.1]², a=0.1, var(ρ)=20
dh = IntervalBox(interval(-1.0, 1.0), interval(-1.0, 1.0))
DEPTH = get(ENV, "DEPTH", "") == "" ? 6 : parse(Int, ENV["DEPTH"])
δ = 2.0 / 2^DEPTH
δh = δ / 0.1                                        # ≈ discretization defect δ/a

L, La, n, _ = build_matvec_cached(henon_rescaled, dh, NOISE, DEPTH;
    periodic = true, enclosure_method = first_order_enclosure)
@printf("Hénon depth=%d n=%d  δ=%.5f  δ_h≈δ/a=%.4f\n", DEPTH, n, δ, δh)
println("circle around z=1, sweep radius ρ₁ (gap 1-|λ₂|≈0.305):")
println("="^78)
@printf("%6s %12s %8s %10s %10s %12s\n", "ρ₁", "σ_min", "cert", "R₁", "R₁·δ_h", "ρ₁·R₁² (err)")
for ρ in (0.03, 0.05, 0.07, 0.09, 0.11, 0.14, 0.17, 0.20, 0.24, 0.28)
    fl, info = certify_gap_locked(L, La, n; nsep = 1, center = 1.0, radius = ρ,
        nblock = 256, band = 0.2, nθ = 512, kmax = 120, prec = Float64)
    R1 = fl > 0 ? 1 / fl : Inf
    @printf("%6.3f %12.4e %8s %10.1f %10.1f %12.3e\n",
        ρ, fl, info.certified ? "✓" : "✗", R1, R1 * δh, ρ * R1^2)
    flush(stdout)
end
