# Confirm the optimal circle around z=1 (ρ₁≈0.17) transfers to finer depths, for Hénon.
# For depths 6 and 7, sweep ρ₁ near the depth-5 optimum and certify σ_min on Γ₁ → R₁.
# The optimal circle has a healthy σ_min (~0.04), so it certifies far faster than the
# auto radius (σ_min~0.01). δ_h = δ/a is the discretization defect; gate is R₁·δ_h<1.
import Pkg; Pkg.activate(@__DIR__)
include(joinpath(@__DIR__, "ulam_matvec.jl"))
using RigorousPseudospectra, Printf
using RigorousUlam2D: first_order_enclosure

NOISE = uniform_noise_box(0.1, 2)                 # a=0.1
dh = IntervalBox(interval(-1.0, 1.0), interval(-1.0, 1.0))
DEPTHS = get(ENV, "DEPTHS", "") == "" ? [6, 7] : parse.(Int, split(ENV["DEPTHS"], ","))
RADII = (0.14, 0.16, 0.17, 0.18, 0.20)

for depth in DEPTHS
    δ = 2.0 / 2^depth; δh = δ / 0.1
    L, La, n, _ = build_matvec_cached(henon_rescaled, dh, NOISE, depth;
        periodic = true, enclosure_method = first_order_enclosure)
    println("="^70)
    @printf("Hénon depth=%d n=%d  δ=%.5f  δ_h≈δ/a=%.4f\n", depth, n, δ, δh)
    @printf("%6s %12s %8s %10s %10s %12s\n", "ρ₁", "σ_min", "cert", "R₁", "R₁·δ_h", "ρ₁·R₁²")
    best = (R1 = Inf, ρ = 0.0)
    for ρ in RADII
        fl, info = certify_gap_locked(L, La, n; nsep = 1, center = 1.0, radius = ρ,
            nblock = 256, band = 0.2, nθ = 512, kmax = 120, prec = Float64)
        R1 = fl > 0 ? 1 / fl : Inf
        @printf("%6.3f %12.4e %8s %10.1f %10.2f %12.3e\n",
            ρ, fl, info.certified ? "✓" : "✗", R1, R1 * δh, ρ * R1^2)
        flush(stdout)
        if info.certified && R1 < best.R1
            best = (R1 = R1, ρ = ρ)
        end
    end
    if isfinite(best.R1)
        @printf("  ==> depth %d best: ρ₁=%.3f R₁=%.1f  R₁·δ_h=%.2f  (gate<1 needs δ<a/R₁=%.4f)\n",
            depth, best.ρ, best.R1, best.R1 * δh, 0.1 / best.R1)
    end
    flush(stdout)
end
