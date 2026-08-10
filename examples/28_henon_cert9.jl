# R₁ for periodic Hénon a=0.3 certified at depth 9, single radius near the depth-7 optimum
# (ρ₁≈0.28), to get a small gate R₁·δ_h(9) and hence a tight Σ_res for the a-posteriori bound.
import Pkg; Pkg.activate(@__DIR__)
include(joinpath(@__DIR__, "ulam_matvec.jl"))
using RigorousPseudospectra, Printf
using RigorousUlam2D: first_order_enclosure

A = 0.3; NOISE = uniform_noise_box(A, 2)
dh = IntervalBox(interval(-1.0, 1.0), interval(-1.0, 1.0))
DEPTH = get(ENV, "DEPTH", "") == "" ? 9 : parse(Int, ENV["DEPTH"])
δ = 2.0 / 2^DEPTH; δh = δ / A
L, La, n, _ = build_matvec_cached(henon_rescaled, dh, NOISE, DEPTH; periodic = true, enclosure_method = first_order_enclosure)
@printf("Hénon depth=%d n=%d a=%.1f δ_h=%.4f\n", DEPTH, n, A, δh); flush(stdout)
for ρ in (0.28,)
    fl, info = certify_gap_locked(L, La, n; nsep = 1, center = 1.0, radius = ρ,
        nblock = 128, band = 0.2, nθ = 160, kmax = 60, prec = Float64)
    R1 = fl > 0 ? 1 / fl : Inf
    @printf("  ρ₁=%.3f σ_min=%.4e %s R₁=%.1f  R₁·δ_h=%.3f  a₂=R₁/(1-R₁δ_h)=%.1f\n",
        ρ, fl, info.certified ? "✓" : "✗", R1, R1 * δh, R1 / (1 - R1 * δh))
    flush(stdout)
end
