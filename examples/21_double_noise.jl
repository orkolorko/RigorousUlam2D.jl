# Double the noise (a=0.2, box [-0.2,0.2]²) and re-measure R₁ on the circle around z=1
# for Hénon. Tests whether more noise reduces non-normality (smaller R₁) and, together
# with δ_h=δ/a halved, opens the gate R₁·δ_h<1 at a feasible depth.
import Pkg; Pkg.activate(@__DIR__)
include(joinpath(@__DIR__, "ulam_matvec.jl"))
using RigorousPseudospectra, Printf
using RigorousUlam2D: first_order_enclosure

A = get(ENV, "A", "") == "" ? 0.2 : parse(Float64, ENV["A"])   # noise half-width
NOISE = uniform_noise_box(A, 2)
dh = IntervalBox(interval(-1.0, 1.0), interval(-1.0, 1.0))
DEPTH = get(ENV, "DEPTH", "") == "" ? 6 : parse(Int, ENV["DEPTH"])
δ = 2.0 / 2^DEPTH; δh = δ / A
varρ = 2.0 / A

L, La, n, _ = build_matvec_cached(henon_rescaled, dh, NOISE, DEPTH;
    periodic = true, enclosure_method = first_order_enclosure)
@printf("Hénon depth=%d n=%d  noise a=%.2f (box[-%.2f,%.2f]²)  var(ρ)=%.1f  δ=%.5f  δ_h=δ/a=%.4f\n",
    DEPTH, n, A, A, A, varρ, δ, δh)
println("circle around z=1, sweep radius ρ₁:")
@printf("%6s %12s %8s %10s %10s %14s\n", "ρ₁", "σ_min", "cert", "R₁", "R₁·δ_h", "δ*=a/R₁ (→depth)")
rows = Tuple{Float64,Float64,Bool}[]
for ρ in (0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60)
    fl, info = certify_gap_locked(L, La, n; nsep = 1, center = 1.0, radius = ρ,
        nblock = 256, band = 0.2, nθ = 512, kmax = 120, prec = Float64)
    R1 = fl > 0 ? 1 / fl : Inf
    δstar = A / R1
    dstar = isfinite(δstar) && δstar > 0 ? ceil(log2(2.0 / δstar)) : NaN
    @printf("%6.3f %12.4e %8s %10.1f %10.2f %14s\n",
        ρ, fl, info.certified ? "✓" : "✗", R1, R1 * δh,
        isfinite(dstar) ? @sprintf("%.4f (d≈%d)", δstar, Int(dstar)) : "—")
    flush(stdout)
    push!(rows, (ρ, R1, info.certified))
end
cert = filter(r -> r[3] && isfinite(r[2]), rows)
if !isempty(cert)
    bρ, bR, _ = cert[argmin([r[2] for r in cert])]
    @printf("  ==> best: ρ₁=%.3f R₁=%.1f  R₁·δ_h=%.2f  gate closes at depth≈%d\n",
        bρ, bR, bR * δh, Int(ceil(log2(2.0 / (A / bR)))))
end
