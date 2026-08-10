# Fine radius sweep around ρ₁≈0.30 for a=0.3 Hénon, at depths 6 AND 7 — to find the true
# optimum and check it's not a depth-6 resonance (cf. the a=0.1 depth-5 fluke).
import Pkg; Pkg.activate(@__DIR__)
include(joinpath(@__DIR__, "ulam_matvec.jl"))
using RigorousPseudospectra, Printf
using RigorousUlam2D: first_order_enclosure

A = 0.3
NOISE = uniform_noise_box(A, 2)
dh = IntervalBox(interval(-1.0, 1.0), interval(-1.0, 1.0))
DEPTHS = get(ENV, "DEPTHS", "") == "" ? [6, 7] : parse.(Int, split(ENV["DEPTHS"], ","))
RADII = (0.22, 0.25, 0.28, 0.30, 0.32, 0.35, 0.38)

for depth in DEPTHS
    δ = 2.0 / 2^depth; δh = δ / A
    L, La, n, _ = build_matvec_cached(henon_rescaled, dh, NOISE, depth;
        periodic = true, enclosure_method = first_order_enclosure)
    println("="^68)
    @printf("Hénon depth=%d n=%d  a=%.2f  δ=%.5f  δ_h=δ/a=%.4f\n", depth, n, A, δ, δh)
    @printf("%6s %12s %8s %10s %10s %14s\n", "ρ₁", "σ_min", "cert", "R₁", "R₁·δ_h", "δ*=a/R₁(→d)")
    rows = Tuple{Float64,Float64,Bool}[]
    for ρ in RADII
        fl, info = certify_gap_locked(L, La, n; nsep = 1, center = 1.0, radius = ρ,
            nblock = 256, band = 0.2, nθ = 512, kmax = 120, prec = Float64)
        R1 = fl > 0 ? 1 / fl : Inf
        δstar = A / R1
        dstar = isfinite(δstar) && δstar > 0 ? Int(ceil(log2(2.0 / δstar))) : 0
        @printf("%6.3f %12.4e %8s %10.1f %10.2f %14s\n",
            ρ, fl, info.certified ? "✓" : "✗", R1, R1 * δh, dstar > 0 ? @sprintf("%.4f(d%d)", δstar, dstar) : "—")
        flush(stdout)
        push!(rows, (ρ, R1, info.certified))
    end
    cert = filter(r -> r[3] && isfinite(r[2]), rows)
    if !isempty(cert)
        bρ, bR, _ = cert[argmin([r[2] for r in cert])]
        @printf("  ==> depth %d best: ρ₁=%.3f R₁=%.1f  R₁·δ_h=%.2f  d*≈%d\n",
            depth, bρ, bR, bR * δh, Int(ceil(log2(2.0 / (A / bR)))))
    end
    flush(stdout)
end
