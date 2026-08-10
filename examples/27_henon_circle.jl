# Circle-around-1 sweep for the contracting Henon map (a-posteriori resolvent).
# Henon: domain [-0.5,0.5]² (width 1 ⇒ δ=1/2^depth), absorbing BC, zeroth-order enclosure,
# |M|=1. Leading eigenvalue ≈0.998; sweep ρ₁ of Γ₁={|z-1|=ρ₁} enclosing only λ₁
# (gap 1-|λ₂| ≈ 0.76, so ρ₁ up to ~0.6). Report R₁=1/σ_min and the gate R₁·δ_h, δ_h=δ/a.
import Pkg; Pkg.activate(@__DIR__)
include(joinpath(@__DIR__, "ulam_matvec.jl"))
using RigorousPseudospectra, Printf
using RigorousUlam2D: first_order_enclosure

A = get(ENV, "A", "") == "" ? 0.1 : parse(Float64, ENV["A"])
NOISE = uniform_noise_box(A, 2)
dl = IntervalBox(interval(-1.0, 1.0), interval(-1.0, 1.0))
DEPTH = get(ENV, "DEPTH", "") == "" ? 6 : parse(Int, ENV["DEPTH"])
PREC = get(ENV, "PREC", "") == "big" ? BigFloat : Float64
get(ENV, "BITS", "") != "" && setprecision(BigFloat, parse(Int, ENV["BITS"]))
δ = 2.0 / 2^DEPTH; δh = δ / A          # Henon domain width = 1

L, La, n, _ = build_matvec_cached(henon_rescaled, dl, NOISE, DEPTH;
    periodic = true, enclosure_method = first_order_enclosure)
@printf("Henon depth=%d n=%d  a=%.2f  δ=%.5f  δ_h=δ/a=%.4f  prec=%s\n",
    DEPTH, n, A, δ, δh, PREC)
@printf("%6s %12s %8s %10s %10s %14s\n", "ρ₁", "σ_min", "cert", "R₁", "R₁·δ_h", "δ*=a/R₁(→d)")
rows = Tuple{Float64,Float64,Bool}[]
for ρ in (0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60)
    fl, info = certify_gap_locked(L, La, n; nsep = 1, center = 1.0, radius = ρ,
        nblock = 256, band = 0.2, nθ = 512, kmax = 120, prec = PREC)
    R1 = fl > 0 ? 1 / fl : Inf
    δstar = A / R1
    dstar = isfinite(δstar) && δstar > 0 ? Int(ceil(log2(1.0 / δstar))) : 0
    @printf("%6.3f %12.4e %8s %10.1f %10.2f %14s\n",
        ρ, Float64(fl), info.certified ? "✓" : "✗", Float64(R1), Float64(R1 * δh),
        dstar > 0 ? @sprintf("%.4f(d%d)", δstar, dstar) : "—")
    flush(stdout)
    push!(rows, (ρ, Float64(R1), info.certified))
end
cert = filter(r -> r[3] && isfinite(r[2]), rows)
if !isempty(cert)
    bρ, bR, _ = cert[argmin([r[2] for r in cert])]
    @printf("  ==> best: ρ₁=%.3f R₁=%.1f  R₁·δ_h=%.2f  d*≈%d\n",
        bρ, bR, bR * δh, Int(ceil(log2(1.0 / (A / bR)))))
end
