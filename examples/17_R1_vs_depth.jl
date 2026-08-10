# R_1 (resolvent on the isolation circle of λ₁=1) vs mesh depth, for Hénon.
# Σ_C = ‖R(1,Q_h)|_{U_0}‖ ≤ R_1(1+ρ₁R_1), read off the leading eigenvalue's certificate.
# Gate for the a-posteriori resolvent bound: η = δ(var(ρ)+1)Σ_C < 1.
import Pkg; Pkg.activate(@__DIR__)
include(joinpath(@__DIR__, "ulam_matvec.jl"))
using RigorousPseudospectra, Printf
using RigorousUlam2D: first_order_enclosure

NOISE = uniform_noise_box(0.1, 2)         # box [-0.1,0.1]², var(ρ)=20
dh = IntervalBox(interval(-1.0, 1.0), interval(-1.0, 1.0))
DEPTHS = get(ENV, "DEPTHS", "") == "" ? [7, 8] : parse.(Int, split(ENV["DEPTHS"], ","))
NEIG = get(ENV, "NEIG", "") == "" ? 10 : parse(Int, ENV["NEIG"])
NBLOCK = get(ENV, "NBLOCK", "") == "" ? 28 : parse(Int, ENV["NBLOCK"])
varρ = 20.0

for depth in DEPTHS
    δ = 2.0 / 2^depth
    L, La, n, _ = build_matvec_cached(henon_rescaled, dh, NOISE, depth;
        periodic = true, enclosure_method = first_order_enclosure)
    println("="^64)
    @printf("Hénon depth=%d  n=%d  δ=%.5f\n", depth, n, δ); flush(stdout)
    out = certify_leading_eigs(L, La, n; nblock = NBLOCK, neig = NEIG, refine = 6, prec = Float64)
    @printf("  κ₂=%.2f  β_N=%.2e\n", out.mr.kappa, out.βN)
    lead = nothing
    for r in out.results
        R = r.cert.floor > 0 ? 1 / r.cert.floor : Inf
        @printf("  λ=%+.5f%+.5fi |λ|=%.5f  ρ=%.4f  σ_min≥%.3e %s  R=%.1f\n",
            real(r.λ), imag(r.λ), abs(r.λ), r.cert.radius, r.cert.floor,
            r.cert.certified ? "✓" : "✗", R)
        if lead === nothing && abs(r.λ - 1) < 0.05
            lead = (ρ1 = r.cert.radius, R1 = R, cert = r.cert.certified)
        end
    end
    if lead !== nothing && isfinite(lead.R1)
        ΣC_tight = lead.R1
        ΣC_loose = lead.R1 * (1 + lead.ρ1 * lead.R1)
        η_tight = δ * (varρ + 1) * ΣC_tight
        η_loose = δ * (varρ + 1) * ΣC_loose
        @printf("  ==> λ₁: ρ₁=%.4f R₁=%.1f %s | Σ_C∈[%.0f,%.0f] | η=δ·21·Σ_C ∈ [%.1f,%.1f] (gate<1)\n",
            lead.ρ1, lead.R1, lead.cert ? "certified✓" : "UNCERT✗",
            ΣC_tight, ΣC_loose, η_tight, η_loose)
    end
    flush(stdout)
end
