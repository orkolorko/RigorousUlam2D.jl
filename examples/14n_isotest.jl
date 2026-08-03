# Does the refactored certify_leading_eigs run FAST now (bounded re-trigger kmax) and still
# certify the leading Hénon eigenvalues? Print per-circle floor, ntrigger, and total time.
#   CACHED=1 DEPTH=6 julia --project=examples examples/14n_isotest.jl
import Pkg; Pkg.activate(@__DIR__)
include(joinpath(@__DIR__, "ulam_matvec.jl"))
using RigorousPseudospectra, Printf
build = get(ENV,"CACHED","")=="1" ? build_matvec_cached : build_matvec
NOISE = uniform_noise_box(0.1,2)
d = IntervalBox(interval(-1.0,1.0), interval(-1.0,1.0))
L,La,n,_ = build(henon_rescaled, d, NOISE, 6; periodic=true, enclosure_method=first_order_enclosure)
t0 = time()
out = certify_leading_eigs(L, La, n; nblock=min(256,n-2), neig=10, block_estimate=false)
el = time()-t0
@printf("certify_leading_eigs: %.1fs   κ₂=%.2f\n", el, out.mr.kappa)
for r in out.results
    @printf("  |λ|=%.4f  ρ=%.4f  σ_min≥%.3e  %s  (σ11=%.3e σ22=%.3e γ=%.3f ntrig=%d)\n",
        abs(r.λ), r.cert.radius, r.cert.floor, r.cert.certified ? "✓" : "✗",
        r.cert.σ11min, r.cert.σ22min, r.cert.γ, r.cert.ntrigger)
    flush(stdout)
end
