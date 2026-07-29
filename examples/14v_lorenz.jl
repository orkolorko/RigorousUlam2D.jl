# Test the consolidated matvec pipeline on the contracting-Lorenz Ulam operator (first 5 eigs).
#   CACHED=1 DEPTH=6 julia --project=examples examples/14v_lorenz.jl
import Pkg; Pkg.activate(@__DIR__)
include(joinpath(@__DIR__, "ulam_matvec.jl"))
using RigorousPseudospectra, Printf
build = get(ENV,"CACHED","")=="1" ? build_matvec_cached : build_matvec
NOISE = uniform_noise_box(0.1,2)
dl = IntervalBox(interval(-0.5,0.5), interval(-0.5,0.5))
L,La,n,_ = build(contracting_lorenz, dl, NOISE, 6; periodic=false, enclosure_method=zeroth_order_enclosure)
t0=time()
out = certify_leading_eigs_sing(L, La, n; neig=5, nblock_eig=40, margin=0.8, ρfloor=0.001)
el=time()-t0
@printf("Lorenz certify_leading_eigs_sing: %.1fs  nsing=%d dim(Q)=%d  ‖G_ugly‖=%.4f  γ=%.4f  |λ_g|=%.4f  κ₂=%.1f\n",
        el, out.nsing, out.dimQ, out.G22norm, out.γ, out.λ_min_good, out.mr.kappa)
for r in out.results
    @printf("  |λ|=%.4f λ=%+.4f%+.4fi  r=%.4f  σ_min≥%.3e %s  (σ11=%.4f σ22=%.4f)\n",
        abs(r.λ), real(r.λ), imag(r.λ), r.cert.radius, r.cert.floor,
        r.cert.certified ? "✓" : "✗", r.cert.σ11min, r.cert.σ22min)
    flush(stdout)
end
