# Automatic "certify everything outside a fixed circle R": fix R, deflate until ‖G₂₂‖<R/2,
# certify ALL eigenvalues with |λ|>R. No neig guessing.
#   CACHED=1 DEPTH=6 RCIRC=0.35 julia --project=examples examples/14x_outside.jl
import Pkg; Pkg.activate(@__DIR__)
include(joinpath(@__DIR__, "ulam_matvec.jl"))
using RigorousPseudospectra, Printf
build = get(ENV,"CACHED","")=="1" ? build_matvec_cached : build_matvec
NOISE = uniform_noise_box(0.1,2)
d = IntervalBox(interval(-1.0,1.0), interval(-1.0,1.0))
L,La,n,_ = build(henon_rescaled, d, NOISE, 6; periodic=true, enclosure_method=first_order_enclosure)
R = parse(Float64, get(ENV,"RCIRC","0.35"))
t0=time()
out = certify_leading_eigs_sing(L, La, n; Rcirc=R, nblock_eig=80, margin=0.8)
el=time()-t0
@printf("Rcirc=%.3f: %.1fs  nsing=%d dim(Q)=%d  ‖G₂₂‖=%.4f (<R? %s)  #outside=%d\n",
        R, el, out.nsing, out.dimQ, out.G22norm, out.far_deep ? "yes ✓" : "NO ✗", length(out.results))
for r in out.results
    @printf("  |λ|=%.4f λ=%+.4f%+.4fi  σ_min≥%.3e %s  r_heur=%.1e enclosure=%.1e\n",
        abs(r.λ), real(r.λ), imag(r.λ), r.cert.floor, r.cert.certified ? "✓" : "✗", r.cert.r_heur, r.cert.enclosure)
    flush(stdout)
end
