import Pkg; Pkg.activate(@__DIR__)
include(joinpath(@__DIR__, "ulam_matvec.jl"))
using RigorousPseudospectra, Printf
build = get(ENV,"CACHED","")=="1" ? build_matvec_cached : build_matvec
NOISE = uniform_noise_box(0.1,2)
d = IntervalBox(interval(-1.0,1.0), interval(-1.0,1.0))
L,La,n,_ = build(henon_rescaled, d, NOISE, 6; periodic=true, enclosure_method=first_order_enclosure)
for (r,nθ) in ((0.75,2048),(0.72,3072))
    fl,i = certify_gap_locked(L,La,n; nsep=1, nblock=256, radius=r, band=0.25, nθ=nθ, kmax=120, prec=Float64)
    @printf("r=%.3f nθ=%d  σ≥%+.4e %s  rate≤%.3f\n", r, nθ, fl, i.certified ? "✓" : "✗", r)
    flush(stdout)
end
