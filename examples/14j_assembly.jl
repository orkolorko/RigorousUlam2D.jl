# Why does certify_gap go NEGATIVE while certify_gap_locked stays positive on |z|=0.834?
# Both are sound lower bounds on the true σ_min(A−zI)≈0.034; print their assembly inputs
# (σ11min, σ22min, coupling γ, floor) side by side. Same operator, same circle, same nθ.
#   CACHED=1 DEPTH=6 julia --project=examples examples/14j_assembly.jl
import Pkg; Pkg.activate(@__DIR__)
include(joinpath(@__DIR__, "ulam_matvec.jl"))
using RigorousPseudospectra, Printf
build = get(ENV,"CACHED","")=="1" ? build_matvec_cached : build_matvec
NOISE = uniform_noise_box(0.1,2)
d = IntervalBox(interval(-1.0,1.0), interval(-1.0,1.0))
L,La,n,_ = build(henon_rescaled, d, NOISE, 6; periodic=true, enclosure_method=first_order_enclosure)

for nθ in (64, 256)
    fg, ig = certify_gap(L, La, n; nsep=1, nblock=256, radius=0.834, nθ=nθ, prec=Float64)
    fl, il = certify_gap_locked(L, La, n; nsep=1, nblock=256, radius=0.834, band=0.25, nθ=nθ, kmax=120, prec=Float64)
    println("nθ=$nθ")
    @printf("  certify_gap        : floor=%+.4e %s   σ22min=%.4e  γ=%.4e  ‖G22‖=%.4e  R2=%.2e\n",
            fg, ig.certified ? "✓" : "✗", ig.σ22min, ig.γ, ig.G22norm, ig.R2)
    @printf("  certify_gap_locked : floor=%+.4e %s   σ11min=%.4e  σ22min=%.4e  γ=%.4e  m=%d\n",
            fl, il.certified ? "✓" : "✗", il.σ11min, il.σ22min, il.γ, il.m)
    flush(stdout)
end
