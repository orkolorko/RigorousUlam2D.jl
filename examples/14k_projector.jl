# What projector should the far-block deflation use?  certify_gap deflates the RAW Krylov
# basis V (A-nearly-invariant, but A*-LEAKY). The Rump–Miyajima frame V·mr.W gives the certified
# leading INVARIANT subspace (near-invariant for A AND A*). Measure forward/adjoint coupling
#   γ_A  = ‖(I−PP*)A  W‖ ,   γ_A* = ‖(I−PP*)A* W‖
# for (a) raw V (rank 256) and (b) the leading-k certified eigenvector frame V·mr.W[:,lead(k)].
#   CACHED=1 DEPTH=6 julia --project=examples examples/14k_projector.jl
import Pkg; Pkg.activate(@__DIR__)
include(joinpath(@__DIR__, "ulam_matvec.jl"))
using RigorousPseudospectra, Printf, LinearAlgebra
const RPS = RigorousPseudospectra
build = get(ENV,"CACHED","")=="1" ? build_matvec_cached : build_matvec
NOISE = uniform_noise_box(0.1,2)
d = IntervalBox(interval(-1.0,1.0), interval(-1.0,1.0))
L,La,n,_ = build(henon_rescaled, d, NOISE, 6; periodic=true, enclosure_method=first_order_enclosure)

H, V, βN = RPS._arnoldi(L, n, 256); Hc = ComplexF64.(H)
mr = RPS._certify_block(Hc, 6); λ = mr.centres
ord = sortperm(abs.(λ); rev=true)

fwd(W) = opnorm(RPS._applycols(L,  W) .- W*(W'*RPS._applycols(L,  W)), 2)
adj(W) = opnorm(RPS._applycols(La, W) .- W*(W'*RPS._applycols(La, W)), 2)

@printf("β_N = %.3e   ‖H‖ = %.3f\n", βN, opnorm(Hc,2))
@printf("%-28s  rank   γ_A(fwd)     γ_A*(adj)   |λ_k| cut\n", "frame")
@printf("%-28s  %4d   %.4e   %.4e\n", "raw Krylov V", size(V,2), fwd(V), adj(V))
for k in (4, 10, 20, 40, 80, 160)
    S = ord[1:k]
    W = RPS._frame(V * mr.W[:, S])       # leading-k certified invariant subspace
    @printf("%-28s  %4d   %.4e   %.4e   %.4f\n",
            "V·mr.W[:,lead($k)]", k, fwd(W), adj(W), abs(λ[ord[k]]))
    flush(stdout)
end
