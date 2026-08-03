# How SMALL is the "bad" singular block? Singular values of the far block G22 = P22 A P22
# (Krylov-256 complement). Deflating its top-p singular vectors drops ‖ugly‖ toward σ_{p+1}(G22);
# if they decay fast, only a small p is needed to get below the deepest good |λ|≈0.38.
#   CACHED=1 DEPTH=6 julia --project=examples examples/14r_farsvd.jl
import Pkg; Pkg.activate(@__DIR__)
include(joinpath(@__DIR__, "ulam_matvec.jl"))
using RigorousPseudospectra, Printf, LinearAlgebra
const RPS = RigorousPseudospectra
build = get(ENV,"CACHED","")=="1" ? build_matvec_cached : build_matvec
NOISE = uniform_noise_box(0.1,2)
d = IntervalBox(interval(-1.0,1.0), interval(-1.0,1.0))
L,La,n,_ = build(henon_rescaled, d, NOISE, 6; periodic=true, enclosure_method=first_order_enclosure)
println("materializing A…"); flush(stdout)
A = zeros(ComplexF64,n,n); e=zeros(ComplexF64,n); for j in 1:n; fill!(e,0); e[j]=1; A[:,j]=L(e); end
for meig in (32, 256)
    H, V, βN = RPS._arnoldi(L, n, meig)
    P = I - V*V'
    G22 = P*A*P
    s = svdvals(G22)
    @printf("eig-block=%d:  ‖G22‖=σ1=%.4f\n", meig, s[1])
    for p in (1,2,4,8,16,32,64,96,128)
        @printf("   after deflating top-%-3d far-sing-vecs: ‖ugly‖=σ_%-3d(G22)=%.4f  %s\n",
            p, p+1, s[p+1], s[p+1]<0.35 ? "< 0.35 (deep enough for isolation)" : "")
        flush(stdout)
    end
end
