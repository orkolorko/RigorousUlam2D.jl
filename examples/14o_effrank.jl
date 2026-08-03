# Is ‖G₂₂‖=0.428 for the 256-complement genuine (effective rank > 256) or fishy (Krylov
# deflation not reducing the operator norm)? Materialize A, look at its singular-value decay,
# and compare ‖G₂₂‖ = ‖P₂₂ A P₂₂‖ (Krylov-256 complement) to σ_257(A).
#   CACHED=1 DEPTH=6 julia --project=examples examples/14o_effrank.jl
import Pkg; Pkg.activate(@__DIR__)
include(joinpath(@__DIR__, "ulam_matvec.jl"))
using RigorousPseudospectra, Printf, LinearAlgebra
const RPS = RigorousPseudospectra
build = get(ENV,"CACHED","")=="1" ? build_matvec_cached : build_matvec
NOISE = uniform_noise_box(0.1,2)
d = IntervalBox(interval(-1.0,1.0), interval(-1.0,1.0))
L,La,n,_ = build(henon_rescaled, d, NOISE, 6; periodic=true, enclosure_method=first_order_enclosure)
println("materializing A ($n×$n)…"); flush(stdout)
A = zeros(ComplexF64, n, n); e = zeros(ComplexF64, n)
for j in 1:n; fill!(e,0); e[j]=1; A[:,j]=L(e); end
sv = svdvals(A)
@printf("‖A‖=σ1=%.4f\n", sv[1])
for k in (64,128,192,256,320,384,512,768,1024,2048)
    @printf("  σ_%-4d = %.4e\n", k, sv[k])
end
# effective (numerical) rank at a few thresholds
for tol in (1e-2, 1e-3, 1e-6)
    @printf("  numerical rank (σ>%.0e·σ1): %d\n", tol, count(>(tol*sv[1]), sv))
end
# ‖G₂₂‖ for the Arnoldi-256 Krylov complement, vs σ_257(A)
H, V, βN = RPS._arnoldi(L, n, 256)
P22(x) = x .- V*(V'*x)
G22 = P22(A*P22(Matrix{ComplexF64}(I,n,n)))    # P₂₂ A P₂₂ as a dense n×n (columns)
@printf("‖G₂₂‖ (P₂₂AP₂₂, true σmax) = %.4f    σ_257(A) = %.4f    ‖A‖=%.4f\n",
        maximum(svdvals(G22)), sv[257], sv[1])
