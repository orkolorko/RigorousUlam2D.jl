# PROTOTYPE (concept validation, Float64, true singular vectors): augment the captured subspace
# with A's top singular vectors so the ugly block is DEEP, and check that leading-eigenvalue
# isolation then certifies via the shared-ugly assembly.
#   captured  Q = orth([V | Vr_k])   (eigenvalue Krylov V ∪ top-k right singular vectors)
#   ugly      G22 = (I-QQ*)A(I-QQ*)  ⇒ ‖G22‖ ≈ σ_{dim(Q)+1}  (deep)
#   per circle: σ11 = min σ_min(Q*AQ - zI), σ22 = |z|-‖G22‖, floor = σ11σ22/(σ11+σ22+γ) - arc
#   CACHED=1 DEPTH=6 julia --project=examples examples/14q_augment.jl
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
FA = svd(A)
H, V, βN = RPS._arnoldi(L, n, 256)
λH = eigvals(ComplexF64.(H)); ord = sortperm(abs.(λH); rev=true)
opn(M) = maximum(svdvals(M)); smin(M) = minimum(svdvals(M))

# choose k singular vectors so σ_{256+k margin} is below the deepest leading |λ|
ksing = get(ENV,"KSING","")=="" ? 200 : parse(Int, ENV["KSING"])
Vr = FA.V[:, 1:ksing]
Q = Matrix(qr(hcat(V, Vr)).Q)[:, 1:min(256+ksing, n)]     # captured = span(V ∪ top-k right sing vecs)
mcap = size(Q,2)
Pc = I - Q*Q'
G22n = opn(Pc*A*Pc)
γ = max(opn(Pc*A*Q), opn(Pc*(A'*Q)))
@printf("dim(V)=256  ksing=%d  dim(Q_cap)=%d   ‖G22‖=%.4f  (σ_%d=%.4f)  γ=%.4f\n",
        ksing, mcap, G22n, mcap+1, FA.S[mcap+1], γ); flush(stdout)
QAQ = Q'*A*Q
frac = 0.4; nθ = 48
for ii in ord[1:10]
    c = λH[ii]
    gap = minimum(abs(c-λH[j]) for j in eachindex(λH) if j!=ii)
    r = frac*gap
    σ11 = minimum(smin(QAQ - (c + r*cis(2π*t/nθ))*I) for t in 0:nθ-1)
    fl = Inf
    for t in 0:nθ-1
        z = c + r*cis(2π*t/nθ); σ22 = abs(z) - G22n
        a = (σ11<=0||σ22<=0) ? 0.0 : σ11*σ22/(σ11+σ22+γ); fl = min(fl, a)
    end
    fl -= 2r*sin(π/(2nθ))
    @printf("  |λ|=%.4f r=%.4f σ11=%.4f σ22≈%.4f  floor=%+.3e %s\n",
        abs(c), r, σ11, minimum(abs(c+r*cis(2π*t/nθ)) for t in 0:nθ-1)-G22n, fl, fl>0 ? "✓" : "✗"); flush(stdout)
end
