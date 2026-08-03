# PROTOTYPE of the consolidated two-block (good + deep-far) certificate:
#   small eigenvalue block V_g (enclose) + singular block U (A*A Lanczos, matvec) → deep far block.
#   Q = orth(V_g ∪ U);  per circle: σ11 = σ_min(Q*AQ - zI),  σ22 = |z| - ‖G_ugly‖,  assemble.
# Validates: (a) matvec A*A Lanczos gives good singular vectors, (b) ‖ugly‖ goes deep,
# (c) well-separated leading eigenvalues isolate, (d) the separating gap certifies.
#   CACHED=1 DEPTH=6 MEIG=32 KSING=64 julia --project=examples examples/14s_consolidated.jl
import Pkg; Pkg.activate(@__DIR__)
include(joinpath(@__DIR__, "ulam_matvec.jl"))
using RigorousPseudospectra, Printf, LinearAlgebra
const RPS = RigorousPseudospectra
build = get(ENV,"CACHED","")=="1" ? build_matvec_cached : build_matvec
NOISE = uniform_noise_box(0.1,2)
d = IntervalBox(interval(-1.0,1.0), interval(-1.0,1.0))
L,La,n,_ = build(henon_rescaled, d, NOISE, 6; periodic=true, enclosure_method=first_order_enclosure)
MEIG = parse(Int, get(ENV,"MEIG","32")); KSING = parse(Int, get(ENV,"KSING","64"))

# top-k right singular vectors of A via Lanczos on M(x)=A*(A x) — matvec only, basis stored
function top_right_singular(Amul, Aadj, n, k; steps=2k+30)
    Q = zeros(ComplexF64, n, steps); α=zeros(steps); β=zeros(steps)
    q = randn(ComplexF64,n); q./=norm(q); Q[:,1]=q; qp=zeros(ComplexF64,n); βp=0.0; kact=steps
    for j in 1:steps
        w = ComplexF64.(Aadj(ComplexF64.(Amul(Q[:,j]))))
        a = real(dot(Q[:,j],w)); α[j]=a; w .-= a.*Q[:,j] .+ βp.*qp
        for _ in 1:2, i in 1:j; c=dot(Q[:,i],w); w .-= c.*Q[:,i]; end   # full reorth
        b=norm(w); β[j]=b
        (b<1e-12 || j==steps) && (kact=j; break)
        Q[:,j+1]=w./b; qp=Q[:,j]; βp=b
    end
    T=SymTridiagonal(α[1:kact], β[1:kact-1]); E=eigen(T)
    idx=sortperm(E.values;rev=true)[1:min(k,kact)]
    return Matrix(qr(Q[:,1:kact]*E.vectors[:,idx]).Q)[:,1:min(k,kact)]
end

println("materializing A (for prototype norms)…"); flush(stdout)
A = zeros(ComplexF64,n,n); e=zeros(ComplexF64,n); for j in 1:n; fill!(e,0); e[j]=1; A[:,j]=L(e); end
opn(M)=maximum(svdvals(M)); smin(M)=minimum(svdvals(M))

Hg, Vg, βN = RPS._arnoldi(L, n, MEIG)
λH = eigvals(ComplexF64.(Hg)); ordH = sortperm(abs.(λH); rev=true)
U = top_right_singular(L, La, n, KSING)
Up = U - Vg*(Vg'*U)                                     # project against the eigen block
keep = [j for j in 1:size(Up,2) if norm(Up[:,j])>1e-8]  # rank-revealing drop
Q = Matrix(qr(hcat(Vg, Up[:,keep])).Q)[:, 1:MEIG+length(keep)]
Pc = I - Q*Q'; G22n = opn(Pc*A*Pc); γ = max(opn(Pc*A*Q), opn(Pc*(A'*Q)))
QAQ = Q'*A*Q
@printf("MEIG=%d KSING=%d (kept %d) dim(Q)=%d  ‖G_ugly‖=%.4f  γ=%.4f  κ₂(Q*AQ eig)≈n/a\n",
        MEIG, KSING, length(keep), size(Q,2), G22n, γ); flush(stdout)

# --- isolation of the leading eigenvalues (verified against the matvec-built Q) ---
frac=0.4; nθ=96
println("isolation (σ11 = svdvals(Q*AQ - zI), σ22 = |z|-‖G_ugly‖):")
for ii in ordH[1:min(10, length(ordH))]
    c=λH[ii]; gap=minimum(abs(c-λH[j]) for j in eachindex(λH) if j!=ii); r=frac*gap
    σ11=minimum(smin(QAQ-(c+r*cis(2π*t/nθ))*I) for t in 0:nθ-1)
    minz=minimum(abs(c+r*cis(2π*t/nθ)) for t in 0:nθ-1); σ22=minz-G22n
    a=(σ11<=0||σ22<=0) ? 0.0 : σ11*σ22/(σ11+σ22+γ); fl=a-2r*sin(π/(2nθ))
    @printf("  |λ|=%.4f r=%.4f σ11=%.4f σ22=%.4f floor=%+.3e %s\n", abs(c),r,σ11,σ22,fl, fl>0 ? "✓" : "✗"); flush(stdout)
end
# --- separating gap circle at |z|=0.834 ---
r=0.834
σ11=minimum(smin(QAQ-(r*cis(2π*t/512))*I) for t in 0:511); σ22=r-G22n
a=(σ11<=0||σ22<=0) ? 0.0 : σ11*σ22/(σ11+σ22+γ); fl=a-2r*sin(π/1024)
@printf("GAP |z|=%.3f: σ11=%.4f σ22=%.4f floor=%+.3e %s\n", r,σ11,σ22,fl, fl>0 ? "✓" : "✗")
