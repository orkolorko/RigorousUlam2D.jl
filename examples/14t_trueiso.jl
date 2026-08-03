# Is σ11=σ_min(Q*AQ-zI)=0.015 on the sub-leading circles caused by an eigenvalue of Q*AQ sitting
# near the circle (user's hypothesis: a "captured" eigenvalue), and is it a REAL eigenvalue of A
# or a compression ghost? For each sub-leading eigenvalue: list Q*AQ eigenvalues within the
# isolation circle, and the true σ_min(A-zI) at the worst of a few sample points.
#   CACHED=1 DEPTH=6 MEIG=32 KSING=64 julia --project=examples examples/14t_trueiso.jl
import Pkg; Pkg.activate(@__DIR__)
include(joinpath(@__DIR__, "ulam_matvec.jl"))
using RigorousPseudospectra, Printf, LinearAlgebra
const RPS = RigorousPseudospectra
build = get(ENV,"CACHED","")=="1" ? build_matvec_cached : build_matvec
NOISE = uniform_noise_box(0.1,2)
d = IntervalBox(interval(-1.0,1.0), interval(-1.0,1.0))
L,La,n,_ = build(henon_rescaled, d, NOISE, 6; periodic=true, enclosure_method=first_order_enclosure)
MEIG=parse(Int,get(ENV,"MEIG","32")); KSING=parse(Int,get(ENV,"KSING","64"))
println("materializing A…"); flush(stdout)
A=zeros(ComplexF64,n,n); e=zeros(ComplexF64,n); for j in 1:n; fill!(e,0); e[j]=1; A[:,j]=L(e); end
smin(M)=minimum(svdvals(M))
function tsv(Amul,Aadj,n,k;steps=2k+30)
    Q=zeros(ComplexF64,n,steps);α=zeros(steps);β=zeros(steps);q=randn(ComplexF64,n);q./=norm(q);Q[:,1]=q;qp=zeros(ComplexF64,n);βp=0.0;kact=steps
    for j in 1:steps
        w=ComplexF64.(Aadj(ComplexF64.(Amul(Q[:,j]))));a=real(dot(Q[:,j],w));α[j]=a;w.-=a.*Q[:,j].+βp.*qp
        for _ in 1:2,i in 1:j;c=dot(Q[:,i],w);w.-=c.*Q[:,i];end
        b=norm(w);β[j]=b;(b<1e-12||j==steps)&&(kact=j;break);Q[:,j+1]=w./b;qp=Q[:,j];βp=b
    end
    T=SymTridiagonal(α[1:kact],β[1:kact-1]);E=eigen(T);idx=sortperm(E.values;rev=true)[1:min(k,kact)]
    Matrix(qr(Q[:,1:kact]*E.vectors[:,idx]).Q)[:,1:min(k,kact)]
end
Hg,Vg,βN=RPS._arnoldi(L,n,MEIG);λH=eigvals(ComplexF64.(Hg));ord=sortperm(abs.(λH);rev=true)
U=tsv(L,La,n,KSING);Up=U-Vg*(Vg'*U);keep=[j for j in 1:size(Up,2) if norm(Up[:,j])>1e-8]
Q=Matrix(qr(hcat(Vg,Up[:,keep])).Q)[:,1:MEIG+length(keep)];QAQ=Q'*A*Q
eQ=eigvals(QAQ); eA=eigvals(A)          # spectrum of the 96-block vs the true operator
for ii in ord[2:6]
    c=λH[ii];gap=minimum(abs(c-λH[j]) for j in eachindex(λH) if j!=ii);r=0.4*gap
    inQ=[e for e in eQ if abs(e-c)<r && abs(e-c)>1e-6]        # OTHER Q*AQ eigenvalues inside Γ
    near=isempty(inQ) ? nothing : argmin([abs(e-c) for e in inQ])
    # for the nearest such block eigenvalue, distance to the TRUE spectrum of A
    dtrueA = near===nothing ? NaN : minimum(abs(inQ[near]-ea) for ea in eA)
    z=c+r  # one sample point on the circle
    @printf("|λ|=%.4f r=%.4f: #Q*AQ-eigs inside Γ (besides λ)=%d", abs(c), r, length(inQ))
    near!==nothing && @printf("  nearest=%.3f%+.3fi (dist %.4f), its dist to σ(A)=%.4f",
        real(inQ[near]),imag(inQ[near]),abs(inQ[near]-c),dtrueA)
    @printf("   true σ_min(A-zI)@z=λ+r: %.4e  vs σ_min(QAQ): %.4e\n", smin(A-z*I), smin(QAQ-z*I))
    flush(stdout)
end
