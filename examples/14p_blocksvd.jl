# Does deflating the SINGULAR vectors of the block G11 (lifted through the frame) shrink the
# ugly ‖P A P‖?  Compare, on the materialized Hénon A:
#   (0) σ_257(A)                                        — the target (true singular deflation)
#   (1) Krylov:        ‖(I-VV*) A (I-VV*)‖              — current ugly
#   (2) block-SVD:     Q=orth(V·V_H[:,1:256]);  ‖(I-QQ*) A (I-QQ*)‖   ← the proposed idea
#   (3) block-SVD k=128 (partial)
#   (4) TRUE right sing vecs of A (1-sided):  ‖(I-Vr Vr*) A (I-Vr Vr*)‖
#   (5) TRUE two-sided (left on left, right on right):  ‖(I-Ul Ul*) A (I-Vr Vr*)‖  → σ_257
#   CACHED=1 DEPTH=6 julia --project=examples examples/14p_blocksvd.jl
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
FA = svd(A)                                    # A = U diag(S) V'   (right sing vecs = FA.V)
opn(M) = maximum(svdvals(M))
defl(P) = opn(P*A*P)
H, V, βN = RPS._arnoldi(L, n, 256)
FH = svd(ComplexF64.(H))                         # block SVD; right sing vecs of H = FH.V
Q = Matrix(qr(V*FH.V).Q)[:,1:256]                # lifted block singular vectors (top 256)
I_n = Matrix{ComplexF64}(I,n,n)
P_V = I_n - V*V'
P_Q = I_n - Q*Q'
@printf("σ_1(A)=%.4f  σ_1(H)=%.4f  σ_257(A)=%.4f\n", FA.S[1], FH.S[1], FA.S[257]); flush(stdout)
@printf("(1) Krylov      (I-VV*)A(I-VV*)          = %.4f\n", defl(P_V)); flush(stdout)
@printf("(2) block-SVD   Q=lift(V·V_H[:,1:256])   = %.4f   (span(Q)=span(V) ⇒ identical?)\n", defl(P_Q)); flush(stdout)
Q2 = Matrix(qr(V*FH.V[:,1:128]).Q)[:,1:128]; P_Q2 = I_n - Q2*Q2'
@printf("(3) block-SVD   k=128 (partial)          = %.4f\n", defl(P_Q2)); flush(stdout)
Vr = FA.V[:,1:256]; Ul = FA.U[:,1:256]
@printf("(4) TRUE right  (I-Vr Vr*)A(I-Vr Vr*)    = %.4f\n", opn((I_n-Vr*Vr')*A*(I_n-Vr*Vr'))); flush(stdout)
@printf("(5) TRUE 2-side (I-Ul Ul*)A(I-Vr Vr*)    = %.4f   (= σ_257)\n", opn((I_n-Ul*Ul')*A*(I_n-Vr*Vr'))); flush(stdout)
