# Is certify_gap's γ=5.33 a real coupling, or the cheap √(L1·Linf) opnorm overestimating the
# WIDE n×256 deflated matrix?  Compare the CHEAP verified bound (what _coupling_cert uses) to a
# TIGHT verified upper bound (svdbox top-σ upper endpoint) and to the true Float64 opnorm, for
# the adjoint coupling (I−VV*)A*V. Then the resulting assembly floor with the tight γ.
#   CACHED=1 DEPTH=6 julia --project=examples examples/14l_couplingnorm.jl
import Pkg; Pkg.activate(@__DIR__)
include(joinpath(@__DIR__, "ulam_matvec.jl"))
using RigorousPseudospectra, BallArithmetic, Printf, LinearAlgebra
const RPS = RigorousPseudospectra
build = get(ENV,"CACHED","")=="1" ? build_matvec_cached : build_matvec
NOISE = uniform_noise_box(0.1,2)
d = IntervalBox(interval(-1.0,1.0), interval(-1.0,1.0))
L,La,n,_ = build(henon_rescaled, d, NOISE, 6; periodic=true, enclosure_method=first_order_enclosure)
H, V, βN = RPS._arnoldi(L, n, 256)

# adjoint coupling matrix  C = (I−VV*)A*V   (n×256, the wide one)
AtV = RPS._applycols(La, V)
C = AtV .- V*(V'*AtV)
Cb = BallMatrix(C)
cheap = upper_bound_L2_opnorm(Cb)                       # what _coupling_cert uses (min Collatz, √L1Linf)
true2 = opnorm(C, 2)                                    # Float64 truth
tight = svd_bound_L2_opnorm(Cb)                         # verified SVD-based UPPER bound on ‖C‖₂
@printf("adjoint coupling (I−VV*)A*V, %d×%d:\n", size(C)...)
@printf("  cheap  upper_bound_L2_opnorm = %.4e   (what certify_gap uses → γ)\n", cheap)
@printf("  tight  svdbox σ_max upper    = %.4e\n", tight)
@printf("  true   Float64 opnorm        = %.4e\n", true2)
# assembly at the worst circle point with each γ (σ11≈0.0497, σ22≈0.406 from earlier diagnostics)
σ11, σ22 = 0.0497, 0.406
asm(γ) = σ11*σ22/(σ11+σ22+γ)
arc = 2*0.834*sin(π/(2*512))
@printf("\nassembly σ11=%.4f σ22=%.4f  arc-cover(nθ=512)=%.4e\n", σ11, σ22, arc)
for (nm,γ) in (("cheap γ",cheap), ("tight γ",tight))
    @printf("  %-8s γ=%.3f → pointwise %.4e → floor %.4e  %s\n",
            nm, γ, asm(γ), asm(γ)-arc, asm(γ)-arc>0 ? "✓" : "✗")
end
