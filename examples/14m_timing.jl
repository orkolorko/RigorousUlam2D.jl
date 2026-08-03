# Where does the refactored certify_leading_eigs spend its time? (Hénon depth 6)
#   CACHED=1 DEPTH=6 julia --project=examples examples/14m_timing.jl
import Pkg; Pkg.activate(@__DIR__)
include(joinpath(@__DIR__, "ulam_matvec.jl"))
using RigorousPseudospectra, BallArithmetic, Printf, LinearAlgebra
const RPS = RigorousPseudospectra
build = get(ENV,"CACHED","")=="1" ? build_matvec_cached : build_matvec
NOISE = uniform_noise_box(0.1,2)
d = IntervalBox(interval(-1.0,1.0), interval(-1.0,1.0))
L,La,n,_ = build(henon_rescaled, d, NOISE, 6; periodic=true, enclosure_method=first_order_enclosure)
lap() = (global _t0; t=time()-_t0; _t0=time(); t)
@printf("n=%d\n", n); flush(stdout)
_t0 = time()
H,V,βN = RPS._arnoldi(L, n, 256);                 @printf("Arnoldi(256):            %.1fs\n", lap()); flush(stdout)
Hc = ComplexF64.(H)
mr = RPS._certify_block(Hc, 6);                    @printf("VBD(256):                %.1fs\n", lap()); flush(stdout)
vr = vbd_resolvent(mr.res);                        @printf("vbd_resolvent:           %.1fs\n", lap()); flush(stdout)
defl(M) = M .- V*(V'*M)
C = defl(RPS._applycols(La, V))
cb = svd_bound_L2_opnorm(BallMatrix(C));           @printf("svd_bound wide 4096x256: %.1fs  (=%.3f)\n", lap(), cb); flush(stdout)
g = sqrt(svd_bound_L2_opnorm(BallMatrix(C'*C)));   @printf("svd_bound Gram 256x256:  %.1fs  (=%.3f)\n", lap(), g); flush(stdout)
z = 0.9 + 0.1im
s = RPS._sigma_min_cert(Hc - z*I);                 t=lap(); @printf("svdbox(256) per-z:       %.2fs  (x240 = %.0fs)\n", t, 240t); flush(stdout)
sb = sigma_min_lower(vr, z; near=true);            t=lap(); @printf("block_estimate per-z:    %.3fs  (x240 = %.0fs)\n", t, 240t); flush(stdout)
