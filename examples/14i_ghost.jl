# Is σ_min(H−zI)≈0.05 at z=−0.834 a TRUE feature of A, or an Arnoldi-compression ghost?
# Materialize A (matvec on the 4096 unit columns), compute the TRUE σ_min(A−zI) at the worst
# circle angles, and compare to the 256-block compression σ_min and the near-circle band σ_min.
#   CACHED=1 DEPTH=6 julia --project=examples examples/14i_ghost.jl
import Pkg; Pkg.activate(@__DIR__)
include(joinpath(@__DIR__, "ulam_matvec.jl"))
using RigorousPseudospectra, Printf, LinearAlgebra
const RPS = RigorousPseudospectra
build = get(ENV,"CACHED","")=="1" ? build_matvec_cached : build_matvec
NOISE = uniform_noise_box(0.1,2)
d = IntervalBox(interval(-1.0,1.0), interval(-1.0,1.0))
L,La,n,_ = build(henon_rescaled, d, NOISE, 6; periodic=true, enclosure_method=first_order_enclosure)

# materialize A = L applied to each unit column (A is n×n, n=4096)
println("materializing A ($n×$n) via matvec…"); flush(stdout)
A = zeros(ComplexF64, n, n)
e = zeros(ComplexF64, n)
for j in 1:n
    fill!(e, 0); e[j] = 1
    A[:,j] = L(e)
end
H, V, βN = RPS._arnoldi(L, n, 256); Hc = ComplexF64.(H)
mr = RPS._certify_block(Hc, 6); λ = mr.centres
r = 0.834
kk = [i for i in eachindex(λ) if abs(abs(λ[i]) - r) <= 0.25*r]
Hk = Matrix(RPS._captured_block(L, RPS._frame(V*mr.W[:,kk])))
@printf("β_N (Arnoldi residual) = %.3e\n", βN)
println("        z            true σ_min(A−zI)   σ_min(H−zI) full   σ_min(band)")
for t in 0:7
    z = r*cis(2π*t/8)
    trueA = minimum(svdvals(A - z*I))
    full  = minimum(svdvals(Hc - z*I))
    band  = minimum(svdvals(Hk - z*I))
    @printf("  %+.3f%+.3fi   %+.4e         %+.4e         %+.4e\n",
            real(z), imag(z), trueA, full, band); flush(stdout)
end
# also: nearest Ritz value of the full block to z=−0.834, and is it near any TRUE eigenvalue?
z0 = -r
eH = eigvals(Hc)
i = argmin(abs.(eH .- z0))
eA = eigvals(A)
j = argmin(abs.(eA .- z0))
@printf("\nAt z=%.3f: nearest H-Ritz = %+.4f%+.4fi (dist %.4f);  nearest true A-eig = %+.4f%+.4fi (dist %.4f)\n",
        z0, real(eH[i]), imag(eH[i]), abs(eH[i]-z0), real(eA[j]), imag(eA[j]), abs(eA[j]-z0))
