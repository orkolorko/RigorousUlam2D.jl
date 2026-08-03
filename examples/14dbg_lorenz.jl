# DIAGNOSTIC: is the Lorenz λ≈1 result right? Test (Nisoli): if λ=1 is simple and nothing else
# lies outside R, the resolvent must behave as |1/(z−1)|, i.e. σ_min(A−zI) ≈ |z−1| near it. So a
# circle of radius r should yield a floor ≈ r, and the lifted Ritz residual should be ~eps.
#   CACHED=1 DEPTH=6 julia --project=examples examples/14dbg_lorenz.jl
import Pkg; Pkg.activate(@__DIR__)
include(joinpath(@__DIR__, "ulam_matvec.jl"))
using RigorousPseudospectra, Printf, LinearAlgebra
const RP = RigorousPseudospectra
build = get(ENV,"CACHED","")=="1" ? build_matvec_cached : build_matvec
NOISE = uniform_noise_box(0.1,2)
dl = IntervalBox(interval(-0.5,0.5), interval(-0.5,0.5))
L,La,n,_ = build(contracting_lorenz, dl, NOISE, 6; periodic=false, enclosure_method=zeroth_order_enclosure)
@printf("n = %d\n", n); flush(stdout)

# 1. raw Arnoldi: leading Ritz values and the TRUE residual of the leading Ritz vector
H,V,βN = RP._arnoldi(x->ComplexF64.(L(x)), n, 120)
λH = eigvals(ComplexF64.(H)); o = sortperm(abs.(λH); rev=true)
@printf("Arnoldi(120) leading |λ|: %s\n", join((@sprintf("%.5f",abs(λH[o[k]])) for k in 1:8), " "))
F = eigen(ComplexF64.(H))
ordF = sortperm(abs.(F.values); rev=true)
for k in 1:3
    j = ordF[k]; y = F.vectors[:, j]; μ = F.values[j]
    q = V*y; q ./= norm(q)
    res = norm(ComplexF64.(L(q)) .- μ.*q)
    @printf("  Ritz %d: λ=%+.6f%+.6fi  |λ|=%.6f  TRUE residual ‖Aq−λq‖ = %.3e\n",
            k, real(μ), imag(μ), abs(μ), res)
end
flush(stdout)

# 2. mass conservation: for an Ulam (column-stochastic) operator, 1ᵀA = 1ᵀ ⇒ λ=1 exactly,
#    with LEFT eigenvector the all-ones. Check both the row sums and A on the uniform density.
ones_v = ones(ComplexF64, n) ./ sqrt(n)
@printf("‖A*1 − 1‖/‖1‖ (adjoint on ones; =0 iff column-stochastic) = %.3e\n",
        norm(ComplexF64.(La(ones_v)) .- ones_v)/norm(ones_v)); flush(stdout)

# 3. the pipeline, then the resolvent profile near λ
out = certify_leading_eigs_sing(L, La, n; Rcirc=0.6, nblock_eig=80, margin=0.8, ρfloor=0.001)
@printf("\npipeline: dim(Q)=%d nsing=%d ‖G22‖=%.4f γ=%.4f  #outside=%d\n",
        out.dimQ, out.nsing, out.G22norm, out.γ, length(out.results)); flush(stdout)
r0 = out.results[1]; λ1 = r0.λ
Winv = inv(out.mr.W)
i1 = argmin(abs.(out.mr.centres .- λ1))
@printf("target index i1=%d  centres[i1]=%+.6f%+.6fi  (λ1=%+.6f%+.6fi)\n",
        i1, real(out.mr.centres[i1]), imag(out.mr.centres[i1]), real(λ1), imag(λ1))
# (a) is W[:,i1] an eigenvector of the BLOCK G11?  (b) is its lift an eigenvector of A?
w = out.mr.W[:, i1]
@printf("  block residual  ‖G11·w − λ·w‖/‖w‖ = %.3e\n",
        norm(out.G11*w .- λ1.*w)/norm(w))
qi = out.Q * w; qi ./= norm(qi)
@printf("  lifted residual ‖A·q − λ·q‖       = %.3e   κ(λ) = %.3e\n",
        norm(ComplexF64.(L(qi)) .- λ1.*qi), norm(w)*norm(Winv[i1,:])); flush(stdout)
# compare against the honest Arnoldi eigenvector of the SAME block, to isolate an index mismatch
FG = eigen(out.G11); jj = argmin(abs.(FG.values .- λ1))
w2 = FG.vectors[:, jj]; q2 = out.Q*w2; q2 ./= norm(q2)
@printf("  via eigen(G11):  block res = %.3e   lifted res = %.3e\n",
        norm(out.G11*w2 .- FG.values[jj].*w2)/norm(w2),
        norm(ComplexF64.(L(q2)) .- FG.values[jj].*q2)); flush(stdout)
# how many block eigenvalues sit near λ1 (is it really simple/isolated?)
d = sort(abs.(out.mr.centres .- λ1))
@printf("distances from λ1 to other BLOCK eigenvalues: %s\n",
        join((@sprintf("%.4f",x) for x in d[2:min(7,end)]), " ")); flush(stdout)

# 4. THE TEST: σ_min(G11−zI) and the rigorous floor vs |z−λ1|
println("\n r        |z-λ1|    σmin(G11-z)  ratio    rigorous floor  ratio")
for r in (0.005, 0.01, 0.02, 0.05, 0.1, 0.2)
    z = λ1 + r
    s11 = minimum(svdvals(out.G11 - z*I))
    fl  = pseudospectrum_outside(out, z)
    @printf(" %-8.4f %-9.4f %-12.4e %-8.3f %-15.4e %.3f\n", r, r, s11, s11/r, fl, fl/r)
    flush(stdout)
end
