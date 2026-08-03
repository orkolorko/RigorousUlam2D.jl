# Is the Lorenz λ≈0.9988 in a SINGLETON VBD cluster? If so its W column must be an eigenvector of
# G11 (residual ~eps); the measured 0.42 says otherwise. Decide between (a) merged into a bigger
# cluster (QR in step 5 then mixes columns) and (b) Newton failed to decouple.
import Pkg; Pkg.activate(@__DIR__)
include(joinpath(@__DIR__, "ulam_matvec.jl"))
using RigorousPseudospectra, BallArithmetic, Printf, LinearAlgebra
const RP = RigorousPseudospectra
build = get(ENV,"CACHED","")=="1" ? build_matvec_cached : build_matvec
NOISE = uniform_noise_box(0.1,2)
dl = IntervalBox(interval(-0.5,0.5), interval(-0.5,0.5))
L,La,n,_ = build(contracting_lorenz, dl, NOISE, 6; periodic=false, enclosure_method=zeroth_order_enclosure)

# rebuild the captured block exactly as the pipeline does
Hg, Vg, βN = RP._arnoldi(x->ComplexF64.(L(x)), n, 80)
λg = eigvals(ComplexF64.(Hg)); ordg = sortperm(abs.(λg); rev=true)
U  = RP._singular_block(x->ComplexF64.(L(x)), x->ComplexF64.(La(x)), n, 0.9*0.6; kmax=256)
Up = U .- Vg*(Vg'*U); keep = [j for j in axes(Up,2) if norm(Up[:,j])>1e-8]
Qf = Matrix(qr(hcat(Vg, Up[:,keep])).Q)[:, 1:(size(Vg,2)+length(keep))]
G11 = Qf' * RP._applycols(x->ComplexF64.(L(x)), Qf)
Hc = ComplexF64.(G11)
@printf("dim(Q)=%d  block size %dx%d\n", size(Qf,2), size(Hc)...); flush(stdout)

res = BallArithmetic.schur_newton_vbd(BallMatrix(Matrix(Hc)); maxsteps=6, kappa_mode=:svdbox)
W = mid(res.basis); ctr = res.eigenvalues; cls = [collect(c) for c in res.clusters]
@printf("clusters: %d total; sizes: %s\n", length(cls),
        join(sort(unique(length.(cls))), ","))
@printf("‖remainder‖ = %.3e   ‖R2‖ = %.3e   beta = %.3e   kappa = %.3e\n",
        res.remainder_norm, res.nrmR2, res.beta, res.kappa); flush(stdout)

i1 = argmax(abs.(ctr))
c1 = cls[findfirst(c -> i1 in c, cls)]
@printf("\nleading centre = %+.6f%+.6fi at index %d\n", real(ctr[i1]), imag(ctr[i1]), i1)
@printf("its cluster has SIZE %d : %s\n", length(c1), string(first(c1, 8)))
w = W[:,i1]
@printf("  ‖G11·w − ctr·w‖/‖w‖ = %.3e   <-- eps ⇒ genuine eigenvector\n",
        norm(Hc*w .- ctr[i1].*w)/norm(w)); flush(stdout)

# how many indices overall fail to be eigenvectors, and do the failures coincide with big clusters?
bad = Int[]
for i in 1:size(Hc,1)
    r = norm(Hc*W[:,i] .- ctr[i].*W[:,i])/norm(W[:,i])
    r > 1e-8 && push!(bad, i)
end
@printf("\nindices whose W column is NOT an eigenvector (res>1e-8): %d of %d\n", length(bad), size(Hc,1))
sing = [i for i in bad if length(cls[findfirst(c->i in c, cls)]) == 1]
@printf("  ... of which sit in a SINGLETON cluster: %d  %s\n", length(sing), string(first(sing,10)))
if !isempty(sing)
    println("  ⇒ singleton columns are NOT eigenvectors: the Newton did not decouple them.")
else
    println("  ⇒ all failures are in multi-element clusters (QR mixing) — singletons are fine.")
end
# the true eigenvalues of G11 near the leading one, to see if a genuine near-degeneracy forced a merge
ev = eigvals(Hc); d = sort(abs.(ev .- ctr[i1]))
@printf("\ndistance from leading centre to G11 eigenvalues: %s\n",
        join((@sprintf("%.5f",x) for x in d[1:6]), " "))
