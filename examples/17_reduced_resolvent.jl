# CERTIFIED REDUCED RESOLVENT AT z = 1 — the constant in the Ulam error bound.
#
# The a-posteriori estimate is ‖f − f_h‖ ≤ ‖(I − L)⁻¹|_V‖ · ‖r_h‖ with ‖r_h‖ ≤ (δ/2)var(f_h),
# where V is the complement of the Perron eigenspace. At z = 1 the operator I − L is SINGULAR
# (that is the Perron eigenvalue), so the meaningful object is the RESTRICTED resolvent, and the
# Miyajima–Rump block diagonalization supplies it directly: after the VBD,
#     Ã = Y G₁₁ W ≈ blockdiag(P₁,…,P_q)   with   ‖M − Λ‖ ≤ β_Λ,
# so the resolvent on the invariant subspace spanned by the NON-Perron blocks is governed by
#     σ₁₁^red(1) = [ min_{j ≠ Perron} σ_min(P_j − I) − β_Λ ] / κ₂(W),
# each σ_min(P_j − I) from a verified SVD. The far block contributes σ₂₂ ≥ |1| − ‖G₂₂‖ > 0
# (guaranteed by the certified gap ‖G₂₂‖ < R < 1), and the two are combined by the usual
# Schur-complement assembly with the coupling γ. Hence
#     ‖(I − L)⁻¹|_V‖ ≤ 1 / assemble(σ₁₁^red, σ₂₂, γ).
#
# Everything here is certified: the σ_min are svdbox, β_Λ and κ₂ come from the VBD certificate,
# ‖G₂₂‖ from the full-basis Lanczos, and γ from the verified Gram SVD.
#
#   CACHED=1 DEPTH=6 MAP=lorenz NOISES=0.3,0.2,0.1,0.05 julia --project=examples examples/17_reduced_resolvent.jl
import Pkg; Pkg.activate(@__DIR__)
include(joinpath(@__DIR__, "ulam_matvec.jl"))
using RigorousPseudospectra, Printf, LinearAlgebra
import BallArithmetic   # QUALIFIED on purpose: `using BallArithmetic` here makes `inf` ambiguous
                        # with IntervalArithmetic (both export it), and Julia then leaves it
                        # UNDEFINED — which surfaced as a TaskFailedException inside the build.
const RP = RigorousPseudospectra

build  = get(ENV, "CACHED", "") == "1" ? build_matvec_cached : build_matvec
depth  = parse(Int, get(ENV, "DEPTH", "6"))
R      = parse(Float64, get(ENV, "RCIRC", "0.6"))
mapnm  = get(ENV, "MAP", "lorenz")
noises = [parse(Float64, s) for s in split(get(ENV, "NOISES", "0.3,0.2,0.1,0.05"), ",")]

if mapnm == "henon"
    dom = IntervalBox(interval(-1.0, 1.0), interval(-1.0, 1.0))
    F, per, encl, ρfl = henon_rescaled, true, first_order_enclosure, 0.004
else
    dom = IntervalBox(interval(-0.5, 0.5), interval(-0.5, 0.5))
    F, per, encl, ρfl = contracting_lorenz, true, zeroth_order_enclosure, 0.001
end

@printf("CERTIFIED RESOLVENT ON A CIRCLE AROUND 1 — %s, depth %d (n=%d), R=%.2f\n",
        mapnm, depth, 4^depth, R)
println("Circle |z−1| = r encloses ONLY the Perron eigenvalue (all other |λ| ≤ R, so r < 1−R).")
println("σ₁₁ from the adaptive pearl cover = verified svdbox on G₁₁ — κ-FREE (exact-SVD escape).")
println("="^104)
@printf("%-7s %-6s %-9s %-11s %-11s %-11s %s\n",
        "noise", "r", "|G22|", "s11(circle)", "s22(circle)", "floor", "‖(z−L)⁻¹‖ ≤ on |z−1|=r")
println("="^104)

for w in noises
    NOISE = uniform_noise_box(w, 2)
    local L, La, n, out
    try
        L, La, n, _ = build(F, dom, NOISE, depth; periodic=per, enclosure_method=encl)
        out = certify_leading_eigs_sing(L, La, n; Rcirc=R, nblock_eig=80, margin=0.8, ρfloor=ρfl)
    catch e
        @printf("%-7.3f FAILED: %s\n", w, first(sprint(showerror,e),70)); flush(stdout); continue
    end
    # CIRCLE AROUND 1, enclosing ONLY the Perron eigenvalue (every other |λ| ≤ R = 0.6, so any
    # r < 1 − R = 0.4 is safe). On that contour σ₁₁ = σ_min(G₁₁ − zI) comes from the ADAPTIVE
    # PEARL COVER — verified svdbox on a small dense block, hence κ-FREE (the exact-SVD escape).
    # This is the whole point: the κ₂-transported block floor is useless here (κ₂ reaches 1e7),
    # while the direct SVD pays no transport at all and costs milliseconds.
    for r in (0.30, 0.20, 0.10)
        σ11 = Float64(RP._sigmin_circle(out.G11, complex(1.0, 0.0), r; ρfloor = 1e-4))
        σ22 = (1.0 - r) - out.G22norm            # min|z| on the circle is 1 − r
        floor = RP._assemble2(σ11, σ22, out.g12, out.g21)
        bound = floor > 0 ? 1 / floor : Inf
        @printf("%-7.3f %-6.2f %-9.4f %-11.4e %-11.4f %-11.4e %s\n",
                w, r, out.G22norm, σ11, σ22, floor,
                isfinite(bound) ? @sprintf("%.4f", bound) : "∞ (declines)")
        flush(stdout)
    end
end
println("="^104)
println("Multiply by ‖r_h‖ ≤ (δ/2)·var(f_h) to obtain the certified ‖f − f_h‖ error bound.")
