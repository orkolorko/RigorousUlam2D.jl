# Certified contraction:  find N with  ‖Lⁿ|_{U₀}‖ ≤ α < 1,  U₀ = mean-zero subspace.
# MATVEC ONLY (L is never assembled), single env (RigorousUlam2D + RigorousPseudospectra).
#
#   slow block:  Arnoldi(L) → H_{n+k};  miyajima_rump(H_{n+k}) certifies its eigenvalues and
#                κ₂ (diagonalising-frame conditioning); the SAFE leading Ritz (tight, separated
#                discs) give ‖(H|_{U₀})ⁿ‖ ≤ κ₂·ρ_slowⁿ and the deflation frame W.
#   far block:   G₂₂ = (I−WW*)L(I−WW*); certify σ_min(G₂₂−zI) ≥ s_far on |z|=r_far by matvec
#                Lanczos (BigFloat where the Float64 √u floor is too coarse) ⇒ ‖G₂₂ⁿ‖ ≤
#                r_far^{n+1}/s_far  (Cauchy).  BOTH are needed — H_{n+k} alone is not a
#                certificate, the far σ_min must be certified too.
#   combine:     ‖Lⁿ|_{U₀}‖ ≤ κ₂ρ_slowⁿ + ‖B‖·Σ_k ρ_slowᵏ·b_{n−1−k} + b_n,  b_j=r_far^{j+1}/s_far.
#
#   julia --project=examples --threads=auto examples/11_contraction_certified.jl

import Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()
include(joinpath(@__DIR__, "ulam_matvec.jl"))
using RigorousPseudospectra, Serialization

# certify the slow block from its oversized Arnoldi compression H (N×N) and residual βN.
# `refine` engages the §4e Sylvester–Newton basis refinement (entrywise + block-Sylvester),
# essential for clustered / near-defective Ritz values (e.g. Lorenz's 0.273, 0.273).
function certify_slow(H, βN; safe_r = 1e-3, refine = 6)
    mr = miyajima_rump(ComplexF64.(H); refine = refine, kappa_mode = :svdbox)
    discs = [(c = mr.centres[i], r = mr.radii[i],
              cl = length(mr.clusters[findfirst(cl -> i in cl.idx, mr.clusters)].idx))
             for i in eachindex(mr.centres)]
    # SAFE = singleton, tight disc, strictly inside the unit circle, and Arnoldi-converged
    # (disc radius dominates the Bauer–Fike residual to σ(L))
    safe = [d for d in discs if d.cl == 1 && d.r < safe_r && abs(d.c) < 0.999]
    return mr, safe, discs
end

function estimate_N(label, Lmul, Ladj, n; Nblock = 14, α = 0.5, nθ = 16, prec::Type = Float64)
    println("\n", "="^72); println(label, "  (n=$n boxes, matvec-only)"); println("="^72)

    # ---- slow block: oversized Arnoldi compression, certified by miyajima_rump ----
    print("  Arnoldi compression H_$Nblock (matvec)... "); flush(stdout)
    H, V, βN, N = arnoldi_block(Lmul, n, Nblock)
    @printf("done (β_N=%.3e).\n", βN)
    mr, safe, discs = certify_slow(H, βN)
    @printf("  miyajima_rump: κ₂=%.3f, ‖R₂‖=%.2e.  certified discs (|λ|, r):\n", mr.kappa, mr.nrmR2)
    for d in sort(discs; by = x -> abs(x.c), rev = true)
        tag = (d.cl == 1 && d.r < 1e-3 && abs(d.c) < 0.999) ? "SAFE" :
              (abs(d.c) > 0.999 ? "λ₁=1" : "tail")
        @printf("    |λ|=%.5f  r=%.2e  clu=%d  %s\n", abs(d.c), d.r, d.cl, tag)
    end
    isempty(safe) && (println("  ✗ no safe non-unit modes captured (raise Nblock)"); return)
    ρ_slow = maximum(abs(d.c) + d.r for d in safe)          # certified slow rate (disc-inflated)
    nsafe = length(safe) + 1                                # safe non-unit + λ₁
    @printf("  ⇒ %d safe slow modes, certified slow rate ρ_slow = %.5f, κ₂ = %.3f\n",
        length(safe), ρ_slow, mr.kappa)

    # deflation frame W = the leading nsafe Krylov vectors (span the certified slow subspace)
    W = Matrix(qr(V[:, 1:min(nsafe, N)]).Q)

    # ---- far block: certify σ_min(G₂₂−zI) on a circle in the gap below the safe modes ----
    r_far = ρ_slow * 0.7                                    # inside the gap, below ρ_slow
    @printf("  far block (%s): certifying σ_min(G₂₂−zI) on |z|=r_far=%.5f ...\n", prec, r_far)
    t = @elapsed (s_far, raw) = far_min_on_circle(Lmul, Ladj, W, r_far, n, prec; nθ = nθ, verbose = true)
    @printf("  far σ_min ≥ %.4e on |z|=r_far  (min sample %.4e, %.1fs)\n", s_far, raw, t)
    if s_far <= 0
        println("  ✗ far block NOT certified (needs more precision / different r_far) — no certificate")
        return
    end

    # ---- assemble  ‖Lⁿ|_{U₀}‖ ≤ κ₂ρ_slowⁿ + b_n + ‖B‖ Σ_k ρ_slowᵏ b_{n−1−k},  b_j=r_far^{j+1}/s_far
    Bcoup = norm(Lmul(W[:, 1]) .- W * (W' * Lmul(W[:, 1])))    # crude ‖(I−WW*)LW‖ proxy (col 1)
    b(j) = r_far^(j + 1) / s_far
    slow(nn) = mr.kappa * ρ_slow^nn
    total(nn) = slow(nn) + b(nn) + Bcoup * sum(slow(k) * b(nn - 1 - k) / mr.kappa for k in 0:(nn - 1))
    println("    n    κ₂ρ_slowⁿ   ‖G₂₂ⁿ‖     bound")
    Nstar = 0
    for nn in 1:300
        tb = total(nn)
        (nn <= 10 || tb < α || nn % 25 == 0) &&
            @printf("    %-4d %.3e  %.3e  %.4e %s\n", nn, slow(nn), b(nn), tb, tb < α ? "← ≤ α" : "")
        tb < α && (Nstar = nn; break)
    end
    rate = max(ρ_slow, r_far)
    if Nstar > 0
        @printf("  ✓ CERTIFIED: ‖Lⁿ|_{U₀}‖₂ ≤ α=%.2f at N = %d  (rate ρ = %.5f, κ₂=%.2f)\n",
            α, Nstar, rate, mr.kappa)
        @printf("    L¹: ‖·‖₁ ≤ √%d ‖·‖₂ ⇒ N_{L¹} ≈ %d\n",
            n, Nstar + ceil(Int, log(sqrt(n)) / (-log(rate))))
    else
        println("  ✗ bound did not reach α within 300 steps")
    end
    return Nstar
end

# ---------------------------------------------------------------- run
DEPTH = get(ENV, "DEPTH", "") == "" ? 5 : parse(Int, ENV["DEPTH"])
NBLOCK = get(ENV, "NBLOCK", "") == "" ? 14 : parse(Int, ENV["NBLOCK"])
ALPHA = get(ENV, "ALPHA", "") == "" ? 0.5 : parse(Float64, ENV["ALPHA"])
NTHETA = get(ENV, "NTHETA", "") == "" ? 16 : parse(Int, ENV["NTHETA"])
BITS = get(ENV, "BITS", "") == "" ? 192 : parse(Int, ENV["BITS"])
setprecision(BigFloat, BITS)
NOISE = uniform_noise_box(0.1, 2)

Lh_mul, Lh_adj, nh, _ = build_matvec(henon_rescaled,
    IntervalBox(interval(-1.0, 1.0), interval(-1.0, 1.0)), NOISE, DEPTH;
    periodic = true, enclosure_method = first_order_enclosure)
estimate_N("Hénon (periodic)", Lh_mul, Lh_adj, nh; Nblock = NBLOCK, α = ALPHA, nθ = NTHETA, prec = Float64)

Ll_mul, Ll_adj, nl, _ = build_matvec(contracting_lorenz,
    IntervalBox(interval(-0.5, 0.5), interval(-0.5, 0.5)), NOISE, DEPTH;
    periodic = false, enclosure_method = zeroth_order_enclosure)
estimate_N("Contracting Lorenz", Ll_mul, Ll_adj, nl; Nblock = NBLOCK, α = ALPHA, nθ = NTHETA, prec = BigFloat)
