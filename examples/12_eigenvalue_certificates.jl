# Certified enclosure of the first few interesting eigenvalues of the noisy transfer operator L.
# MATVEC ONLY (recompute scatter; L never assembled), single merged env.
#
# Block structure from Arnoldi:  L Vₙ = Vₙ Hₙ + β_N q_{N+1} eₙᵀ, so with W = Vₙ the slow block is
# Hₙ (= W*LW), the far block is G₂₂ = (I−WW*)L(I−WW*), and the coupling is exactly γ = β_N.
#   • Hₙ certified by miyajima_rump (refine = §4e Sylvester–Newton) → rigorous discs + κ₂.
#   • ONE far-block certificate σ̲₂₂ = σ_min(G₂₂ − z₀I) (matvec Lanczos, BigFloat for Lorenz),
#     REUSED for every small circle by Lipschitz: σ_min(G₂₂−zI) ≥ σ̲₂₂ − |z − z₀|. The Lanczos
#     collapses fast (noise kernel ⇒ G₂₂ low rank), so this is cheap.
#   • around each λᵢ: Γᵢ = {λᵢ + ρᵢe^{iθ}}, ρᵢ < ½ dist to nearest other eigenvalue;
#     σ₁₁(z) = σ_min(Hₙ − zI) (small SVD), σ₂₂(z) ≥ σ̲₂₂ − (|λᵢ−z₀|+ρᵢ);
#     σ_min(L−zI) ≥ σ₁₁σ₂₂/(σ₁₁+σ₂₂+γ) > 0 on Γᵢ ⇒ λᵢ isolated, gap ≥ ρᵢ.
# Writes a per-map LaTeX table.
#
#   julia --project=examples --threads=auto examples/12_eigenvalue_certificates.jl

import Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()
include(joinpath(@__DIR__, "ulam_matvec.jl"))
using RigorousPseudospectra, Printf

assemble(σ11, σ22, γ) = (σ11 <= 0 || σ22 <= 0) ? 0.0 : σ11 * σ22 / (σ11 + σ22 + γ)

function certify_eigs(label, tag, Lmul, Ladj, n; Nblock = 16, neig = 5, frac = 0.4,
        prec::Type = Float64, nθ = 24)
    println("\n", "="^74); println(label, "  (n=$n, matvec-only)"); println("="^74); flush(stdout)
    H, V, βN, N = arnoldi_block(Lmul, n, Nblock)
    mr = miyajima_rump(ComplexF64.(H); refine = 6, kappa_mode = :svdbox)
    λ = mr.centres; rad = mr.radii; ord = sortperm(abs.(λ); rev = true)
    γ = βN                                                  # coupling = Arnoldi residual
    @printf("  Arnoldi N=%d, β_N(=γ)=%.3e ; miyajima_rump κ₂=%.3f, ‖R₂‖=%.1e\n",
        N, βN, mr.kappa, mr.nrmR2); flush(stdout)

    # PER-CLUSTER capture: for the cluster containing λᵢ, deflate ONLY that cluster's Ritz
    # vectors Wᵢ = Vₙ·Yᵢ (Yᵢ = H's eigenvectors for the cluster). This is clean (the cluster is
    # removed exactly up to its Ritz residual γᵢ = ‖(I−WᵢWᵢ*)L Wᵢ‖), so the far block keeps the
    # OTHER modes and σ₂₂ = dist(λᵢ, σ(L)∖cluster) > 0 — no leak (unlike deflating the whole,
    # non-invariant Krylov basis). One far-block Lanczos per cluster, reused over its disk.
    # Use the CERTIFIED, refined frame from miyajima_rump: mr.W diagonalises H (Ã=mr.Atil), so
    # the cluster's columns of mr.W span its refined (Sylvester–Newton-decoupled) subspace, and
    # mr.Atil[cl,cl] is the certified cluster block. (Far better than raw eigen(H) Ritz vectors.)
    done = Set{Int}(); results = NamedTuple[]; nfar = 0
    for i in ord[1:min(neig, length(ord))]
        i in done && continue
        cl = mr.clusters[findfirst(c -> i in c.idx, mr.clusters)].idx
        union!(done, cl)
        c = λ[i]
        outside = [j for j in eachindex(λ) if !(j in cl)]
        gap = minimum(abs(c - λ[j]) for j in outside)        # dist to nearest eigenvalue outside
        Wi = Matrix(qr(V[:, 1:N] * mr.W[:, cl]).Q)            # lift the refined cluster frame
        LWi = reduce(hcat, [Lmul(Vector{ComplexF64}(Wi[:, j])) for j in axes(Wi, 2)])
        Hi = Wi' * LWi                                        # |cl|×|cl| cluster block
        γi = opnorm(LWi .- Wi * Hi, 2)                        # coupling = ‖(I−WᵢWᵢ*)L Wᵢ‖
        t = @elapsed (σ22c, R2, kf) = far_sigma_min(Lmul, Ladj, Wi, c, n, prec); nfar += 1
        # radius respects BOTH the gap (encloses only λᵢ) AND the far Lipschitz reach σ22c
        ρ = min(frac * gap, 0.5 * σ22c)
        σ11 = minimum(minimum(svdvals(Hi - (c + ρ * cis(2π * t2 / nθ)) * I)) for t2 in 0:(nθ - 1)) -
              2ρ * sin(π / (2nθ))
        σ22 = σ22c - ρ                                        # Lipschitz over the disk
        fl = assemble(σ11, σ22, γi); ok = fl > 0
        @printf("  λ=%+.6f%+.6fi |λ|=%.5f disc=%.1e clu=%d | ρ=%.4f σ₁₁=%.3e σ₂₂=%.3e γ=%.1e → σ_min≥%.3e %s [far k=%d,%.0fs]\n",
            real(c), imag(c), abs(c), rad[i], length(cl), ρ, σ11, σ22, γi, fl,
            ok ? "✓ ISOLATED" : "✗", kf, t)
        push!(results, (; c, disc = rad[i], ρ, σ11, σ22, γ = γi, floor = fl, certified = ok))
    end
    @printf("  far-block Lanczos runs: %d\n", nfar)
    flush(stdout)

    open(joinpath(@__DIR__, "eigcert_$(tag).tex"), "w") do io
        println(io, "% $(label): first $(length(results)) eigenvalues of L certified isolated.")
        println(io, "% n=$n, noise xi=0.1, matvec-only, kappa2=$(round(mr.kappa,digits=3)), coupling gamma=$(round(γ,sigdigits=3)).")
        println(io, "\\begin{table}[h]\\centering\\begin{tabular}{r l l l l l}\\toprule")
        println(io, "\$i\$ & \$\\lambda_i\$ (enclosed) & disc & \$\\rho_i\$ & \$\\sigma_{\\min}(L-zI)\$ on \$\\Gamma_i\$ & isolated \\\\\\midrule")
        for (k, r) in enumerate(results)
            @printf(io, "%d & \$%+.5f%+.5fi\$ & %.1e & %.4f & %.3e & %s \\\\\n",
                k, real(r.c), imag(r.c), r.disc, r.ρ, r.floor, r.certified ? "yes" : "no")
        end
        println(io, "\\bottomrule\\end{tabular}")
        println(io, "\\caption{$(label): the leading eigenvalues of the noisy transfer operator ",
            "\$L\$ ($(n) boxes) certified isolated. Each \$\\lambda_i\$ lies in a Miyajima--Rump ",
            "disc (refinement: \\S4e Sylvester--Newton); deflating its cluster gives a far block ",
            "whose \$\\sigma_{\\min}(G_{22}-zI)\$ (one matvec Lanczos, reused over \$\\Gamma_i\$ by ",
            "Lipschitz continuity) certifies \$\\sigma_{\\min}(L-zI)>0\$ on \$\\Gamma_i\$, so ",
            "\$\\Gamma_i\$ encloses exactly \$\\lambda_i\$ (gap \$\\ge\\rho_i\$). \$\\kappa_2=",
            "$(round(mr.kappa,digits=2))\$.}")
        println(io, "\\end{table}")
    end
    @printf("  → wrote eigcert_%s.tex\n", tag); flush(stdout)
    return results
end

DEPTH = get(ENV, "DEPTH", "") == "" ? 6 : parse(Int, ENV["DEPTH"])
NEIG = get(ENV, "NEIG", "") == "" ? 5 : parse(Int, ENV["NEIG"])
NBLOCK = get(ENV, "NBLOCK", "") == "" ? 16 : parse(Int, ENV["NBLOCK"])
BITS = get(ENV, "BITS", "") == "" ? 192 : parse(Int, ENV["BITS"]); setprecision(BigFloat, BITS)
NOISE = uniform_noise_box(0.1, 2)
build = get(ENV, "CACHED", "") == "1" ? build_matvec_cached : build_matvec   # cached spmv for quick checks

Lh, Lha, nh, _ = build(henon_rescaled,
    IntervalBox(interval(-1.0, 1.0), interval(-1.0, 1.0)), NOISE, DEPTH;
    periodic = true, enclosure_method = first_order_enclosure)
certify_eigs("Hénon (periodic)", "henon", Lh, Lha, nh; Nblock = NBLOCK, neig = NEIG, prec = Float64)

Ll, Lla, nl, _ = build(contracting_lorenz,
    IntervalBox(interval(-0.5, 0.5), interval(-0.5, 0.5)), NOISE, DEPTH;
    periodic = false, enclosure_method = zeroth_order_enclosure)
certify_eigs("Contracting Lorenz", "lorenz", Ll, Lla, nl; Nblock = NBLOCK, neig = NEIG, prec = BigFloat)
