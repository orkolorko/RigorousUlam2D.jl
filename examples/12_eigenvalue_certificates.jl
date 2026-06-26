# Certified enclosure of the leading eigenvalues of the noisy transfer operator L, matvec-only.
# Uses RigorousPseudospectra.certify_leading_eigs (the Krylov–Rump–Miyajima decoupling: Arnoldi
# → miyajima_rump refined block-diagonal → per-cluster resolvent circle). This script only
# builds the operator and writes a per-map LaTeX table.
#
# REMARK. This certifies each leading eigenvalue ISOLATED (a small circle around it has no other
# eigenvalue). The SPECTRAL GAP needs one more step: with λ₁,…,λ_k the certified leading
# eigenvalues, deflate all k and certify σ_min(z−G₂₂)>0 on a circle enclosing the REMAINING
# spectrum λ_{k+1},… — a resolvent bound on a contour enclosing σ(G₂₂), the contour
# certification of Nisoli (Gauss paper, arXiv:2602.19435). Same far-block primitive run ONCE on
# that bulk-enclosing circle (cf. example 11); the binding step for strongly non-normal operators.
#
#   julia --project=examples --threads=auto examples/12_eigenvalue_certificates.jl
#   CACHED=1 ...   # cached P_det spmv (fast; recompute matvec is the scaling path)

import Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()
include(joinpath(@__DIR__, "ulam_matvec.jl"))
using RigorousPseudospectra, Printf

function run_map(label, tag, Lmul, Ladj, n; nblock = 16, neig = 5, prec = Float64)
    println("\n", "="^74); println(label, "  (n=$n boxes, matvec-only)"); println("="^74); flush(stdout)
    t = @elapsed out = certify_leading_eigs(Lmul, Ladj, n;
        nblock = nblock, neig = neig, refine = 6, prec = prec, frac = 0.4)
    @printf("  Arnoldi N=%d β_N=%.2e ; miyajima_rump κ₂=%.3f, ‖R₂‖=%.1e ; %.0fs\n",
        nblock, out.βN, out.mr.kappa, out.mr.nrmR2, t)
    for r in out.results
        c = r.cert
        @printf("  λ=%+.6f%+.6fi |λ|=%.5f disc=%.1e clu=%d | ρ=%.4f σ_min≥%.3e %s\n",
            real(r.λ), imag(r.λ), abs(r.λ), r.disc, r.cluster, c.radius, c.floor,
            c.certified ? "✓ ISOLATED" : "✗")
    end
    flush(stdout)

    open(joinpath(@__DIR__, "eigcert_$(tag).tex"), "w") do io
        println(io, "% $(label): leading eigenvalues of L certified isolated (matvec-only).")
        println(io, "% n=$n, noise xi=0.1, kappa2=$(round(out.mr.kappa,digits=3)), Arnoldi residual beta_N=$(round(out.βN,sigdigits=3)).")
        println(io, "\\begin{table}[h]\\centering\\begin{tabular}{r l l l l l}\\toprule")
        println(io, "\$i\$ & \$\\lambda_i\$ (enclosed) & disc & clu & \$\\rho_i\$ & \$\\sigma_{\\min}(L-zI)\$ on \$\\Gamma_i\$ \\\\\\midrule")
        for (k, r) in enumerate(out.results)
            @printf(io, "%d & \$%+.5f%+.5fi\$ & %.1e & %d & %.4f & %.3e %s\\\\\n",
                k, real(r.λ), imag(r.λ), r.disc, r.cluster, r.cert.radius, r.cert.floor,
                r.cert.certified ? "" : "(uncert.) ")
        end
        println(io, "\\bottomrule\\end{tabular}")
        println(io, "\\caption{$(label): the leading eigenvalues of the noisy transfer operator ",
            "\$L\$ ($(n) boxes) certified isolated by the Krylov--Rump--Miyajima decoupling ",
            "(Arnoldi \$\\to\$ Miyajima--Rump refined block-diagonalization, \\S4e ",
            "Sylvester--Newton \$\\to\$ per-cluster resolvent circle). Each \$\\lambda_i\$ lies in ",
            "a certified disc; on \$\\Gamma_i=\\{\\lambda_i+\\rho_i e^{i\\theta}\\}\$ the matvec ",
            "Lanczos certifies \$\\sigma_{\\min}(L-zI)>0\$, so \$\\Gamma_i\$ encloses exactly ",
            "\$\\lambda_i\$ (gap \$\\ge\\rho_i\$). \$\\kappa_2=$(round(out.mr.kappa,digits=2))\$.}")
        println(io, "\\end{table}")
    end
    @printf("  → wrote eigcert_%s.tex\n", tag); flush(stdout)
    return out
end

DEPTH = get(ENV, "DEPTH", "") == "" ? 6 : parse(Int, ENV["DEPTH"])
NEIG = get(ENV, "NEIG", "") == "" ? 5 : parse(Int, ENV["NEIG"])
NBLOCK = get(ENV, "NBLOCK", "") == "" ? 16 : parse(Int, ENV["NBLOCK"])
BITS = get(ENV, "BITS", "") == "" ? 192 : parse(Int, ENV["BITS"]); setprecision(BigFloat, BITS)
build = get(ENV, "CACHED", "") == "1" ? build_matvec_cached : build_matvec
NOISE = uniform_noise_box(0.1, 2)

Lh, Lha, nh, _ = build(henon_rescaled,
    IntervalBox(interval(-1.0, 1.0), interval(-1.0, 1.0)), NOISE, DEPTH;
    periodic = true, enclosure_method = first_order_enclosure)
run_map("Hénon (periodic)", "henon", Lh, Lha, nh; nblock = NBLOCK, neig = NEIG, prec = Float64)

Ll, Lla, nl, _ = build(contracting_lorenz,
    IntervalBox(interval(-0.5, 0.5), interval(-0.5, 0.5)), NOISE, DEPTH;
    periodic = false, enclosure_method = zeroth_order_enclosure)
run_map("Contracting Lorenz", "lorenz", Ll, Lla, nl; nblock = NBLOCK, neig = NEIG, prec = BigFloat)
