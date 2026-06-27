# Certified spectral picture of the noisy transfer operator L, for Hénon and contracting Lorenz:
#   • the leading eigenvalues, each certified ISOLATED by a small circle (certify_leading_eigs);
#   • the CONTRACTION / gap circle separating the certified leading block from the really fast
#     spectrum, certified by certify_gap (the far-block Gram / Gauss-paper method).
# Produces one figure with both maps: eigenvalues, isolation circles, and the gap circle.
#
#   CACHED=1 julia --project=examples --threads=auto examples/13_spectral_certificate_plot.jl

import Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()
include(joinpath(@__DIR__, "ulam_matvec.jl"))
using RigorousPseudospectra, CairoMakie, Printf

function certify_map(label, Lmul, Ladj, n; nblock = 20, neig = 5, prec = Float64)
    println("\n", "="^70); println(label); println("="^70); flush(stdout)
    out = certify_leading_eigs(Lmul, Ladj, n; nblock = nblock, neig = neig, refine = 6, prec = prec)
    @printf("  miyajima_rump κ₂=%.2f, β_N=%.2e\n", out.mr.kappa, out.βN)
    for r in out.results
        @printf("  λ=%+.5f%+.5fi |λ|=%.5f ρ=%.4f σ_min≥%.3e %s\n",
            real(r.λ), imag(r.λ), abs(r.λ), r.cert.radius, r.cert.floor,
            r.cert.certified ? "✓" : "✗")
    end
    # gap circle: between the smallest certified-leading |λ| and the largest fast |λ|
    nlead = sum(r.cluster for r in out.results)
    av = abs.(out.Hritz)
    r_gap = nlead < length(av) ? sqrt(av[nlead] * av[nlead + 1]) : 0.5 * av[nlead]
    gfl, ginfo = certify_gap(Lmul, Ladj, n, out.Wlead, r_gap; nθ = 64, prec = prec)
    @printf("  GAP circle |z|=%.5f: σ_min≥%.3e %s (‖R₂‖=%.1e, σ22min=%.3e)\n",
        r_gap, gfl, ginfo.certified ? "✓ separates leading from fast spectrum" : "✗",
        ginfo.R2, ginfo.σ22min)
    flush(stdout)
    return out, r_gap, ginfo
end

function plot_panel!(ax, label, out, r_gap, ginfo)
    θu = range(0, 2π; length = 240)
    lines!(ax, cos.(θu), sin.(θu); color = (:gray, 0.35), linestyle = :dash)   # unit circle
    nlead = sum(r.cluster for r in out.results)
    λ = out.Hritz
    # fast (inside the gap circle) vs leading
    scatter!(ax, real.(λ[(nlead + 1):end]), imag.(λ[(nlead + 1):end]);
        color = (:gray, 0.6), markersize = 7, label = "fast spectrum")
    # gap / contraction circle
    gc = ginfo.certified ? :seagreen : :red
    lines!(ax, r_gap .* cos.(θu), r_gap .* sin.(θu); color = gc, linewidth = 2.5,
        label = ginfo.certified ? "gap circle (contraction ≤ $(round(r_gap,digits=3)))" : "gap (uncertified)")
    # leading eigenvalues + their isolation circles
    for r in out.results
        c = r.cert
        col = c.certified ? :seagreen : :red
        lines!(ax, real(c.center) .+ c.radius .* cos.(θu), imag(c.center) .+ c.radius .* sin.(θu);
            color = col, linewidth = 1.5)
        scatter!(ax, [real(r.λ)], [imag(r.λ)]; color = :black, marker = :xcross, markersize = 12)
    end
    ax.title = label
    ax.aspect = DataAspect()
    axislegend(ax; position = :lt, framevisible = false, labelsize = 9)
end

DEPTH = get(ENV, "DEPTH", "") == "" ? 6 : parse(Int, ENV["DEPTH"])
NEIG = get(ENV, "NEIG", "") == "" ? 5 : parse(Int, ENV["NEIG"])
NBLOCK = get(ENV, "NBLOCK", "") == "" ? 20 : parse(Int, ENV["NBLOCK"])
BITS = get(ENV, "BITS", "") == "" ? 192 : parse(Int, ENV["BITS"]); setprecision(BigFloat, BITS)
build = get(ENV, "CACHED", "") == "1" ? build_matvec_cached : build_matvec
NOISE = uniform_noise_box(0.1, 2)

Lh, Lha, nh, _ = build(henon_rescaled,
    IntervalBox(interval(-1.0, 1.0), interval(-1.0, 1.0)), NOISE, DEPTH;
    periodic = true, enclosure_method = first_order_enclosure)
oh, rgh, gih = certify_map("Hénon (periodic)", Lh, Lha, nh; nblock = NBLOCK, neig = NEIG, prec = Float64)

Ll, Lla, nl, _ = build(contracting_lorenz,
    IntervalBox(interval(-0.5, 0.5), interval(-0.5, 0.5)), NOISE, DEPTH;
    periodic = false, enclosure_method = zeroth_order_enclosure)
ol, rgl, gil = certify_map("Contracting Lorenz", Ll, Lla, nl; nblock = NBLOCK, neig = NEIG, prec = BigFloat)

fig = Figure(size = (1200, 560))
plot_panel!(Axis(fig[1, 1]; xlabel = "Re", ylabel = "Im"), "Hénon: certified spectrum", oh, rgh, gih)
plot_panel!(Axis(fig[1, 2]; xlabel = "Re", ylabel = "Im"), "Contracting Lorenz: certified spectrum", ol, rgl, gil)
out = joinpath(@__DIR__, "13_spectral_certificate.png")
save(out, fig)
println("\nSaved ", out)
