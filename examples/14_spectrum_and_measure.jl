# Certified spectral picture + stationary measure for the noisy transfer operator L,
# for Hénon and contracting Lorenz. For each example, ONE figure with two panels:
#   (left)  certified leading eigenvalues, each ISOLATED by a small circle, plus the
#           gap / contraction circle separating the leading block from the fast spectrum;
#   (right) the certified-leading invariant density (λ≈1 eigenvector) as a heatmap.
#
#   CACHED=1 NEIG=10 julia --project=examples --threads=auto examples/14_spectrum_and_measure.jl
import Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()
include(joinpath(@__DIR__, "ulam_matvec.jl"))
using RigorousPseudospectra, CairoMakie, Printf
using RigorousUlam2D: build_uniform_partition, build_grid_index_maps

function certify(label, Lmul, Ladj, n; nblock, neig, nsep, prec)
    println("\n", "="^70); println(label); println("="^70); flush(stdout)
    out = certify_leading_eigs(Lmul, Ladj, n; nblock = nblock, neig = neig, refine = 6, prec = prec)
    @printf("  κ₂=%.2f, β_N=%.2e\n", out.mr.kappa, out.βN)
    for r in out.results
        @printf("  λ=%+.5f%+.5fi |λ|=%.5f ρ=%.4f σ_min≥%.3e %s\n",
            real(r.λ), imag(r.λ), abs(r.λ), r.cert.radius, r.cert.floor,
            r.cert.certified ? "✓" : "✗")
    end
    # LOCKED gap certificate (singular-direction lock) — the sound, certifying variant.
    gfl, ginfo = certify_gap_locked(Lmul, Ladj, n; nsep = nsep, nblock = min(256, n - 2),
        band = GAPBAND, nθ = NTHETA, kmax = GAPKMAX, prec = prec)
    @printf("  GAP |z|=%.5f (nsep=%d, m=%d): σ_min≥%.3e %s (resolvent ≤ %.1f)\n",
        ginfo.radius, nsep, ginfo.m, gfl, ginfo.certified ? "✓" : "✗", gfl > 0 ? 1 / gfl : Inf)
    flush(stdout)
    return out, ginfo
end

# invariant density = the certified λ≈1 eigenvector, lifted, made positive, reshaped to the grid
function density_grid(out, domain, depth)
    nx = ny = 2^depth
    λ = out.mr.centres
    lead = argmax(abs.(λ))
    d = real(out.V * out.mr.W[:, lead])
    sum(d) < 0 && (d .= .-d)
    d ./= sum(abs, d)
    part = build_uniform_partition(domain, depth)
    b2g, _ = build_grid_index_maps(part, (nx, ny))
    Vd = fill(NaN, ny, nx)
    @inbounds for k in 1:length(d)
        ix, iy = b2g[k]
        Vd[iy, ix] = d[k]
    end
    Vd
end

function make_figure(label, out, ginfo, Vd, domain)
    fig = Figure(size = (1150, 500))
    # --- left: certified eigenvalues + isolation circles + gap circle ---
    ax1 = Axis(fig[1, 1]; xlabel = "Re", ylabel = "Im", title = "$label — certified spectrum",
        aspect = DataAspect())
    θu = range(0, 2π; length = 240)
    lines!(ax1, cos.(θu), sin.(θu); color = (:gray, 0.3), linestyle = :dash)        # unit circle
    nlead = sum(r.cluster for r in out.results)
    λ = out.Hritz
    scatter!(ax1, real.(λ[(nlead + 1):end]), imag.(λ[(nlead + 1):end]);
        color = (:gray, 0.55), markersize = 6, label = "fast spectrum")
    gc = ginfo.certified ? :seagreen : :red
    lines!(ax1, ginfo.radius .* cos.(θu), ginfo.radius .* sin.(θu); color = gc, linewidth = 2.5,
        label = ginfo.certified ? "gap |z|=$(round(ginfo.radius,digits=3))" : "gap (uncertified)")
    for r in out.results
        c = r.cert
        col = c.certified ? :seagreen : :red
        lines!(ax1, real(c.center) .+ c.radius .* cos.(θu), imag(c.center) .+ c.radius .* sin.(θu);
            color = col, linewidth = 1.4)
        scatter!(ax1, [real(r.λ)], [imag(r.λ)]; color = :black, marker = :xcross, markersize = 11)
    end
    axislegend(ax1; position = :lt, framevisible = false, labelsize = 9)
    # --- right: stationary measure (invariant density) ---
    xs = range(inf(domain[1]), sup(domain[1]); length = size(Vd, 2))
    ys = range(inf(domain[2]), sup(domain[2]); length = size(Vd, 1))
    ax2 = Axis(fig[1, 2]; xlabel = "x", ylabel = "y", title = "$label — stationary measure",
        aspect = DataAspect())
    hm = heatmap!(ax2, xs, ys, permutedims(Vd); colormap = :viridis)
    Colorbar(fig[1, 3], hm; label = "density")
    fig
end

DEPTH = get(ENV, "DEPTH", "") == "" ? 6 : parse(Int, ENV["DEPTH"])
NEIG = get(ENV, "NEIG", "") == "" ? 10 : parse(Int, ENV["NEIG"])
NSEP = get(ENV, "NSEP", "") == "" ? 1 : parse(Int, ENV["NSEP"])
NBLOCK = get(ENV, "NBLOCK", "") == "" ? 28 : parse(Int, ENV["NBLOCK"])
BITS = get(ENV, "BITS", "") == "" ? 192 : parse(Int, ENV["BITS"]); setprecision(BigFloat, BITS)
NTHETA = get(ENV, "NTHETA", "") == "" ? 512 : parse(Int, ENV["NTHETA"])   # gap-circle samples
GAPKMAX = get(ENV, "GAPKMAX", "") == "" ? 120 : parse(Int, ENV["GAPKMAX"]) # far-block Lanczos depth
GAPBAND = get(ENV, "GAPBAND", "") == "" ? 0.2 : parse(Float64, ENV["GAPBAND"])
build = get(ENV, "CACHED", "") == "1" ? build_matvec_cached : build_matvec
NOISE = uniform_noise_box(0.1, 2)

dh = IntervalBox(interval(-1.0, 1.0), interval(-1.0, 1.0))
Lh, Lha, nh, _ = build(henon_rescaled, dh, NOISE, DEPTH; periodic = true, enclosure_method = first_order_enclosure)
oh, gih = certify("Hénon", Lh, Lha, nh; nblock = NBLOCK, neig = NEIG, nsep = NSEP, prec = Float64)
figh = make_figure("Hénon", oh, gih, density_grid(oh, dh, DEPTH), dh)
save(joinpath(@__DIR__, "14_henon.png"), figh)
println("saved 14_henon.png")

dl = IntervalBox(interval(-0.5, 0.5), interval(-0.5, 0.5))
Ll, Lla, nl, _ = build(contracting_lorenz, dl, NOISE, DEPTH; periodic = false, enclosure_method = zeroth_order_enclosure)
ol, gil = certify("Contracting Lorenz", Ll, Lla, nl; nblock = NBLOCK, neig = NEIG, nsep = NSEP, prec = BigFloat)
figl = make_figure("Contracting Lorenz", ol, gil, density_grid(ol, dl, DEPTH), dl)
save(joinpath(@__DIR__, "14_lorenz.png"), figl)
println("saved 14_lorenz.png")
