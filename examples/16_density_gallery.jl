# Stationary-measure gallery: for each map, the certified stationary density at four noise
# half-widths a ∈ {0.05, 0.1, 0.2, 0.3} (box noise [-a,a]², side ξ=2a, var(ρ_ξ)=2/a).
# P_det is built ONCE per map (noise-independent) and reused across the four noise sizes.
# The spectral gap that certifies each measure is the |z|=0.6 exterior certificate at depth 6
# (results/*_R06_certified.txt, noise_sweep_*.txt); here we compute+plot the densities and the
# residual ‖r_h‖ ≤ (δ/2)var(f_h). Density on a depth-8 grid (decoupling: certify coarse, refine free).
import Pkg; Pkg.activate(@__DIR__)
include(joinpath(@__DIR__, "ulam_matvec.jl"))
using RigorousUlam2D: build_uniform_partition, build_grid_index_maps,
                      first_order_enclosure, zeroth_order_enclosure
using LinearAlgebra, Printf, SparseArrays, CairoMakie

perron(Lmul, n; iters = 500, tol = 1e-11) = begin
    v = fill(1.0 / n, n)
    for _ in 1:iters
        w = Lmul(v); s = sum(w); s == 0 && break; w ./= s
        d = norm(w .- v, 1); v = w; d < tol && break
    end
    v
end

function density_grid(w, b2g, nx, ny, area)
    Vd = fill(NaN, ny, nx)
    for k in 1:length(w); ix, iy = b2g[k]; Vd[iy, ix] = w[k] / area; end
    Vd
end

# anisotropic (ℓ¹) total variation of a piecewise-constant density on the grid, periodic wrap
function grid_tv(Vd, dx, dy; periodic = true)
    ny, nx = size(Vd); tv = 0.0
    for iy in 1:ny, ix in 1:nx
        ixr = ix < nx ? ix + 1 : (periodic ? 1 : 0)
        iyd = iy < ny ? iy + 1 : (periodic ? 1 : 0)
        ixr != 0 && (tv += abs(Vd[iy, ixr] - Vd[iy, ix]) * dy)
        iyd != 0 && (tv += abs(Vd[iyd, ix] - Vd[iy, ix]) * dx)
    end
    tv
end

DEPTH = parse(Int, get(ENV, "DEPTH", "8"))
NOISES = [0.05, 0.1, 0.2, 0.3]          # half-widths a
configs = [
    ("henon", "Hénon", henon_rescaled,
        IntervalBox(interval(-1.0, 1.0), interval(-1.0, 1.0)), first_order_enclosure, true),
    ("lorenz", "contracting Lorenz", contracting_lorenz,
        IntervalBox(interval(-0.5, 0.5), interval(-0.5, 0.5)), zeroth_order_enclosure, true),
]

# ---- convention check: depth-6 leading spectrum for Hénon a=0.1 must reproduce the certificate
if get(ENV, "CHECK", "1") == "1"
    part6 = build_uniform_partition(configs[1][4], 6); n6 = part6.n_boxes
    b2g6, _ = build_grid_index_maps(part6, (64, 64))
    Pd6 = RigorousUlam2D.build_transfer_forward(configs[1][3], part6, (64, 64);
        subdiv = 4, enclosure_method = first_order_enclosure, periodic = true)
    RigorousUlam2D.normalize_columns!(Pd6)
    dx6 = diam(configs[1][4][1]) / 64; rx6 = max(1, Int(round(0.1 / dx6)))
    L6(v) = box_noise(Pd6 * v, rx6, rx6, 64, 64, b2g6; periodic = true)
    H, _, _ = arnoldi_block(L6, n6, 40)
    ev = sort(abs.(eigvals(H)); rev = true)[1:5]
    @printf("[check] Hénon depth6 a=0.1 rx=%d  top|λ|: %s\n", rx6,
        join((@sprintf("%.4f", e) for e in ev), ", "))
    @printf("        (certificate: 1.0000, 0.6955, 0.6955, 0.6748)\n"); flush(stdout)
end

for (tag, label, F, dom, encl, per) in configs
    nx = ny = 2^DEPTH
    part = build_uniform_partition(dom, DEPTH); n = part.n_boxes
    b2g, _ = build_grid_index_maps(part, (nx, ny))
    dx = diam(dom[1]) / nx; dy = diam(dom[2]) / ny; area = dx * dy
    @printf("== %s: building P_det at depth %d (n=%d) ==\n", label, DEPTH, n); flush(stdout)
    Pd = RigorousUlam2D.build_transfer_forward(F, part, (nx, ny);
        subdiv = 4, enclosure_method = encl, periodic = per)
    RigorousUlam2D.normalize_columns!(Pd)

    fig = Figure(size = (1080, 940))
    for (idx, a) in enumerate(NOISES)
        rx = max(1, Int(round(a / dx))); ry = max(1, Int(round(a / dy)))
        Lmul(v) = box_noise(Pd * v, rx, ry, nx, ny, b2g; periodic = per)
        w = perron(Lmul, n)
        Vd = density_grid(w, b2g, nx, ny, area)
        tvf = grid_tv(Vd, dx, dy; periodic = per)
        rh = (dx / 2) * tvf
        @printf("%s a=%.2f ξ=%.2f var(ρ)=%.1f rx=%d  maxρ=%.3f  var(f_h)=%.3f  ‖r_h‖≤%.2e\n",
            label, a, 2a, 2 / a, rx, maximum(Vd), tvf, rh); flush(stdout)
        row, blk = fldmod1(idx, 2); acol = 2blk - 1
        ax = Axis(fig[row, acol]; xlabel = "x", ylabel = "y", aspect = DataAspect(),
            title = @sprintf("a=%.2f  (ξ=%.2f, var ρ=%.1f)", a, 2a, 2 / a))
        xs = range(inf(dom[1]), sup(dom[1]); length = nx)
        ys = range(inf(dom[2]), sup(dom[2]); length = ny)
        hm = heatmap!(ax, xs, ys, permutedims(Vd); colormap = :viridis)
        Colorbar(fig[row, acol + 1], hm)
    end
    Label(fig[0, :], "$label — certified stationary densities (periodic, depth $DEPTH)",
        fontsize = 18, font = :bold)
    out = joinpath(@__DIR__, "..", "results", "gallery_$(tag).png")
    save(out, fig)
    @printf("saved %s\n", out); flush(stdout)
end
