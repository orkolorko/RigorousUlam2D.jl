# Stationary-measure heatmaps at a=0.3 (periodic) for Hénon and Lorenz, via the power method.
# Density on the uniform grid = mass/box_area; saved as PNGs for the draft.
import Pkg; Pkg.activate(@__DIR__)
include(joinpath(@__DIR__, "ulam_matvec.jl"))
using RigorousUlam2D: build_uniform_partition, build_grid_index_maps,
                      first_order_enclosure, zeroth_order_enclosure
using LinearAlgebra, Printf, CairoMakie

function perron(Lmul, n; iters = 200, tol = 1e-10)
    v = fill(1.0 / n, n); local w
    for _ in 1:iters
        w = Lmul(v); s = sum(w); s == 0 && break; w ./= s
        norm(w .- v, 1) < tol && (v = w; break); v = w
    end
    return v
end

function density_grid(w, part, nx, ny, domain)
    b2g, _ = build_grid_index_maps(part, (nx, ny))
    area = (diam(domain[1]) / nx) * (diam(domain[2]) / ny)
    Vd = fill(NaN, ny, nx)
    for k in 1:length(w); ix, iy = b2g[k]; Vd[iy, ix] = w[k] / area; end
    return Vd
end

DEPTH = get(ENV, "DEPTH", "") == "" ? 8 : parse(Int, ENV["DEPTH"])
NOISE = uniform_noise_box(0.3, 2)

for (label, F, domain, enc, fname) in [
    ("Hénon", henon_rescaled, IntervalBox(interval(-1.0, 1.0), interval(-1.0, 1.0)),
        first_order_enclosure, "14_henon_a03_density.png"),
    ("Lorenz", contracting_lorenz, IntervalBox(interval(-0.5, 0.5), interval(-0.5, 0.5)),
        zeroth_order_enclosure, "14_lorenz_a03_density.png"),
]
    L, _, n, _ = build_matvec_cached(F, domain, NOISE, DEPTH; periodic = true, enclosure_method = enc)
    w = perron(L, n)
    part = build_uniform_partition(domain, DEPTH); nx = ny = 2^DEPTH
    Vd = density_grid(w, part, nx, ny, domain)
    xs = range(inf(domain[1]), sup(domain[1]); length = nx)
    ys = range(inf(domain[2]), sup(domain[2]); length = ny)
    fig = Figure(size = (520, 460))
    ax = Axis(fig[1, 1]; xlabel = "x", ylabel = "y",
        title = "$label — stationary density (a=0.3, periodic)", aspect = DataAspect())
    hm = heatmap!(ax, xs, ys, permutedims(Vd); colormap = :viridis)
    Colorbar(fig[1, 2], hm; label = "density")
    save(joinpath(@__DIR__, fname), fig)
    @printf("saved %s (n=%d, max density %.2f)\n", fname, n, maximum(filter(!isnan, Vd)))
    flush(stdout)
end
