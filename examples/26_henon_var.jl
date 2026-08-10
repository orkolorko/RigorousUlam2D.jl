# var(f_h) for periodic Hénon at a=0.3 via CACHED sparse transfer + O(n) box_noise power
# method (fast). Domain [-1,1]² (width 2 ⇒ δ=2/2^depth, |M|=4).
import Pkg; Pkg.activate(@__DIR__)
include(joinpath(@__DIR__, "ulam_matvec.jl"))
using RigorousUlam2D: build_uniform_partition, build_grid_index_maps, noise_variation,
                      zeroth_order_enclosure
using LinearAlgebra, Printf

function perron(Lmul, n; iters = 150, tol = 1e-9)
    v = fill(1.0 / n, n); local w
    for _ in 1:iters
        w = Lmul(v); s = sum(w); s == 0 && break; w ./= s
        norm(w .- v, 1) < tol && (v = w; break); v = w
    end
    return v
end
function grid_tv(d, part, nx, ny, domain)
    b2g, _ = build_grid_index_maps(part, (nx, ny))
    dx = diam(domain[1]) / nx; dy = diam(domain[2]) / ny
    M = zeros(ny, nx); for k in 1:length(d); ix, iy = b2g[k]; M[iy, ix] = d[k]; end
    tv = 0.0
    for iy in 1:ny, ix in 1:nx
        tv += abs(M[iy, ix % nx + 1] - M[iy, ix]) / dx + abs(M[iy % ny + 1, ix] - M[iy, ix]) / dy
    end
    return tv
end

A = 0.3; NOISE = uniform_noise_box(A, 2); varρ = noise_variation(NOISE)
dh = IntervalBox(interval(-1.0, 1.0), interval(-1.0, 1.0))   # |M|=4
DEPTHS = get(ENV, "DEPTHS", "") == "" ? [8, 10, 11] : parse.(Int, split(ENV["DEPTHS"], ","))
@printf("periodic Hénon a=%.1f  var(ρ)=%.3f  |M|=4\n", A, varρ)
@printf("%4s %10s %9s %11s %10s %10s\n", "h", "n", "var(f_h)", "δ", "‖r‖=(δ/2)vf", "build+solve")
for h in DEPTHS
    δ = 2.0 / 2^h
    t = @elapsed begin
        L, La, n, _ = build_matvec_cached(henon_rescaled, dh, NOISE, h; periodic = true, enclosure_method = zeroth_order_enclosure)
        global w = perron(L, n); global nn = n
    end
    part = build_uniform_partition(dh, h); nx = ny = 2^h
    vw = grid_tv(w, part, nx, ny, dh)
    r = (δ / 2) * vw
    @printf("%4d %10d %9.4f %11.6f %11.6f %9.1fs\n", h, nn, vw, δ, r, t)
    flush(stdout)
end
