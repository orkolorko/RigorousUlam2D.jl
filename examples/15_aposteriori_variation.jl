# A posteriori variation of the stationary density: var(L_ξ w) on the uniform grid.
#
# Gets the Perron (λ≈1) eigenvector w by Float64 power iteration on the matvec oracle,
# applies L once more, and evaluates the total variation of the density on the grid.
# TV of a piecewise-constant density (mass vector d, density = mass/(dx*dy)) on a
# uniform nx×ny grid is  Σ_faces |Δρ|·(edge length) = Σ_horiz |Δmass|/dx + Σ_vert |Δmass|/dy.
#
#   DEPTH=6 julia --project=examples --threads=auto examples/15_aposteriori_variation.jl
import Pkg
Pkg.activate(@__DIR__)
include(joinpath(@__DIR__, "ulam_matvec.jl"))
using RigorousUlam2D: build_uniform_partition, build_grid_index_maps, noise_variation,
                      first_order_enclosure, zeroth_order_enclosure
using LinearAlgebra, Printf

function perron(Lmul, n; iters = 400, tol = 1e-13)
    v = fill(1.0 / n, n)
    local w
    for _ in 1:iters
        w = Lmul(v)
        s = sum(w)
        s == 0 && break
        w ./= s
        norm(w .- v, 1) < tol && (v = w; break)
        v = w
    end
    return v
end

# total variation of the density whose per-box masses are d, on a uniform nx×ny grid
function grid_total_variation(d, part, nx, ny, domain; periodic)
    b2g, _ = build_grid_index_maps(part, (nx, ny))
    dx = diam(domain[1]) / nx
    dy = diam(domain[2]) / ny
    M = zeros(ny, nx)
    @inbounds for k in 1:length(d)
        ix, iy = b2g[k]
        M[iy, ix] = d[k]
    end
    tv = 0.0
    @inbounds for iy in 1:ny, ix in 1:nx
        if ix < nx
            tv += abs(M[iy, ix + 1] - M[iy, ix]) / dx
        elseif periodic
            tv += abs(M[iy, 1] - M[iy, ix]) / dx
        end
        if iy < ny
            tv += abs(M[iy + 1, ix] - M[iy, ix]) / dy
        elseif periodic
            tv += abs(M[1, ix] - M[iy, ix]) / dy
        end
    end
    return tv
end

DEPTH = get(ENV, "DEPTH", "") == "" ? 6 : parse(Int, ENV["DEPTH"])
NOISE = uniform_noise_box(0.1, 2)
var_ρ = noise_variation(NOISE)
build = build_matvec

@printf("\nDEPTH=%d  noise ξ=0.1  var(ρ)=%.4f\n", DEPTH, var_ρ)
println("="^70)

for (label, F, domain, periodic, enc) in [
    ("Henon", henon_rescaled, IntervalBox(interval(-1.0, 1.0), interval(-1.0, 1.0)),
        true, first_order_enclosure),
    ("Lorenz", contracting_lorenz, IntervalBox(interval(-0.5, 0.5), interval(-0.5, 0.5)),
        false, zeroth_order_enclosure),
]
    Lmul, Ladj, n, _ = build(F, domain, NOISE, DEPTH; periodic = periodic, enclosure_method = enc)
    part = build_uniform_partition(domain, DEPTH)
    nx = ny = 2^DEPTH
    w = perron(Lmul, n)
    Lw = Lmul(w); Lw ./= sum(Lw)
    vw = grid_total_variation(w, part, nx, ny, domain; periodic = periodic)
    vLw = grid_total_variation(Lw, part, nx, ny, domain; periodic = periodic)
    resid = norm(Lmul(w) ./ sum(Lmul(w)) .- w, 1)
    @printf("%-7s n=%d  |Lw-w|_1=%.2e   var(w)=%.4f   var(L w)=%.4f   [var(ρ)=%.2f]\n",
        label, n, resid, vw, vLw, var_ρ)
    flush(stdout)
end
