# Feasibility test: rigorous interval upper bound on var(f_h) for one config.
# var(f_h) ≤ var(P_h w) + var(ρ_ξ)·‖f_h−w‖₁ (a-posteriori variation, discrete).
# P_det from build_transition_matrix (rigorous lower P and width U ⇒ weight ∈ [P, P+U]);
# box_noise is exact grid-window averaging (monotone), applied to the lo/hi enclosure.
import Pkg; Pkg.activate(@__DIR__)
include(joinpath(@__DIR__, "ulam_matvec.jl"))
using RigorousUlam2D: build_uniform_partition, build_grid_index_maps, build_transition_matrix,
                      first_order_enclosure
using LinearAlgebra, Printf, SparseArrays, Statistics

DEPTH = parse(Int, get(ENV, "DEPTH", "6"))
A     = parse(Float64, get(ENV, "A", "0.3"))
dom   = IntervalBox(interval(-1.0, 1.0), interval(-1.0, 1.0))
nx = ny = 2^DEPTH
part = build_uniform_partition(dom, DEPTH); n = part.n_boxes
b2g, _ = build_grid_index_maps(part, (nx, ny))
dx = diam(dom[1]) / nx; dy = diam(dom[2]) / ny
rx = max(1, Int(round(A / dx))); ry = max(1, Int(round(A / dy)))

@printf("rigorous var test: Hénon depth %d n=%d a=%.2f rx=%d\n", DEPTH, n, A, rx); flush(stdout)

t0 = time()
P, U = build_transition_matrix(henon_rescaled, part; enclosure_method = first_order_enclosure,
                               periodic = true)
@printf("  build_transition_matrix: %.1fs  nnz(P)=%d  max col-sum gap=%.2e\n",
        time() - t0, nnz(P), maximum(abs.(1 .- vec(sum(P; dims = 1))))); flush(stdout)
Phi = P .+ U                                   # upper-bound weights

# float power method on the midpoint operator (for w)
Pmid = P .+ U ./ 2
@printf("  loose columns (colsum(P)<0.9): %d of %d ; median colsum(P)=%.4f\n",
        count(<(0.9), vec(sum(P; dims=1))), n, median(vec(sum(P; dims=1)))); flush(stdout)
noised(v) = box_noise(Pmid * v, rx, ry, nx, ny, b2g; periodic = true)
function powmeth(noised, n)
    w = fill(1.0 / n, n)
    for _ in 1:600
        z = noised(w); z ./= sum(z)
        norm(z .- w, 1) < 1e-13 && return z
        w = z
    end
    w
end
w = powmeth(noised, n)

# rigorous enclosure of P_h w = boxnoise(L_T w), w ≥ 0 ⇒ L_T w ∈ [P*w, Phi*w]
lo = box_noise(P * w, rx, ry, nx, ny, b2g; periodic = true)
hi = box_noise(Phi * w, rx, ry, nx, ny, b2g; periodic = true)
area = dx * dy
dlo = lo ./ area; dhi = hi ./ area                # density enclosure

# rigorous upper bound on var(density) = Σ_edges sup|Δ|·len (periodic), enclosure per node
function var_upper(dlo, dhi, b2g, nx, ny, dx, dy)
    g = fill(0, ny, nx); glo = fill(0.0, ny, nx); ghi = fill(0.0, ny, nx)
    for k in 1:length(dlo); ix, iy = b2g[k]; glo[iy, ix] = dlo[k]; ghi[iy, ix] = dhi[k]; end
    tv = 0.0
    for iy in 1:ny, ix in 1:nx
        ixr = mod1(ix + 1, nx); iyd = mod1(iy + 1, ny)
        # sup|d_a - d_b| over enclosures [glo,ghi]
        sx = max(abs(ghi[iy, ix] - glo[iy, ixr]), abs(ghi[iy, ixr] - glo[iy, ix]))
        sy = max(abs(ghi[iy, ix] - glo[iyd, ix]), abs(ghi[iyd, ix] - glo[iy, ix]))
        tv += sx * dy + sy * dx
    end
    tv
end
var_hi = var_upper(dlo, dhi, b2g, nx, ny, dx, dy)

# float var for comparison (midpoint density)
dmid = (dlo .+ dhi) ./ 2
var_mid = let g = fill(0.0, ny, nx)
    for k in 1:length(dmid); ix, iy = b2g[k]; g[iy, ix] = dmid[k]; end
    tv = 0.0
    for iy in 1:ny, ix in 1:nx
        tv += abs(g[iy, mod1(ix+1,nx)] - g[iy,ix]) * dy + abs(g[mod1(iy+1,ny), ix] - g[iy,ix]) * dx
    end
    tv
end

alg_res = norm(box_noise(Pmid * w, rx, ry, nx, ny, b2g; periodic=true) .- w, 1)
@printf("  var(P_h w): float≈%.4f   RIGOROUS upper≤%.4f   (enclosure width %.2e)\n",
        var_mid, var_hi, var_hi - var_mid)
@printf("  algebraic residual ‖P_h w − w‖₁ = %.2e ;  var(ρ_ξ)=%.1f\n", alg_res, 2/A)
@printf("  total time %.1fs\n", time() - t0); flush(stdout)
