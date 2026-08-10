# var(f_{δ,ξ}) for periodic Lorenz at a=0.3, depth 6 — the computed input to the
# a-posteriori residual  ‖r‖ ≤ ε₁ + (δ/2)(var(w)+var(ρ))  (ε₁=‖P_δ w-w‖≈0).
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
# TV of density (mass vector d) on uniform nx×ny grid = Σ|Δmass|/dx + Σ|Δmass|/dy (periodic)
function grid_tv(d, part, nx, ny, domain)
    b2g, _ = build_grid_index_maps(part, (nx, ny))
    dx = diam(domain[1]) / nx; dy = diam(domain[2]) / ny
    M = zeros(ny, nx)
    for k in 1:length(d); ix, iy = b2g[k]; M[iy, ix] = d[k]; end
    tv = 0.0
    for iy in 1:ny, ix in 1:nx
        tv += abs(M[iy, ix % nx + 1] - M[iy, ix]) / dx
        tv += abs(M[iy % ny + 1, ix] - M[iy, ix]) / dy
    end
    return tv
end

A = 0.3
NOISE = uniform_noise_box(A, 2)
dl = IntervalBox(interval(-0.5, 0.5), interval(-0.5, 0.5))
varρ = noise_variation(NOISE)
Aabs = 17.2   # abstract reduced resolvent ‖(I-L_ξ)^{-1}|_V‖ ≤ R₁/(1-R₁δ_h) from depth-6 cert (R₁=9.1)
DEPTHS = get(ENV, "DEPTHS", "") == "" ? [6, 8, 9] : parse.(Int, split(ENV["DEPTHS"], ","))
@printf("periodic Lorenz a=%.1f  var(ρ)=%.3f  abstract resolvent ≤ %.1f (from depth-6 cert)\n", A, varρ, Aabs)
@printf("%6s %9s %9s %10s %12s %12s\n", "d'", "n", "var(w)", "δ", "‖r‖bound", "≈A·‖r‖ (err)")
for DEPTH in DEPTHS
    δ = 1.0 / 2^DEPTH
    # matrix-free matvec (power method only needs matvecs) — low memory, feasible at n=10^6
    L, La, n, _ = build_matvec(contracting_lorenz, dl, NOISE, DEPTH;
        periodic = true, enclosure_method = zeroth_order_enclosure)
    w = perron(L, n)
    part = build_uniform_partition(dl, DEPTH); nx = ny = 2^DEPTH
    vw = grid_tv(w, part, nx, ny, dl)
    ε1 = norm(L(w) .- w, 1)
    r_bound = ε1 + (δ / 2) * (vw + varρ)
    @printf("%6d %9d %9.4f %10.5f %12.5f %12.4f\n", DEPTH, n, vw, δ, r_bound, Aabs * r_bound)
    flush(stdout)
end
