# ACTUAL residual ‖(L_ξ − P_h)f_h‖ for periodic Lorenz a=0.3, vs the a-priori bound
# (δ/2)(var(f_h)+var(ρ)). L_ξ approximated by the operator one level finer (P_{h+1}); the
# residual is ‖P_{h+1}(lift f_h) - lift f_h‖ in both L¹ and L² (density norms).
import Pkg; Pkg.activate(@__DIR__)
include(joinpath(@__DIR__, "ulam_matvec.jl"))
using RigorousUlam2D: build_uniform_partition, build_grid_index_maps, noise_variation,
                      zeroth_order_enclosure
using LinearAlgebra, Printf

function perron(Lmul, n; iters = 500, tol = 1e-14)
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
function lift_mass(wc, domain, d0, d1)
    nc = 2^d0; nf = 2^d1
    partc = build_uniform_partition(domain, d0); partf = build_uniform_partition(domain, d1)
    b2gc, _ = build_grid_index_maps(partc, (nc, nc)); b2gf, _ = build_grid_index_maps(partf, (nf, nf))
    Wc = zeros(nc, nc); for k in 1:length(wc); ix, iy = b2gc[k]; Wc[iy, ix] = wc[k]; end
    ratio = nf ÷ nc; split = 1.0 / ratio^2; wf = zeros(nf * nf)
    for k in 1:(nf * nf); ix, iy = b2gf[k]; wf[k] = Wc[(iy-1)÷ratio+1, (ix-1)÷ratio+1] * split; end
    return wf
end

A = 0.3; NOISE = uniform_noise_box(A, 2); varρ = noise_variation(NOISE)
dl = IntervalBox(interval(-0.5, 0.5), interval(-0.5, 0.5))
DEPTHS = get(ENV, "DEPTHS", "") == "" ? [6, 8] : parse.(Int, split(ENV["DEPTHS"], ","))
@printf("periodic Lorenz a=%.1f var(ρ)=%.2f\n", A, varρ)
@printf("%4s %9s %9s %11s %11s %11s %11s\n", "h", "var(f_h)", "ε₁", "apriori‖r‖", "actual‖r‖L1", "actual‖r‖L2", "ratio act/apri")
for h in DEPTHS
    δ = 1.0 / 2^h
    L, _, n, _ = build_matvec_cached(contracting_lorenz, dl, NOISE, h; periodic = true, enclosure_method = zeroth_order_enclosure)
    w = perron(L, n)
    part = build_uniform_partition(dl, h); nx = ny = 2^h
    vw = grid_tv(w, part, nx, ny, dl); ε1 = norm(L(w) .- w, 1)
    apri = ε1 + (δ / 2) * (vw + varρ)
    # actual residual via one-finer operator (proxy for L_ξ)
    Lf, _, nf, _ = build_matvec_cached(contracting_lorenz, dl, NOISE, h + 1; periodic = true, enclosure_method = zeroth_order_enclosure)
    wf = lift_mass(w, dl, h, h + 1)
    r = Lf(wf) .- wf
    rL1 = norm(r, 1)
    mf = 1.0 / nf                          # box area on fine grid (|M|=1)
    rL2 = sqrt(sum(abs2, r) / mf)          # ‖r‖_{L²} = sqrt(Σ mass²/area)
    @printf("%4d %9.4f %9.1e %11.5f %11.5f %11.5f %11.3f\n", h, vw, ε1, apri, rL1, rL2, rL1 / apri)
    flush(stdout)
end
