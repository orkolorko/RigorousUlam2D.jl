# A-posteriori residual  ‖r‖ = ‖(L_ξ − L_{δ,ξ}) f_{δ,ξ}‖_{L¹}, estimated by applying a
# finer-grid operator (proxy for the true L_ξ) to the depth-6 stationary density w6.
# ‖·‖_{L¹} of a density difference = Σ_boxes |Δmass|.  Noise is consistent between depth
# 6 and 7 (box radius rounds to the same effective half-width 0.09375), so the depth-7
# proxy isolates the transfer-operator refinement.
import Pkg; Pkg.activate(@__DIR__)
include(joinpath(@__DIR__, "ulam_matvec.jl"))
using RigorousUlam2D: build_uniform_partition, build_grid_index_maps, noise_variation,
                      first_order_enclosure, zeroth_order_enclosure
using LinearAlgebra, Printf

function perron(Lmul, n; iters = 500, tol = 1e-14)
    v = fill(1.0 / n, n); local w
    for _ in 1:iters
        w = Lmul(v); s = sum(w); s == 0 && break; w ./= s
        norm(w .- v, 1) < tol && (v = w; break); v = w
    end
    return v
end

# lift a coarse mass vector (depth d0) to a fine grid (depth d1>d0): each coarse box
# splits its mass equally among the (2^(d1-d0))² fine boxes it contains.
function lift_mass(wc, domain, d0, d1)
    nc = 2^d0; nf = 2^d1
    partc = build_uniform_partition(domain, d0); partf = build_uniform_partition(domain, d1)
    b2gc, _ = build_grid_index_maps(partc, (nc, nc))
    b2gf, _ = build_grid_index_maps(partf, (nf, nf))
    Wc = zeros(nc, nc)
    for k in 1:length(wc); ix, iy = b2gc[k]; Wc[iy, ix] = wc[k]; end
    ratio = nf ÷ nc; split = 1.0 / (ratio * ratio)
    wf = zeros(nf * nf)
    for k in 1:(nf * nf)
        ix, iy = b2gf[k]
        icx = (ix - 1) ÷ ratio + 1; icy = (iy - 1) ÷ ratio + 1
        wf[k] = Wc[icy, icx] * split
    end
    return wf
end

DEPTH0 = 6
NOISE = uniform_noise_box(0.1, 2)
FINE = (get(ENV, "FINE", "") == "" ? [7] : parse.(Int, split(ENV["FINE"], ",")))
for (label, F, domain, periodic, enc) in [
    ("Henon", henon_rescaled, IntervalBox(interval(-1.0, 1.0), interval(-1.0, 1.0)),
        true, first_order_enclosure),
]
    L6, _, n6, r6 = build_matvec_cached(F, domain, NOISE, DEPTH0; periodic = periodic, enclosure_method = enc)
    w6 = perron(L6, n6)
    @printf("%s depth %d: n=%d noise-radius=%d  discrete residual |L6 w-w|_1=%.2e\n",
        label, DEPTH0, n6, r6, norm(L6(w6) .- w6, 1)); flush(stdout)
    for d1 in FINE
        Ld, _, nd, rd = build_matvec_cached(F, domain, NOISE, d1; periodic = periodic, enclosure_method = enc)
        wf = lift_mass(w6, domain, DEPTH0, d1)
        r = Ld(wf) .- wf
        @printf("   proxy L_%d (n=%d, noise-radius=%d): ‖r‖_L1 = ‖L_%d(lift w6)-lift w6‖_1 = %.4e\n",
            d1, nd, rd, d1, norm(r, 1)); flush(stdout)
    end
end
