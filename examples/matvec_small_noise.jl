# Hénon & contracting-Lorenz with SMALL noise via the matrix-free (mat-vec) operator.
#
# Unlike compute_error_bounds_matvec.jl (which only switches to the mat-vec oracle
# when the composed matrix would be too dense to store, i.e. LARGE noise), this
# script ALWAYS uses the mat-vec oracle
#
#     v  ↦  N * (P_det * v)          # P_det sparse; N a rigorous 2-D convolution
#
# and runs it in the SMALL-noise regime. With small ξ the composed P is sparse, so
# storing it would be cheap — we use the oracle anyway, because the point here is
# the matrix-free operator (it is what couples to the spectral / pseudospectral
# machinery and what scales to finer grids where P would not fit).
#
# Rigour of the resulting bounds is a separate discussion; this is the harness.
#
#   julia --project=examples --threads=auto examples/matvec_small_noise.jl

import Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using RigorousUlam2D
using IntervalArithmetic, IntervalBoxes
using LinearAlgebra, SparseArrays
using Printf

println("Julia threads: ", Threads.nthreads())
flush(stdout)

# ============================================================
# Maps
# ============================================================
function henon_rescaled(x, y; a = 1.4, b = 0.3)
    x_phys = 1.5 * x
    y_phys = 0.5 * y
    x_new = (1 - a * x_phys^2 + y_phys) / 1.5
    y_new = (b * x_phys) / 0.5
    return (x_new, y_new)
end

function contracting_lorenz(x, y; s = 0.5, r = 3.0, ρ = 1.3, c0 = -0.25, c1 = 0.25)
    ax = abs(x)
    if x isa Interval
        if inf(x) > 0
            return (-ρ * ax^s + 0.5, y * ax^r + c0)
        elseif sup(x) < 0
            return (ρ * ax^s - 0.5, -y * ax^r + c1)
        else
            Tx_pos = -ρ * ax^s + 0.5
            Gxy_pos = y * ax^r + c0
            Tx_neg = ρ * ax^s - 0.5
            Gxy_neg = -y * ax^r + c1
            return (RigorousUlam2D.hull(Tx_pos, Tx_neg), RigorousUlam2D.hull(Gxy_pos, Gxy_neg))
        end
    else
        x_val = x isa Real ? x : ForwardDiff.value(x)
        if x_val > 0
            return (-ρ * ax^s + 0.5, y * ax^r + c0)
        else
            return (ρ * ax^s - 0.5, -y * ax^r + c1)
        end
    end
end

# ============================================================
# Build the matrix-free composed operator on a UNIFORM grid
# ============================================================
# Returns (matvec, n, partition, δ, ξ, var_ρ, nx, ny, P_det).
# `matvec` is the rigorous mat-vec oracle  v ↦ N * (P_det * v).
function build_matvec_operator(F, domain, noise, label;
        depth::Int = 7,
        enclosure_method = zeroth_order_enclosure,
        periodic::Bool = false,
        safety_margin = 1.1)
    ξ = minimum(diam(noise[i]) for i in 1:length(noise))
    σ = ξ / 2
    var_ρ = noise_var12(noise)

    println("\n" * "="^70)
    @printf("%s  —  ξ = %.5f (σ = %.5f), var(ρ) = %.2f, depth = %d\n",
        label, ξ, σ, var_ρ, depth)
    println("="^70)
    flush(stdout)

    # Full uniform nx×ny grid (every grid cell backed by a box — the convolution needs it).
    partition = build_uniform_partition(domain, depth)
    n = partition.n_boxes
    nx = 2^depth
    ny = 2^depth
    δ = maximum(box_width(box) for box in partition.boxes)
    @printf("  Uniform partition: %d boxes (%d×%d), δ = %.6f\n", n, nx, ny, δ)
    n == nx * ny || error("unexpected box count $n (≠ $(nx*ny))")

    # Sanity: small noise must still align to the grid (σ = rx·dx, rx ≥ 1)
    dx = diam(domain[1]) / nx
    rx = Int(round(σ / dx))
    @printf("  Noise half-width σ = %.5f spans rx = %d grid cell(s) (dx = %.5f)\n", σ, rx, dx)
    rx < 1 && error("noise too small for this grid: σ < dx ⇒ rx = 0. Increase ξ or depth.")
    flush(stdout)

    # Deterministic transfer by FORWARD-IMAGE SCATTER — no graph, no all-pairs scan.
    # P_det is sparse; the dense composed P and the noise operator N are never formed.
    println("  Building P_det by forward-image scatter (no graph)...")
    t_det = @elapsed begin
        P_det = build_transfer_forward(F, partition, (nx, ny);
            subdiv = 4, enclosure_method = enclosure_method, periodic = periodic)
        normalize_columns!(P_det)
    end
    @printf("  P_det: %d×%d, nnz = %d (%.1f/col, %.1fs)\n",
        size(P_det)..., nnz(P_det), nnz(P_det) / n, t_det)
    flush(stdout)

    # Composed matvec  v ↦ N(P_det v)  applied on the fly.
    matvec = transfer_operator_matvec(P_det, noise, partition, (nx, ny); periodic = periodic)

    return (; matvec, n, partition, δ, ξ, var_ρ, nx, ny, P_det)
end

# ============================================================
# Analyse one operator: norms of powers (matrix-free) + error bound
# ============================================================
function analyse(op, label; max_power::Int = 30)
    (; matvec, n, δ, ξ, var_ρ, P_det) = op

    # max nonzeros per row of the composed P (for the rigorous Higham padding)
    max_nnz_row = maximum(diff(P_det.colptr))     # P_det is column-stored; bound it

    println("  Computing norms of powers via mat-vec (rigorous, multithreaded)...")
    t = @elapsed begin
        raw = norms_of_powers_matvec(matvec, n, max_power;
            nrmM = 1.0, max_nnz_per_row = n,
            rigorous = true, verbose = false)
        refined = refine_norms_of_powers(raw, max_power)
    end
    @printf("  done (%.1fs).  ‖P^k|_V‖ (refined):\n", t)
    for k in 1:max_power
        @printf("    k=%2d  raw=%.6f  refined=%.6f\n", k, raw[k], refined[k])
    end

    # Mixing / error bound (same criterion as compute_error_bounds_matvec.jl)
    best_bound, best_N, best_α = Inf, 0, 0.0
    for k in 1:max_power
        α = refined[k]
        if α + (3 * δ / ξ) * var_ρ * k < 1.0
            bound = (3 * δ * var_ρ) / (ξ * (1 - α)) * k
            bound < best_bound && ((best_bound, best_N, best_α) = (bound, k, α))
        end
    end
    if best_bound < Inf
        @printf("  ✓ mixing at N=%d, α=%.6f → ‖f_ξ − f_{δ,ξ}‖_{L¹} ≤ %.6f\n",
            best_N, best_α, best_bound)
    else
        @printf("  ✗ mixing NOT satisfied up to k=%d (small noise ⇒ need finer δ)\n", max_power)
    end
    flush(stdout)
    return (; raw, refined, best_bound, best_N, best_α, label)
end

# ============================================================
# Run: SMALL noise, both maps, matrix-free
# ============================================================
domain_henon = IntervalBox(interval(-1.0, 1.0), interval(-1.0, 1.0))
domain_lorenz = IntervalBox(interval(-0.5, 0.5), interval(-0.5, 0.5))

# Small grid-aligned noise (the values used in the visual examples 02 / 05):
#   Hénon:  ξ = 2.0/64 = 0.03125   on width-2 domain
#   Lorenz: ξ = 1.0/16 = 0.0625    on width-1 domain
DEPTH = get(ENV, "DEPTH", "") == "" ? 7 : parse(Int, ENV["DEPTH"])

results = Any[]

op_h = build_matvec_operator(
    henon_rescaled, domain_henon, uniform_noise_box(2.0 / 64, 2), "Hénon (periodic)";
    depth = DEPTH, enclosure_method = first_order_enclosure, periodic = true)
push!(results, analyse(op_h, "Hénon"))

op_l = build_matvec_operator(
    contracting_lorenz, domain_lorenz, uniform_noise_box(1.0 / 16, 2), "Contracting Lorenz";
    depth = DEPTH, enclosure_method = zeroth_order_enclosure, periodic = false)
push!(results, analyse(op_l, "Lorenz"))

# ============================================================
# Summary
# ============================================================
println("\n" * "="^70)
println("SUMMARY (matrix-free, small noise, depth = $DEPTH)")
println("="^70)
@printf("%-16s %10s %6s %10s\n", "map", "N", "α", "L¹ bound")
println("-"^46)
for r in results
    if r.best_bound < Inf
        @printf("%-16s %10d %6.4f %10.6f\n", r.label, r.best_N, r.best_α, r.best_bound)
    else
        @printf("%-16s %10s %6s %10s\n", r.label, "---", "---", "∞")
    end
end
