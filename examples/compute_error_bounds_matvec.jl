# Rigorous error bounds using mat-vec oracle (avoids storing dense P = N * P_det)
#
# For large noise ξ, the composed matrix P = N * P_det is very dense (nnz ∝ (ξ/δ)²).
# Instead of forming P explicitly, we use the mat-vec:
#   v → N * (P_det * v)
# where P_det is sparse and N is applied as a 2D convolution.
#
# Run: julia --project=.. --threads=auto examples/compute_error_bounds_matvec.jl

import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using RigorousUlam2D
using IntervalArithmetic, IntervalBoxes
using LinearAlgebra, SparseArrays
using Printf

println("Julia threads: ", Threads.nthreads())
flush(stdout)

# ============================================================
# Contracting Lorenz Map
# ============================================================
function contracting_lorenz(x, y; s=0.5, r=3.0, ρ=1.3, c0=-0.25, c1=0.25)
    ax = abs(x)
    if x isa Interval
        if inf(x) > 0
            return (-ρ * ax^s + 0.5, y * ax^r + c0)
        elseif sup(x) < 0
            return (ρ * ax^s - 0.5, -y * ax^r + c1)
        else
            Tx_pos = -ρ * ax^s + 0.5; Gxy_pos = y * ax^r + c0
            Tx_neg = ρ * ax^s - 0.5; Gxy_neg = -y * ax^r + c1
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

function henon_rescaled(x, y; a=1.4, b=0.3)
    x_phys = 1.5 * x
    y_phys = 0.5 * y
    x_new = (1 - a * x_phys^2 + y_phys) / 1.5
    y_new = (b * x_phys) / 0.5
    return (x_new, y_new)
end

# ============================================================
# Helper: compute error bound using mat-vec oracle
# ============================================================
function compute_error_bound_matvec(F, domain, noise, label;
                                     depth::Int=7,
                                     enclosure_method=zeroth_order_enclosure,
                                     periodic=false,
                                     max_power=30,
                                     safety_margin=1.0)
    ξ = minimum(diam(noise[i]) for i in 1:length(noise))
    σ = ξ / 2
    var_ρ = noise_variation(noise)

    println("\n" * "="^70)
    @printf("%s: ξ = %.4f (σ = %.4f), var(ρ) = %.2f, depth = %d\n", label, ξ, σ, var_ρ, depth)
    println("="^70)
    flush(stdout)

    # Step 1: Build uniform partition at given depth
    println("Building uniform partition (depth=$depth)...")
    t_part = @elapsed begin
        partition = build_uniform_partition(domain, depth)
    end
    n = partition.n_boxes
    nx = 2^depth
    ny = 2^depth
    δ = maximum(box_width(box) for box in partition.boxes)
    @printf("  Partition: %d boxes (%d×%d), δ = %.6f (%.1fs)\n", n, nx, ny, δ, t_part)
    flush(stdout)

    # Step 2: Build transition graph and deterministic matrix
    println("Building adaptive partition for graph...")
    t_graph = @elapsed begin
        _, dyn_graph, topo_graph = build_adaptive_partition(
            F, domain, noise;
            max_diameter=δ * 1.1,  # match the uniform partition
            initial_depth=depth,
            enclosure_method=enclosure_method,
            verbose=false,
            safety_margin=safety_margin,
        )
    end
    @printf("  Graph built (%.1fs)\n", t_graph)
    flush(stdout)

    println("Building deterministic transition matrix...")
    t_det = @elapsed begin
        P_det, _ = build_transition_matrix(
            F, partition,
            transition_graph=dyn_graph,
            enclosure_method=enclosure_method,
            max_iterations=12, ε=1e-6,
            periodic=periodic,
        )
        normalize_columns!(P_det)
    end
    @printf("  P_det: %d × %d, nnz = %d (%.1fs)\n", size(P_det)..., nnz(P_det), t_det)
    flush(stdout)

    # Step 3: Decide strategy based on fill ratio of composed matrix
    noise_cells_x = Int(round(diam(noise[1]) / (diam(domain[1]) / nx)))
    noise_cells_y = Int(round(diam(noise[2]) / (diam(domain[2]) / ny)))
    estimated_fill = noise_cells_x * noise_cells_y / n
    estimated_nnz = Int(round(estimated_fill * n * n))

    @printf("  Noise spans %d×%d grid cells, fill ≈ %.1f%%, estimated nnz ≈ %s\n",
            noise_cells_x, noise_cells_y, estimated_fill * 100,
            estimated_nnz > 1e9 ? @sprintf("%.1fB", estimated_nnz/1e9) :
            estimated_nnz > 1e6 ? @sprintf("%.1fM", estimated_nnz/1e6) :
            string(estimated_nnz))
    flush(stdout)

    use_matvec = estimated_nnz > 2_000_000_000  # >2B nnz → use mat-vec

    if use_matvec
        println("  → Using mat-vec oracle (matrix too large to store)")
        flush(stdout)

        # Build index maps for noise_matvec
        b2g, g2b = build_grid_index_maps(partition, (nx, ny))

        # Build the mat-vec: v → N * (P_det * v)
        # where N is applied as 2D convolution (periodic or absorbing)
        function composed_matvec(v)
            w = P_det * v  # sparse mat-vec (fast)
            return noise_matvec(w, noise, partition, grid_dims=(nx, ny),
                                periodic=periodic,
                                box_to_grid=b2g, grid_to_box=g2b)
        end

        # For Higham bounds, we need ||P||_{ℓ^1} and max_nnz_per_row
        # For composed P = N * P_det, ||P||_{ℓ^1} ≤ ||N||_{ℓ^1} * ||P_det||_{ℓ^1} = 1 * 1 = 1
        # max_nnz_per_row of P ≈ n (fully dense rows due to large noise)
        println("Computing norms of powers (mat-vec, multithreaded)...")
        t_norms = @elapsed begin
            raw_norms = norms_of_powers_matvec(composed_matvec, n, max_power;
                                                nrmM=1.0,
                                                max_nnz_per_row=n,
                                                rigorous=true,
                                                verbose=false)
            refined = refine_norms_of_powers(raw_norms, max_power)
        end
    else
        println("  → Building explicit composed matrix")
        flush(stdout)

        t_noise = @elapsed begin
            N_op = build_grid_aligned_noise_operator(
                noise, partition,
                topological_graph=topo_graph,
                periodic=periodic,
            )
            P = N_op * P_det
            normalize_columns!(P)
        end
        @printf("  P composed: nnz = %d (%.1fs)\n", nnz(P), t_noise)
        flush(stdout)

        println("Computing norms of powers (sparse, multithreaded)...")
        t_norms = @elapsed begin
            raw_norms = norms_of_powers_sparse(P, max_power, rigorous=true, verbose=false)
            refined = refine_norms_of_powers(raw_norms, max_power)
        end
    end

    @printf("  Norms computed in %.1fs\n", t_norms)
    println("  Refined norms:")
    for k in 1:max_power
        @printf("    k=%2d: raw=%.6f  refined=%.6f\n", k, raw_norms[k], refined[k])
    end
    flush(stdout)

    # Step 4: Check mixing condition
    best_bound = Inf
    best_N = 0
    best_α = 0.0
    for k in 1:max_power
        α = refined[k]
        mixing_term = (3 * δ / ξ) * var_ρ * k
        total = α + mixing_term
        if total < 1.0
            bound = (3 * δ * var_ρ) / (ξ * (1 - α)) * k
            if bound < best_bound
                best_bound = bound
                best_N = k
                best_α = α
            end
        end
    end

    if best_bound < Inf
        @printf("\n  ✓ MIXING SATISFIED at N=%d, α=%.6f\n", best_N, best_α)
        @printf("    Rigorous error bound: ||f_ξ - f_{δ,ξ}||_{L¹} ≤ %.6f\n", best_bound)
    else
        # Show what δ would be needed
        for k in 1:max_power
            α = refined[k]
            if α < 1.0
                δ_needed = (1 - α) * ξ / (3 * var_ρ * k)
                n_needed = Int(ceil(1 / δ_needed))^2
                @printf("\n  ✗ Mixing NOT satisfied. Best at k=%d: α=%.4f\n", k, α)
                @printf("    Need δ < %.6f (n ≈ %d boxes)\n", δ_needed, n_needed)
                break
            end
        end
    end
    flush(stdout)

    return (n=n, δ=δ, ξ=ξ, var_ρ=var_ρ, raw_norms=raw_norms, refined=refined,
            best_bound=best_bound, best_N=best_N, best_α=best_α,
            t_part=t_part, t_det=t_det, t_norms=t_norms)
end

# ============================================================
# Run experiments
# ============================================================
results = Dict{String, Any}()

domain_lorenz = IntervalBox(interval(-0.5, 0.5), interval(-0.5, 0.5))
domain_henon = IntervalBox(interval(-1.0, 1.0), interval(-1.0, 1.0))

# Lorenz with ξ=0.5 at increasing grid depth
# Mixing requires δ < ~0.001, so depth=10 (δ≈0.001) is the target
# But start with smaller depths to see the trend
for depth in [7, 8, 9]
    ξ = 0.5
    σ = ξ / 2
    noise = uniform_noise_box(σ, 2)
    label = "Lorenz ξ=0.5 d=$depth"
    result = compute_error_bound_matvec(
        contracting_lorenz, domain_lorenz, noise, label;
        depth=depth,
        enclosure_method=zeroth_order_enclosure,
        safety_margin=1.1,
        max_power=30,
    )
    results[label] = result
end

# Hénon with ξ=0.5, periodic BC
for depth in [7, 8]
    ξ = 0.5
    σ = ξ / 2
    noise = uniform_noise_box(σ, 2)
    label = "Hénon ξ=0.5 d=$depth"
    result = compute_error_bound_matvec(
        henon_rescaled, domain_henon, noise, label;
        depth=depth,
        enclosure_method=zeroth_order_enclosure,
        periodic=true,
        max_power=30,
    )
    results[label] = result
end

# ============================================================
# Summary
# ============================================================
println("\n\n" * "="^70)
println("SUMMARY")
println("="^70)
@printf("%-25s %7s %10s %6s %8s %6s %6s %10s %8s\n",
        "Config", "n", "δ", "ξ", "var(ρ)", "N", "α", "bound", "t_norm")
println("-"^80)
for label in sort(collect(keys(results)))
    r = results[label]
    if r.best_bound < Inf
        @printf("%-25s %7d %10.6f %6.4f %8.2f %6d %6.4f %10.6f %7.1fs\n",
                label, r.n, r.δ, r.ξ, r.var_ρ, r.best_N, r.best_α, r.best_bound, r.t_norms)
    else
        @printf("%-25s %7d %10.6f %6.4f %8.2f %6s %6s %10s %7.1fs\n",
                label, r.n, r.δ, r.ξ, r.var_ρ, "---", "---", "∞", r.t_norms)
    end
end
flush(stdout)
