# Rigorous error bounds computation script for ibis (32 threads)
# Run: julia --project=.. --threads=auto examples/compute_error_bounds_ibis.jl
#
# Tries multiple grid-aligned noise levels for the Contracting Lorenz map
# and the Hénon map (periodic BC), computing rigorous error bounds.

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
# Helper: run full pipeline for one configuration
# ============================================================
function compute_error_bound(F, domain, noise, label;
                              max_diameter=0.02,
                              initial_depth=2,
                              enclosure_method=zeroth_order_enclosure,
                              periodic=false,
                              max_power=30,
                              safety_margin=1.0)
    ξ = minimum(diam(noise[i]) for i in 1:length(noise))
    σ = ξ / 2
    var_ρ = noise_var12(noise)

    println("\n" * "="^70)
    @printf("%s: ξ = %.4f (σ = %.4f), var(ρ) = %.2f\n", label, ξ, σ, var_ρ)
    @printf("  max_diameter = %.6f, periodic = %s\n", max_diameter, periodic)
    println("="^70)
    flush(stdout)

    # Step 1: Build partition
    println("Building adaptive partition...")
    t_part = @elapsed begin
        kw = Dict{Symbol,Any}(
            :max_diameter => max_diameter,
            :initial_depth => initial_depth,
            :enclosure_method => enclosure_method,
            :verbose => false,
        )
        if periodic
            kw[:periodic] = true
        end
        if safety_margin != 1.0
            kw[:safety_margin] = safety_margin
        end
        partition, dyn_graph, topo_graph = build_adaptive_partition(F, domain, noise; kw...)
    end
    n = partition.n_boxes
    δ = maximum(box_width(box) for box in partition.boxes)
    @printf("  Partition: %d boxes, δ = %.6f (%.1fs)\n", n, δ, t_part)
    flush(stdout)

    # Step 2: Build transition matrix
    println("Building transition matrix...")
    t_mat = @elapsed begin
        P_det, _ = build_transition_matrix(
            F, partition,
            transition_graph=dyn_graph,
            enclosure_method=enclosure_method,
            max_iterations=12, ε=1e-6,
            periodic=periodic,
        )
        N_op = build_grid_aligned_noise_operator(
            noise, partition,
            topological_graph=topo_graph,
            periodic=periodic,
        )
        P = N_op * P_det
        normalize_columns!(P)
    end
    @printf("  Matrix: %d × %d, nnz = %d (%.1fs)\n", size(P)..., nnz(P), t_mat)
    flush(stdout)

    # Step 3: Compute norms of powers
    println("Computing norms of powers (sparse, multithreaded)...")
    t_norms = @elapsed begin
        raw_norms = norms_of_powers_sparse(P, max_power, rigorous=true, verbose=false)
        refined = refine_norms_of_powers(raw_norms, max_power)
    end
    @printf("  Norms computed in %.1fs\n", t_norms)

    println("  Refined norms:")
    for k in 1:max_power
        @printf("    k=%2d: raw=%.6f  refined=%.6f\n", k, raw_norms[k], refined[k])
    end
    flush(stdout)

    # Step 4: Check mixing condition for each power
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
        best_total = minimum(refined[k] + (3*δ/ξ)*var_ρ*k for k in 1:max_power)
        @printf("\n  ✗ Mixing NOT satisfied (best total = %.4f)\n", best_total)
        @printf("    Need δ < %.6f for this ξ\n", (1 - minimum(refined)) * ξ / (3 * var_ρ * argmin(refined)))
    end
    flush(stdout)

    return (n=n, δ=δ, ξ=ξ, var_ρ=var_ρ, raw_norms=raw_norms, refined=refined,
            best_bound=best_bound, best_N=best_N, best_α=best_α,
            t_part=t_part, t_mat=t_mat, t_norms=t_norms, nnz_P=nnz(P))
end

# ============================================================
# Run experiments
# ============================================================
results = Dict{String, Any}()

domain_lorenz = IntervalBox(interval(-0.5, 0.5), interval(-0.5, 0.5))
domain_henon = IntervalBox(interval(-1.0, 1.0), interval(-1.0, 1.0))

# Lorenz: try multiple noise levels with appropriate δ
# Grid-aligned: ξ = 1.0/2^k for domain width 1.0
lorenz_configs = [
    # (ξ, max_diameter, label)
    (0.5,    0.01,  "Lorenz ξ=0.5"),      # var(ρ)=8, need 48δN<1-α
    (0.25,   0.005, "Lorenz ξ=0.25"),     # var(ρ)=16, need 192δN<1-α
    (0.125,  0.002, "Lorenz ξ=0.125"),    # var(ρ)=32, need 768δN<1-α
]

for (ξ, max_diam, label) in lorenz_configs
    σ = ξ / 2
    noise = uniform_noise_box(σ, 2)
    result = compute_error_bound(
        contracting_lorenz, domain_lorenz, noise, label;
        max_diameter=max_diam,
        initial_depth=2,
        enclosure_method=zeroth_order_enclosure,
        safety_margin=1.1,
        max_power=30,
    )
    results[label] = result
end

# Hénon: try with periodic BC
# Grid-aligned: ξ = 2.0/2^k for domain width 2.0
henon_configs = [
    (0.5,    0.01,  "Hénon ξ=0.5"),       # var(ρ)=8
    (0.25,   0.005, "Hénon ξ=0.25"),      # var(ρ)=16
]

for (ξ, max_diam, label) in henon_configs
    σ = ξ / 2
    noise = uniform_noise_box(σ, 2)
    result = compute_error_bound(
        henon_rescaled, domain_henon, noise, label;
        max_diameter=max_diam,
        initial_depth=2,
        enclosure_method=zeroth_order_enclosure,
        periodic=true,
        max_power=30,
    )
    results[label] = result
end

# ============================================================
# Summary Table
# ============================================================
println("\n\n" * "="^70)
println("SUMMARY")
println("="^70)
@printf("%-20s %7s %10s %6s %8s %6s %6s %10s\n",
        "Config", "n", "δ", "ξ", "var(ρ)", "N", "α", "bound")
println("-"^70)
for label in sort(collect(keys(results)))
    r = results[label]
    if r.best_bound < Inf
        @printf("%-20s %7d %10.6f %6.4f %8.2f %6d %6.4f %10.6f\n",
                label, r.n, r.δ, r.ξ, r.var_ρ, r.best_N, r.best_α, r.best_bound)
    else
        @printf("%-20s %7d %10.6f %6.4f %8.2f %6s %6s %10s\n",
                label, r.n, r.δ, r.ξ, r.var_ρ, "---", "---", "∞")
    end
end
flush(stdout)

# ============================================================
# LaTeX Table
# ============================================================
println("\n\n" * "="^70)
println("LaTeX Source")
println("="^70)

latex = raw"""
\begin{table}[htbp]
\centering
\caption{Rigorous error bounds for invariant density approximation}
\label{tab:error-bounds}
\begin{tabular}{lccccc}
\toprule
& $\xi$ & $n$ & $\delta$ & $\alpha$ (at $N$) & $\|f_\xi - f_{\delta,\xi}\|_{L^1}$ \\
\midrule
"""

for label in sort(collect(keys(results)))
    r = results[label]
    if r.best_bound < Inf
        line = @sprintf("%s & %.4f & %d & %.6f & %.4f (N=%d) & ", label, r.ξ, r.n, r.δ, r.best_α, r.best_N)
        line *= @sprintf("\$\\leq %.6f\$ \\\\\n", r.best_bound)
        latex *= line
    else
        line = @sprintf("%s & %.4f & %d & %.6f & --- & \$\\infty\$ \\\\\n", label, r.ξ, r.n, r.δ)
        latex *= line
    end
end

latex *= raw"""
\bottomrule
\end{tabular}
\end{table}
"""
println(latex)

output_path = joinpath(@__DIR__, "error_bounds_ibis.tex")
open(output_path, "w") do io
    write(io, latex)
end
println("LaTeX written to: $output_path")
flush(stdout)
