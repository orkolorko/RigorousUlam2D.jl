# Lightweight script: compute rigorous error bounds for Hénon and Lorenz examples
# Outputs LaTeX table source with all data

import Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using RigorousUlam2D
using IntervalArithmetic, IntervalBoxes
using LinearAlgebra, SparseArrays
using Arpack
using Printf

# ============================================================
# Example 1: Hénon Map with Periodic BC
# ============================================================

function henon_rescaled(x, y; a=1.4, b=0.3)
    x_phys = 1.5 * x
    y_phys = 0.5 * y
    x_new = (1 - a * x_phys^2 + y_phys) / 1.5
    y_new = (b * x_phys) / 0.5
    return (x_new, y_new)
end

domain_henon = IntervalBox(interval(-1.0, 1.0), interval(-1.0, 1.0))
# Grid-aligned noise: width = domain_width / 2^k
# Use k=2 → ξ = 2.0/4 = 0.5 (half-width σ = 0.25) for finite error bound
noise_henon = uniform_noise_box(2.0/4, 2)

println("="^70)
println("Hénon Map (periodic BC)")
println("="^70)

t_henon = @elapsed begin
    partition_h, dyn_graph_h, topo_graph_h = build_adaptive_partition(
        henon_rescaled, domain_henon, noise_henon,
        max_diameter=0.02,
        initial_depth=2,
        periodic=true,
        verbose=true
    )
end

println("\nPartition: $(partition_h.n_boxes) boxes")
max_diam_h = maximum(box_width(box) for box in partition_h.boxes)
@printf("Max diameter: %.6f\n", max_diam_h)
@printf("Partition time: %.2f s\n", t_henon)

println("\nBuilding transition matrix...")
t_matrix_h = @elapsed begin
    P_det_h, U_det_h = build_transition_matrix(
        henon_rescaled, partition_h,
        transition_graph=dyn_graph_h,
        periodic=true,
        enclosure_method=first_order_enclosure,
        max_iterations=8, ε=1e-4
    )

    N_op_h = build_grid_aligned_noise_operator(
        noise_henon, partition_h,
        topological_graph=topo_graph_h,
        periodic=true
    )

    P_h = N_op_h * P_det_h
    col_sums_h = normalize_columns!(P_h)
end

@printf("Matrix size: %d × %d\n", size(P_h)...)
@printf("Non-zeros: %d\n", nnz(P_h))
@printf("Matrix time: %.2f s\n", t_matrix_h)

# Spectral analysis
println("\nSpectral analysis...")
if size(P_h, 1) <= 500
    λ_h = eigvals(Matrix(P_h))
    λ_sorted_h = sort(abs.(λ_h), rev=true)
else
    λ_vals_h, _ = eigs(P_h, nev=min(10, size(P_h,1)-2), which=:LM)
    λ_sorted_h = sort(abs.(λ_vals_h), rev=true)
end

gap_h = length(λ_sorted_h) >= 2 ? 1.0 - λ_sorted_h[2] : NaN
@printf("Spectral gap: %.6f\n", gap_h)

# Rigorous error bound
println("\nRigorous error bound...")
error_h = rigorous_error_bound(P_h, partition_h, noise_henon, verbose=true)

# Invariant measure
if size(P_h, 1) <= 500
    λ_all_h, v_all_h = eigen(Matrix(P_h))
    idx_h = argmax(abs.(λ_all_h))
    μ_h = abs.(v_all_h[:, idx_h])
    μ_h ./= sum(μ_h)
else
    λ_lead_h, v_lead_h = eigs(P_h, nev=1, which=:LR)
    μ_h = abs.(v_lead_h[:, 1])
    μ_h ./= sum(μ_h)
end
active_h = count(μ_h .> 1e-4)

# ============================================================
# Example 2: Contracting Lorenz Map
# ============================================================

function contracting_lorenz(x, y; s=0.5, r=3.0, ρ=1.3, c0=-0.25, c1=0.25)
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

domain_lorenz = IntervalBox(interval(-0.5, 0.5), interval(-0.5, 0.5))
# Grid-aligned noise: width = domain_width / 2^k
# Use k=1 → ξ = 1.0/2 = 0.5 (half-width σ = 0.25) for finite error bound
noise_lorenz = uniform_noise_box(1.0/2, 2)

println("\n\n" * "="^70)
println("Contracting Lorenz Map")
println("="^70)

t_lorenz = @elapsed begin
    partition_l, dyn_graph_l, topo_graph_l = build_adaptive_partition(
        contracting_lorenz, domain_lorenz, noise_lorenz,
        max_diameter=0.02,
        initial_depth=2,
        safety_margin=1.1,
        verbose=true,
        enclosure_method=zeroth_order_enclosure,
    )
end

println("\nPartition: $(partition_l.n_boxes) boxes")
max_diam_l = maximum(box_width(box) for box in partition_l.boxes)
@printf("Max diameter: %.6f\n", max_diam_l)
@printf("Partition time: %.2f s\n", t_lorenz)

println("\nBuilding transition matrix...")
t_matrix_l = @elapsed begin
    P_det_l, U_det_l = build_transition_matrix(
        contracting_lorenz, partition_l,
        transition_graph=dyn_graph_l,
        enclosure_method=zeroth_order_enclosure,
        max_iterations=12, ε=1e-6
    )

    N_op_l = build_grid_aligned_noise_operator(
        noise_lorenz, partition_l,
        topological_graph=topo_graph_l
    )

    P_l = N_op_l * P_det_l
    col_sums_l = normalize_columns!(P_l)
end

@printf("Matrix size: %d × %d\n", size(P_l)...)
@printf("Non-zeros: %d\n", nnz(P_l))
@printf("Matrix time: %.2f s\n", t_matrix_l)

# Spectral analysis
println("\nSpectral analysis...")
if size(P_l, 1) <= 500
    λ_l = eigvals(Matrix(P_l))
    λ_sorted_l = sort(abs.(λ_l), rev=true)
else
    λ_vals_l, _ = eigs(P_l, nev=min(10, size(P_l,1)-2), which=:LM)
    λ_sorted_l = sort(abs.(λ_vals_l), rev=true)
end

gap_l = length(λ_sorted_l) >= 2 ? 1.0 - λ_sorted_l[2] : NaN
@printf("Spectral gap: %.6f\n", gap_l)

# Rigorous error bound
println("\nRigorous error bound...")
error_l = rigorous_error_bound(P_l, partition_l, noise_lorenz, verbose=true)

# Invariant measure
if size(P_l, 1) <= 500
    λ_all_l, v_all_l = eigen(Matrix(P_l))
    idx_l = argmax(abs.(λ_all_l))
    μ_l = abs.(v_all_l[:, idx_l])
    μ_l ./= sum(μ_l)
else
    λ_lead_l, v_lead_l = eigs(P_l, nev=1, which=:LR)
    μ_l = abs.(v_lead_l[:, 1])
    μ_l ./= sum(μ_l)
end
active_l = count(μ_l .> 1e-4)

# ============================================================
# LaTeX Output
# ============================================================

println("\n\n" * "="^70)
println("LaTeX Source")
println("="^70)

latex = raw"""
\begin{table}[htbp]
\centering
\caption{Rigorous error bounds for invariant density approximation}
\label{tab:error-bounds}
\begin{tabular}{lcc}
\toprule
& \textbf{H\'enon (periodic)} & \textbf{Contracting Lorenz} \\
\midrule
\multicolumn{3}{l}{\emph{System parameters}} \\
Domain & $[-1,1]^2$ & $[-0.5, 0.5]^2$ \\
Boundary conditions & Periodic & Absorbing \\
"""

σ_henon = diam(noise_henon[1]) / 2
σ_lorenz = diam(noise_lorenz[1]) / 2
latex *= @sprintf("Noise \\(\\xi\\) & \$\\pm %.4f\$ & \$\\pm %.4f\$ \\\\\n", σ_henon, σ_lorenz)
latex *= raw"""
\midrule
\multicolumn{3}{l}{\emph{Partition \& matrix}} \\
"""
latex *= @sprintf("Boxes & %d & %d \\\\\n", partition_h.n_boxes, partition_l.n_boxes)
latex *= @sprintf("Max diameter \$\\delta\$ & %.6f & %.6f \\\\\n", max_diam_h, max_diam_l)
latex *= @sprintf("Matrix size & \$%d \\times %d\$ & \$%d \\times %d\$ \\\\\n",
                  size(P_h)..., size(P_l)...)
latex *= @sprintf("Non-zeros & %d & %d \\\\\n", nnz(P_h), nnz(P_l))
latex *= @sprintf("Sparsity & %.1f\\%% & %.1f\\%% \\\\\n",
                  100*(1 - nnz(P_h)/prod(size(P_h))),
                  100*(1 - nnz(P_l)/prod(size(P_l))))

latex *= raw"""
\midrule
\multicolumn{3}{l}{\emph{Spectral analysis}} \\
"""
latex *= @sprintf("Spectral gap \$\\gamma\$ & %.6f & %.6f \\\\\n", gap_h, gap_l)

if length(λ_sorted_h) >= 2
    latex *= @sprintf("\$|\\lambda_2|\$ & %.6f & %.6f \\\\\n", λ_sorted_h[2], λ_sorted_l[2])
end

latex *= raw"""
\midrule
\multicolumn{3}{l}{\emph{Rigorous error bound (Theorem 4.2)}} \\
"""

if error_h.mixing_satisfied
    latex *= @sprintf("Contraction power \$N\$ & %d & ", error_h.N)
else
    latex *= "Contraction power \$N\$ & --- & "
end
if error_l.mixing_satisfied
    latex *= @sprintf("%d \\\\\n", error_l.N)
else
    latex *= "--- \\\\\n"
end

if error_h.mixing_satisfied
    latex *= @sprintf("Contraction rate \$\\alpha\$ & %.6f & ", error_h.α)
else
    latex *= "Contraction rate \$\\alpha\$ & --- & "
end
if error_l.mixing_satisfied
    latex *= @sprintf("%.6f \\\\\n", error_l.α)
else
    latex *= "--- \\\\\n"
end

if error_h.mixing_satisfied
    latex *= @sprintf("\\(\\operatorname{var}(\\rho)\\) & %.4f & ", error_h.var_ρ)
else
    latex *= "\\(\\operatorname{var}(\\rho)\\) & --- & "
end
if error_l.mixing_satisfied
    latex *= @sprintf("%.4f \\\\\n", error_l.var_ρ)
else
    latex *= "--- \\\\\n"
end

latex *= "Mixing condition & "
latex *= (error_h.mixing_satisfied ? "\\checkmark" : "\\texttimes") * " & "
latex *= (error_l.mixing_satisfied ? "\\checkmark" : "\\texttimes") * " \\\\\n"

if error_h.mixing_satisfied
    latex *= @sprintf("\$\\|f_\\xi - f_{\\delta,\\xi}\\|_{L^1}\$ & \$\\leq %.6f\$ & ", error_h.error_bound)
else
    latex *= "\$\\|f_\\xi - f_{\\delta,\\xi}\\|_{L^1}\$ & \$\\infty\$ & "
end
if error_l.mixing_satisfied
    latex *= @sprintf("\$\\leq %.6f\$ \\\\\n", error_l.error_bound)
else
    latex *= "\$\\infty\$ \\\\\n"
end

latex *= raw"""
\midrule
\multicolumn{3}{l}{\emph{Computation time}} \\
"""
latex *= @sprintf("Partition & %.1f s & %.1f s \\\\\n", t_henon, t_lorenz)
latex *= @sprintf("Matrix construction & %.1f s & %.1f s \\\\\n", t_matrix_h, t_matrix_l)

latex *= raw"""
\bottomrule
\end{tabular}
\end{table}
"""

println(latex)

# Also write to file
output_path = joinpath(@__DIR__, "error_bounds_table.tex")
open(output_path, "w") do io
    write(io, latex)
end
println("LaTeX source written to: $output_path")
