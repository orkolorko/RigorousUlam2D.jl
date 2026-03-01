# Activate example environment
import Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()  # Install dependencies if needed

# Example 2: Hénon Map with Periodic Boundary Conditions
#
# This example demonstrates:
# 1. Periodic boundary conditions for torus topology
# 2. Handling of a chaotic map (Hénon)
# 3. Spectral analysis

using RigorousUlam2D
using IntervalArithmetic, IntervalBoxes
using LinearAlgebra, SparseArrays
using Arpack
using Printf
using CairoMakie

println("="^70)
println("Example 2: Hénon Map with Periodic BC")
println("="^70)

# Rescaled Hénon map to fit in [-1,1]²
function henon_rescaled(x, y; a=1.4, b=0.3)
    x_phys = 1.5 * x
    y_phys = 0.5 * y
    x_new = (1 - a * x_phys^2 + y_phys) / 1.5
    y_new = (b * x_phys) / 0.5
    return (x_new, y_new)
end

# Setup
domain = IntervalBox(interval(-1.0, 1.0), interval(-1.0, 1.0))
# Grid-aligned noise: width = domain_width / 2^k (k=6)
# For domain width 2.0: 2.0/64 = 0.03125
noise = uniform_noise_box(2.0/64, 2)

println("\nSystem:")
println("  Hénon map (a=1.4, b=0.3), rescaled to [-1,1]²")
println("  Domain: [-1,1]²")
println("  Noise: ±$(2.0/64) uniform (grid-aligned)")
println("  Boundary conditions: Periodic (torus)")

# Build adaptive partition with periodic BC
println("\nBuilding adaptive partition with periodic BC...")
partition, dyn_graph, topo_graph = build_adaptive_partition(
    henon_rescaled, domain, noise,
    max_diameter=0.02,
    initial_depth=2,
    periodic=true,  # Enable periodic boundary conditions
    verbose=true
)

max_diam = maximum(box_width(box) for box in partition.boxes)
println("\nPartition statistics:")
@printf("  Achieved diameter: %.6f\n", max_diam)
println("  Number of boxes: $(partition.n_boxes)")
@printf("  Graph sparsity: %.1f%%\n",
        100*(1 - nnz(dyn_graph)/partition.n_boxes^2))

# Build transition matrix with periodic BC using operator composition
println("\nBuilding deterministic transition matrix with periodic BC...")
P_det, U_det = build_transition_matrix(
    henon_rescaled, partition,
    transition_graph=dyn_graph,
    periodic=true,  # Enable periodic BC
    enclosure_method=first_order_enclosure,
    max_iterations=8,
    ε=1e-4
)

println("\nBuilding grid-aligned noise operator...")
N = build_grid_aligned_noise_operator(
    noise, partition,
    topological_graph=topo_graph,
    periodic=true
)

println("\nComposing operators: P = N * P_det")
P = N * P_det

# Normalize
col_sums = normalize_columns!(P)

println("\nTransition matrix statistics:")
@printf("  Size: %d × %d\n", size(P)...)
@printf("  Non-zeros: %d\n", nnz(P))
stats = estimate_sparsity(P)
@printf("  Sparsity: %.2f%%\n", 100*stats.sparsity)

# Spectral analysis
println("\nSpectral analysis...")
if size(P, 1) <= 500
    # Use dense eigendecomposition for small matrices
    λ = eigvals(Matrix(P))
    λ_sorted = sort(abs.(λ), rev=true)
else
    # Use Arpack for large matrices (compute leading eigenvalues)
    println("  Matrix size $(size(P, 1)) - using sparse Arnoldi method...")
    λ_vals, _ = eigs(P, nev=min(10, size(P,1)-2), which=:LM)
    λ_sorted = sort(abs.(λ_vals), rev=true)
end

println("  Leading eigenvalues:")
for i in 1:min(5, length(λ_sorted))
    @printf("    λ_%d = %.6f\n", i, λ_sorted[i])
end

if length(λ_sorted) >= 2
    gap = 1.0 - λ_sorted[2]
    rate = -log(λ_sorted[2])
    @printf("\n  Spectral gap: %.6f\n", gap)
    @printf("  Mixing rate: %.6f\n", rate)
    @printf("  Mixing time: ~%.1f iterations\n", 1/rate)
end

# Rigorous error bound
println("\nRigorous error bound (Theorem 4.2, ZZ paper)...")
error_result = rigorous_error_bound(P, partition, noise, verbose=true)
if error_result.mixing_satisfied
    @printf("\n  ✓ Rigorous L¹ error bound: %.6f\n", error_result.error_bound)
    @printf("    Contraction power N = %d, rate α = %.6f\n", error_result.N, error_result.α)
    @printf("    Mesh size δ = %.6f, noise scale ξ = %.6f\n", error_result.δ, error_result.ξ)
else
    println("\n  ✗ Mixing condition not satisfied — bound is infinite")
    println("    Try increasing noise or refining the partition")
end

# Compute invariant measure
println("\nComputing invariant measure...")
if size(P, 1) <= 500
    λ_all, v_all = eigen(Matrix(P))
    idx = argmax(abs.(λ_all))
    μ = abs.(v_all[:, idx])
    μ ./= sum(μ)
else
    println("  Matrix size $(size(P, 1)) - using sparse Arnoldi method...")
    λ_lead, v_lead = eigs(P, nev=1, which=:LR)
    μ = abs.(v_lead[:, 1])
    μ ./= sum(μ)
end
println("  Active boxes (>1e-4): ", count(μ .> 1e-4))

# Visualization
println("\nCreating visualization...")
densities = [μ[i] / box_measure(partition.boxes[i]) for i in 1:partition.n_boxes]

fig = Figure(size=(800, 700))
ax = Axis(fig[1, 1],
    xlabel="x", ylabel="y",
    title="Invariant Density - Hénon Map (Periodic BC)",
    aspect=1
)

# Plot all boxes as colored rectangles (vectorized for performance)
rects = [Rect2f(inf(box[1]), inf(box[2]),
                sup(box[1]) - inf(box[1]), sup(box[2]) - inf(box[2]))
         for box in partition.boxes]
poly!(ax, rects, color=densities, colormap=:viridis,
      colorrange=(0, maximum(densities)))

Colorbar(fig[1, 2], colormap=:viridis,
         limits=(0, maximum(densities)), label="Density ρ(x,y)")
output_file = joinpath(@__DIR__, "02_henon_periodic_output.png")
save(output_file, fig)
println("Saved visualization to ", output_file)

println("\n" * "="^70)
println("Example completed successfully!")
println("="^70)
