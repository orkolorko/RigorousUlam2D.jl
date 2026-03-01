# Activate example environment
import Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()  # Install dependencies if needed

# Example 1: Simple Contracting Map
#
# This example demonstrates the basic workflow for a contracting map:
# 1. Define the dynamical system
# 2. Build adaptive partition with diameter-based stopping
# 3. Construct transition matrix using graph optimization
# 4. Analyze the results

using RigorousUlam2D
using IntervalArithmetic, IntervalBoxes
using LinearAlgebra, SparseArrays
using Arpack
using Printf
using CairoMakie

println("="^70)
println("Example 1: Simple Contracting Map")
println("="^70)

# Define a simple contracting map
F(x, y) = (0.5*x + 0.25, 0.5*y + 0.25)

# Setup domain
domain = IntervalBox(interval(0.0, 1.0), interval(0.0, 1.0))
# Grid-aligned noise: width = domain_width / 2^k (k=6)
# For domain width 1.0: 1.0/64 = 0.015625
noise = uniform_noise_box(1.0/64, 2)

println("\nSystem:")
println("  F(x,y) = (0.5x + 0.25, 0.5y + 0.25)")
println("  Domain: [0,1]²")
println("  Noise: ±$(1.0/64) uniform (grid-aligned)")

# Build adaptive partition with diameter-based stopping
println("\nBuilding adaptive partition...")
partition, dyn_graph, topo_graph = build_adaptive_partition(
    F, domain, noise,
    initial_depth=2,
    max_diameter=0.03,
    safety_margin=1.1,
    verbose=true
)

# Verify achieved diameter
max_diam = maximum(box_width(box) for box in partition.boxes)
println("\nPartition statistics:")
println("  Target diameter: 0.03")
@printf("  Achieved diameter: %.6f\n", max_diam)
println("  Number of boxes: $(partition.n_boxes)")
println("  Graph non-zeros: $(nnz(dyn_graph))")
@printf("  Graph sparsity: %.1f%%\n",
        100*(1 - nnz(dyn_graph)/partition.n_boxes^2))

# Build deterministic transition matrix (without noise)
println("\nBuilding deterministic transition matrix P...")
P, U = build_transition_matrix(
    F, partition,  # No noise parameter!
    transition_graph=dyn_graph,
    max_iterations=8,
    ε=1e-4
)

# Build grid-aligned noise operator
println("\nBuilding grid-aligned noise operator N...")
N = build_grid_aligned_noise_operator(
    noise, partition,
    topological_graph=topo_graph
)

# Compose to get noisy transition matrix
println("\nComposing P_noisy = N * P...")
P = N * P

# Normalize to stochastic matrix
col_sums_before = normalize_columns!(P)

# Compute column sums after normalization
col_sums_after = [sum(P[:, j]) for j in 1:size(P, 2)]

println("\nTransition matrix statistics:")
@printf("  Size: %d × %d\n", size(P)...)
@printf("  Non-zeros: %d\n", nnz(P))
stats = estimate_sparsity(P)
@printf("  Sparsity: %.2f%%\n", 100*stats.sparsity)
@printf("  Column sum range (normalized): [%.6f, %.6f]\n", minimum(col_sums_after), maximum(col_sums_after))

# Find invariant measure (leading eigenvector)
println("\nComputing invariant measure...")
if size(P, 1) <= 500
    # Use dense eigendecomposition for small matrices
    λ, v = eigen(Matrix(P))
    idx = argmax(abs.(λ))
    μ = abs.(v[:, idx])
    μ ./= sum(μ)

    println("  Leading eigenvalue: ", round(λ[idx], digits=6))
    println("  Invariant measure computed successfully")
    println("  Active boxes (>1e-4): ", count(μ .> 1e-4))
else
    # Use Arpack for large matrices (compute only leading eigenvector)
    println("  Matrix size $(size(P, 1)) - using sparse Arnoldi method...")
    λ, v = eigs(P, nev=1, which=:LR)
    μ = abs.(v[:, 1])
    μ ./= sum(μ)

    println("  Leading eigenvalue: ", round(λ[1], digits=6))
    println("  Invariant measure computed successfully")
    println("  Active boxes (>1e-4): ", count(μ .> 1e-4))
end

# Visualization
println("\nCreating visualization...")
densities = [μ[i] / box_measure(partition.boxes[i]) for i in 1:partition.n_boxes]

fig = Figure(size=(800, 700))
ax = Axis(fig[1, 1],
    xlabel="x", ylabel="y",
    title="Invariant Density - Contracting Map",
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
output_file = joinpath(@__DIR__, "01_contracting_map_output.png")
save(output_file, fig)
println("Saved visualization to ", output_file)

println("\n" * "="^70)
println("Example completed successfully!")
println("="^70)
