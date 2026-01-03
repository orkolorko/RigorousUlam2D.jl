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
using Printf

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
noise = uniform_noise_box(0.025, 2)  # Noise: ±0.025

println("\nSystem:")
println("  Hénon map (a=1.4, b=0.3), rescaled to [-1,1]²")
println("  Domain: [-1,1]²")
println("  Noise: ±0.025 uniform")
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

println("\nBuilding noise convolution operator...")
N = build_noise_convolution_operator(
    noise, partition,
    transition_graph=topo_graph,
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
    λ = eigvals(Matrix(P))
    λ_sorted = sort(abs.(λ), rev=true)

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
else
    println("  Partition too large for dense eigenvalue computation")
end

println("\n" * "="^70)
println("Example completed successfully!")
println("="^70)
