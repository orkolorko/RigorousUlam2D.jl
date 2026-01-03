# Activate example environment
import Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()  # Install dependencies if needed

# Example 3: Operator Decomposition
#
# This example demonstrates operator decomposition: P_noisy = N * P_deterministic
# This allows:
# 1. Computing deterministic and noise operators separately
# 2. Reusing deterministic operator for different noise levels
# 3. Understanding how noise affects the dynamics

using RigorousUlam2D
using IntervalArithmetic, IntervalBoxes
using LinearAlgebra, SparseArrays
using Printf

println("="^70)
println("Example 3: Operator Decomposition")
println("="^70)

# Simple contracting map
F(x, y) = (0.5*x + 0.25, 0.5*y + 0.25)
domain = IntervalBox(interval(0.0, 1.0), interval(0.0, 1.0))
noise = uniform_noise_box(0.02, 2)

println("\nSystem:")
println("  F(x,y) = (0.5x + 0.25, 0.5y + 0.25)")
println("  Noise: ±0.02 uniform")

# Build adaptive partition
println("\nBuilding adaptive partition...")
partition, dyn_graph, topo_graph = build_adaptive_partition(
    F, domain, noise,
    max_diameter=0.03,
    initial_depth=2,
    verbose=false
)

println("  Boxes: $(partition.n_boxes)")
println("  Graph non-zeros: $(nnz(dyn_graph))")

# Method 1: Direct computation of noisy operator
println("\n" * "="^70)
println("Operator Decomposition (P_noisy = N * P)")
println("="^70)

t1 = @elapsed begin
    # Build deterministic operator
    println("\n  Building P (deterministic)...")
    P_det, U_det = build_transition_matrix(
        F, partition,
        transition_graph=dyn_graph,
        max_iterations=8, ε=1e-4
    )
    println("    P non-zeros: $(nnz(P_det))")

    # Build noise convolution operator
    println("\n  Building N (noise convolution)...")
    N = build_noise_convolution_operator(
        noise, partition,
        transition_graph=topo_graph,
        max_iterations=8, ε=1e-4
    )
    println("    N non-zeros: $(nnz(N))")

    # Compose: P_noisy = N * P_deterministic
    println("\n  Computing P_noisy = N * P...")
    P_composed = N * P_det
end

@printf("\n  Total time: %.3f s\n", t1)
println("  P_noisy non-zeros: $(nnz(P_composed))")

# Results
println("\n" * "="^70)
println("Results")
println("="^70)

println("\n  ✓ Successfully built noisy transition operator using composition")

# Demonstrate reusability: use P with different noise levels
println("\n" * "="^70)
println("Bonus: Reusing P for different noise levels")
println("="^70)

noise_levels = [0.01, 0.02, 0.05]
println("\nComputing P_noisy for multiple noise levels...")

for σ in noise_levels
    noise_new = uniform_noise_box(σ, 2)
    N_new = build_noise_convolution_operator(noise_new, partition,
                                              transition_graph=topo_graph,
                                              max_iterations=8, ε=1e-4)
    P_noisy_new = N_new * P_det
    @printf("  σ = %.2f: %d non-zeros\n", σ, nnz(P_noisy_new))
end

println("\n  ✓ Reused P_det for all noise levels (no recomputation needed!)")

println("\n" * "="^70)
println("Example completed successfully!")
println("="^70)
