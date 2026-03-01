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
using Arpack
using Printf
using CairoMakie

println("="^70)
println("Example 3: Operator Decomposition")
println("="^70)

# Simple contracting map
F(x, y) = (0.5*x + 0.25, 0.5*y + 0.25)
domain = IntervalBox(interval(0.0, 1.0), interval(0.0, 1.0))
# Grid-aligned noise: width = domain_width / 2^k (k=5)
# For domain width 1.0: 1.0/32 = 0.03125
noise = uniform_noise_box(1.0/32, 2)

println("\nSystem:")
println("  F(x,y) = (0.5x + 0.25, 0.5y + 0.25)")
println("  Noise: ±$(1.0/32) uniform (grid-aligned)")

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

    # Build grid-aligned noise operator
    println("\n  Building N (grid-aligned noise)...")
    N = build_grid_aligned_noise_operator(
        noise, partition,
        topological_graph=topo_graph
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

# Grid-aligned noise levels: domain_width / 2^k for different k values
noise_levels = [1.0/64, 1.0/32, 1.0/16]  # k=6, k=5, k=4
println("\nComputing P_noisy for multiple grid-aligned noise levels...")

for σ in noise_levels
    noise_new = uniform_noise_box(σ, 2)
    N_new = build_grid_aligned_noise_operator(noise_new, partition,
                                               topological_graph=topo_graph)
    P_noisy_new = N_new * P_det
    @printf("  σ = %.5f: %d non-zeros\n", σ, nnz(P_noisy_new))
end

println("\n  ✓ Reused P_det for all noise levels (no recomputation needed!)")

# Visualization: density for the composed operator
println("\n" * "="^70)
println("Density Visualization")
println("="^70)

P_vis = copy(P_composed)
normalize_columns!(P_vis)

println("\nComputing invariant measure...")
if size(P_vis, 1) <= 500
    λ_all, v_all = eigen(Matrix(P_vis))
    idx = argmax(abs.(λ_all))
    μ = abs.(v_all[:, idx])
    μ ./= sum(μ)
else
    λ_lead, v_lead = eigs(P_vis, nev=1, which=:LR)
    μ = abs.(v_lead[:, 1])
    μ ./= sum(μ)
end
println("  Active boxes (>1e-4): ", count(μ .> 1e-4))

densities = [μ[i] / box_measure(partition.boxes[i]) for i in 1:partition.n_boxes]

fig = Figure(size=(800, 700))
ax = Axis(fig[1, 1],
    xlabel="x", ylabel="y",
    title="Invariant Density - Operator Decomposition",
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
output_file = joinpath(@__DIR__, "03_operator_decomposition_output.png")
save(output_file, fig)
println("Saved visualization to ", output_file)

println("\n" * "="^70)
println("Example completed successfully!")
println("="^70)
