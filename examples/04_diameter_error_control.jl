# Example 4: Diameter-Based Error Control
#
# This example demonstrates:
# 1. Diameter-based stopping for direct error control
# 2. Comparison with fixed refinement levels
# 3. Trade-off between accuracy and computational cost

using RigorousUlam2D
using IntervalArithmetic, IntervalBoxes
using SparseArrays
using Printf

println("="^70)
println("Example 4: Diameter-Based Error Control")
println("="^70)

F(x, y) = (0.5*x + 0.25, 0.5*y + 0.25)
domain = IntervalBox(interval(0.0, 1.0), interval(0.0, 1.0))
# Grid-aligned noise: width = domain_width / 2^k (k=6)
# For domain width 1.0: 1.0/64 = 0.015625
noise = uniform_noise_box(1.0/64, 2)

println("\nSystem:")
println("  F(x,y) = (0.5x + 0.25, 0.5y + 0.25)")
println("  Noise: ±$(1.0/64) uniform (grid-aligned)")

# Target different error tolerances
target_diameters = [0.1, 0.05, 0.02, 0.01]

println("\n" * "="^70)
println("Error Tolerance Analysis")
println("="^70)
println()
@printf("%-12s | %-12s | %-10s | %-12s | %-12s\n",
        "Target δ", "Achieved δ", "Boxes", "Graph NNZ", "Time (s)")
println("-"^70)

for target in target_diameters
    t = @elapsed begin
        partition, dyn_graph, topo_graph = build_adaptive_partition(
            F, domain, noise,
            max_diameter=target,
            initial_depth=2,
            verbose=false
        )
    end

    achieved_diam = maximum(box_width(box) for box in partition.boxes)

    @printf("%-12.6f | %-12.6f | %-10d | %-12d | %-12.3f\n",
            target, achieved_diam, partition.n_boxes, nnz(dyn_graph), t)
end

println("-"^70)

# Demonstrate theory: error ≤ C × diameter
println("\n" * "="^70)
println("Theoretical Error Bound")
println("="^70)

println("\nTheorem: Ulam approximation error ≤ C × δ")
println("where:")
println("  δ = partition diameter")
println("  C = constant depending on system")
println()
println("Implication: To achieve error ε, set max_diameter = ε/C")
println()
println("Example:")
println("  If C ≈ 10 for your system")
println("  To get error ≤ 0.1, set max_diameter = 0.01")

# Compare diameter-based vs fixed levels
println("\n" * "="^70)
println("Diameter-Based vs Fixed Refinement Levels")
println("="^70)

println("\nFixed refinement levels:")
for levels in [1, 2, 3, 4]
    partition, dyn_graph, topo_graph = build_adaptive_partition(
        F, domain, noise,
        initial_depth=2,
        refinement_levels=levels,
        verbose=false
    )
    diam = maximum(box_width(box) for box in partition.boxes)
    @printf("  %d levels → %d boxes, diameter = %.6f\n",
            levels, partition.n_boxes, diam)
end

println("\nDiameter-based (automatic level selection):")
for target in [0.1, 0.05, 0.02]
    partition, dyn_graph, topo_graph = build_adaptive_partition(
        F, domain, noise,
        initial_depth=2,
        max_diameter=target,
        verbose=false
    )
    @printf("  diameter ≤ %.2f → %d boxes\n",
            target, partition.n_boxes)
end

println("\n✓ Diameter-based stopping provides direct error control!")
println("✓ No need to guess refinement levels!")

println("\n" * "="^70)
println("Example completed successfully!")
println("="^70)
