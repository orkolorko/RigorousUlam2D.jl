# Example 5: Contracting Lorenz Map
#
# This example demonstrates adaptive partitioning on a complex nonlinear map:
# the contracting Lorenz map, which has been studied in rigorous dynamics.
#
# Key features:
# 1. Highly nonlinear dynamics with singularities
# 2. Strong contraction in some directions
# 3. Demonstrates the power of adaptive partitioning for complex systems

# Activate example environment
import Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()  # Install dependencies if needed

using RigorousUlam2D
using IntervalArithmetic, IntervalBoxes
using LinearAlgebra, SparseArrays
using Printf
using CairoMakie

println("="^70)
println("Example 5: Contracting Lorenz Map")
println("="^70)

# Contracting Lorenz map
# This is a Poincaré section of the Lorenz attractor with strong contraction
function contracting_lorenz(x, y; s=2.5, r=6.0, ρ=0.7, c0=-0.25, c1=0.25)
    ax = abs(x)
    
    # For interval arguments, check which branch we're on
    if x isa Interval
        if inf(x) > 0
            # Entirely in positive branch (x > 0)
            return (-ρ * ax^s + 0.5, y * ax^r + c0)
        elseif sup(x) < 0
            # Entirely in negative branch (x < 0)
            return (ρ * ax^s - 0.5, -y * ax^r + c1)
        else
            # Contains x=0: take conservative hull of both branches
            Tx_pos = -ρ * ax^s + 0.5
            Gxy_pos = y * ax^r + c0
            Tx_neg = ρ * ax^s - 0.5
            Gxy_neg = -y * ax^r + c1
            return (RigorousUlam2D.hull(Tx_pos, Tx_neg), RigorousUlam2D.hull(Gxy_pos, Gxy_neg))
        end
    else
        # For dual numbers: the function is being evaluated at a point within a box
        # that doesn't contain 0 (since partition is refined at 0)
        # We can use the real part to determine the branch
        x_val = x isa Real ? x : ForwardDiff.value(x)
        if x_val > 0
            return (-ρ * ax^s + 0.5, y * ax^r + c0)
        else
            return (ρ * ax^s - 0.5, -y * ax^r + c1)
        end
    end
end

# Setup
domain = IntervalBox(interval(-0.5, 0.5), interval(-0.5, 0.5))
noise = uniform_noise_box(0.01, 2)

println("\nSystem:")
println("  Contracting Lorenz map (s=2.5, r=6.0, ρ=0.8)")
println("  Domain: [-0.5, 0.5]²")
println("  Noise: ±0.01 uniform")
println("\nThis map has:")
println("  - Strong contraction (ρ < 1)")
println("  - Nonlinearity (power law x^s, x^r)")
println("  - Singularity at x=0")

# Compare uniform vs adaptive partitioning
println("\n" * "="^70)
println("COMPARISON: Uniform vs Adaptive Partitioning")
println("="^70)

# Uniform partition (baseline)
depth = 5
partition_uniform = build_uniform_partition(domain, depth)
n_uniform = partition_uniform.n_boxes

println("\nUniform partition (depth $depth):")
println("  Boxes: $n_uniform")
@printf("  Potential matrix entries: %d (%.2e)\n", n_uniform^2, Float64(n_uniform^2))

# Adaptive partition with diameter-based stopping
target_diameter = 0.03125  # Approximately same as uniform depth 5

println("\nBuilding adaptive partition (target diameter: $target_diameter)...")
t_partition = @elapsed begin
    partition_adaptive, dyn_graph, topo_graph = build_adaptive_partition(
        contracting_lorenz, domain, noise,
        max_diameter=target_diameter,
        initial_depth=2,
        safety_margin=1.1,
        verbose=true,
        enclosure_method = zeroth_order_enclosure,
    )
end

max_diam = maximum(box_width(box) for box in partition_adaptive.boxes)

println("\nAdaptive partition results:")
@printf("  Time: %.2f s\n", t_partition)
@printf("  Target diameter: %.6f\n", target_diameter)
@printf("  Achieved diameter: %.6f\n", max_diam)
println("  Final boxes: $(partition_adaptive.n_boxes)")
println("  Graph non-zeros: $(nnz(dyn_graph))")
@printf("  Graph density: %.2f%%\n", 100*nnz(dyn_graph)/partition_adaptive.n_boxes^2)
@printf("  Graph sparsity: %.2f%%\n", 100*(1 - nnz(dyn_graph)/partition_adaptive.n_boxes^2))

# Compute reduction factors
box_reduction = n_uniform / partition_adaptive.n_boxes
graph_reduction = n_uniform^2 / nnz(dyn_graph)

println("\nReduction factors:")
@printf("  Boxes: %.1f× fewer\n", box_reduction)
@printf("  Graph entries: %.1f× sparser\n", graph_reduction)

# Build transition matrix
println("\n" * "="^70)
println("Building Transition Matrix")
println("="^70)

println("\nBuilding deterministic transition matrix...")
t_det = @elapsed begin
    P_det, U_det = build_transition_matrix(
        contracting_lorenz, partition_adaptive,
        transition_graph=dyn_graph,
        enclosure_method=zeroth_order_enclosure,
        max_iterations=8,
        ε=1e-4
    )
end

println("\nBuilding noise convolution operator...")
t_noise = @elapsed begin
    println("  Topo graph non-zeros: $(nnz(topo_graph)) (vs dynamics: $(nnz(dyn_graph)))")

    N = build_noise_convolution_operator(
        noise, partition_adaptive,
        transition_graph=topo_graph,
        ε=1e-3,
        max_iterations=10
    )
end

println("\nComposing operators: P = N * P_det")
P = N * P_det
t_matrix = t_det + t_noise

println("\nMatrix construction:")
@printf("  Time: %.2f s\n", t_matrix)
@printf("  Size: %d × %d\n", size(P)...)
@printf("  Non-zeros: %d\n", nnz(P))
stats = estimate_sparsity(P)
@printf("  Sparsity: %.2f%%\n", 100*stats.sparsity)

# Estimate time savings
estimated_time_without_graph = t_matrix * graph_reduction
println("\nEstimated performance:")
@printf("  With graph optimization: %.2f s\n", t_matrix)
@printf("  Without graph (estimated): %.2f s (%.1f min)\n",
        estimated_time_without_graph, estimated_time_without_graph/60)
@printf("  Speedup: %.1f×\n", graph_reduction)

# Normalize and analyze
println("\n" * "="^70)
println("Spectral Analysis")
println("="^70)

col_sums = normalize_columns!(P)
@printf("\nColumn sums (before normalization): [%.6f, %.6f]\n",
        minimum(col_sums), maximum(col_sums))

if size(P, 1) <= 500
    println("\nComputing eigenvalues...")
    λ = eigvals(Matrix(P))
    λ_sorted = sort(abs.(λ), rev=true)

    println("\nLeading eigenvalues:")
    for i in 1:min(10, length(λ_sorted))
        @printf("  λ_%d = %.8f\n", i, λ_sorted[i])
    end

    if length(λ_sorted) >= 2
        gap = 1.0 - λ_sorted[2]
        rate = -log(λ_sorted[2])
        @printf("\nSpectral properties:\n")
        @printf("  Spectral gap: %.6f\n", gap)
        @printf("  Mixing rate: %.6f\n", rate)
        @printf("  Mixing time: ~%.1f iterations\n", 1/rate)
    end

    # Invariant measure
    idx = argmax(abs.(λ))
    μ = abs.(real.(eigvecs(Matrix(P))[:, idx]))
    μ ./= sum(μ)

    active_boxes = count(μ .> 1e-4)
    @printf("\nInvariant measure:\n")
    @printf("  Active boxes (weight > 1e-4): %d / %d (%.1f%%)\n",
            active_boxes, partition_adaptive.n_boxes,
            100*active_boxes/partition_adaptive.n_boxes)
    @printf("  Max weight: %.6f\n", maximum(μ))
    @printf("  Min positive weight: %.2e\n", minimum(μ[μ .> 0]))
else
    println("\nPartition too large for dense eigenvalue computation")
    println("Consider using iterative methods (Arnoldi, ARPACK, etc.)")
end

# Summary
println("\n" * "="^70)
println("Summary")
println("="^70)

println("\nKey Results:")
println("  ✓ Adaptive partitioning: $(partition_adaptive.n_boxes) boxes (vs $n_uniform uniform)")
println("  ✓ Box reduction: $(round(box_reduction, digits=1))×")
println("  ✓ Graph sparsity: $(round(100*(1 - nnz(dyn_graph)/partition_adaptive.n_boxes^2), digits=1))%")
println("  ✓ Estimated speedup: $(round(graph_reduction, digits=1))×")
println("\nConclusion:")
println("  Adaptive partitioning with forward flagging is ESSENTIAL for")
println("  contracting maps. Without it, this computation would take")
println("  $(round(estimated_time_without_graph/60, digits=1)) minutes instead of $(round(t_matrix, digits=1)) seconds!")

#  Visualization
if size(P, 1) <= 500
    println("\n" * "="^70)
    println("Creating Visualizations")
    println("="^70)

    # Prepare density data
    x_centers = [mid(box[1]) for box in partition_adaptive.boxes]
    y_centers = [mid(box[2]) for box in partition_adaptive.boxes]
    densities = [μ[i] / box_measure(partition_adaptive.boxes[i]) for i in 1:partition_adaptive.n_boxes]

    # Create figure
    fig = Figure(size=(1200, 500))

    # Plot 1: Partition structure
    ax1 = Axis(fig[1, 1],
        xlabel="x", ylabel="y",
        title="Adaptive Partition ($(partition_adaptive.n_boxes) boxes)",
        aspect=1
    )

    # Draw partition boxes
    for box in partition_adaptive.boxes
        x_coords = [inf(box[1]), sup(box[1]), sup(box[1]), inf(box[1]), inf(box[1])]
        y_coords = [inf(box[2]), inf(box[2]), sup(box[2]), sup(box[2]), inf(box[2])]
        lines!(ax1, x_coords, y_coords, color=:gray, linewidth=0.5)
    end

    # Plot 2: Invariant density
    ax2 = Axis(fig[1, 2],
        xlabel="x", ylabel="y",
        title="Invariant Density",
        aspect=1
    )

    sc = scatter!(ax2, x_centers, y_centers,
        color=densities,
        colormap=:viridis,
        markersize=10,
        colorrange=(0, maximum(densities))
    )

    Colorbar(fig[1, 3], sc, label="Density ρ(x,y)")

    # Save figure
    save("examples/05_contracting_lorenz_output.png", fig)
    println("\nSaved visualization to examples/05_contracting_lorenz_output.png")
end

println("\n" * "="^70)
println("Example completed successfully!")
println("="^70)
