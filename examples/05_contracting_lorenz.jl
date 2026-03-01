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
using Arpack
using Printf
using CairoMakie

println("="^70)
println("Example 5: Contracting Lorenz Map")
println("="^70)

# Contracting Lorenz map
# This is a Poincaré section of the Lorenz attractor with strong contraction
# s < 1 gives expansion at x=0 (|T'| ~ |x|^{s-1} → ∞), creating the booklet structure
# r > 1 gives contraction in y, collapsing onto the 1D attractor
function contracting_lorenz(x, y; s=0.5, r=3.0, ρ=1.3, c0=-0.25, c1=0.25)
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
# Grid-aligned noise: width = domain_width / 2^k (k=4)
# For domain width 1.0: 1.0/16 = 0.0625
noise = uniform_noise_box(1.0/16, 2)

println("\nSystem:")
println("  Contracting Lorenz map (s=0.5, r=3.0, ρ=1.3)")
println("  Domain: [-0.5, 0.5]²")
println("  Noise: ±$(1.0/16) uniform (grid-aligned)")
println("\nThis map has:")
println("  - Expansion at x=0 (|T'| = sρ|x|^{s-1} → ∞)")
println("  - Strong y-contraction (|x|^r → 0)")
println("  - Booklet structure: two sheets merging at the singularity")

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
target_diameter = 0.02  # Fine enough to resolve the thin booklet structure

println("\nBuilding adaptive partition (target diameter: $target_diameter)...")
t_partition = @elapsed begin
    partition_adaptive, dyn_graph, topo_graph = build_adaptive_partition(
        contracting_lorenz, domain, noise,
        max_diameter=target_diameter,
        initial_depth=2,
        safety_margin=1.1,
        verbose=true,
        enclosure_method=zeroth_order_enclosure,
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
        max_iterations=12,
        ε=1e-6
    )
end

println("\nBuilding grid-aligned noise operator...")
t_noise = @elapsed begin
    println("  Topo graph non-zeros: $(nnz(topo_graph)) (vs dynamics: $(nnz(dyn_graph)))")

    N = build_grid_aligned_noise_operator(
        noise, partition_adaptive,
        topological_graph=topo_graph
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
    # Use dense eigendecomposition for small matrices
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
else
    # Use Arpack for large matrices
    println("\nMatrix size $(size(P, 1)) - using sparse Arnoldi method...")

    # Compute leading eigenvalues
    λ_vals, _ = eigs(P, nev=min(10, size(P,1)-2), which=:LM)
    λ_sorted = sort(abs.(λ_vals), rev=true)

    println("\nLeading eigenvalues:")
    for i in 1:length(λ_sorted)
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

    # Invariant measure (right eigenvector of column-stochastic P)
    # Use nev=2 because this period-2 map has two eigenvalues at +1
    # (left and right branches are separate ergodic components of P²)
    λ_lead, v_lead = eigs(P, nev=2, which=:LR)
    n_unity = count(real.(λ_lead) .> 0.999)
    if n_unity > 1
        # Combine both components for the full invariant density
        μ = sum(abs.(real.(v_lead[:, i])) for i in 1:n_unity)
    else
        μ = abs.(real.(v_lead[:, 1]))
    end
    μ ./= sum(μ)
end

# Rigorous error bound
println("\nRigorous error bound (Theorem 4.2, ZZ paper)...")
error_result = rigorous_error_bound(P, partition_adaptive, noise, verbose=true)
if error_result.mixing_satisfied
    @printf("\n  ✓ Rigorous L¹ error bound: %.6f\n", error_result.error_bound)
    @printf("    Contraction power N = %d, rate α = %.6f\n", error_result.N, error_result.α)
    @printf("    Mesh size δ = %.6f, noise scale ξ = %.6f\n", error_result.δ, error_result.ξ)
else
    println("\n  ✗ Mixing condition not satisfied — bound is infinite")
    println("    Try increasing noise or refining the partition")
end

active_boxes = count(μ .> 1e-4)
@printf("\nInvariant measure:\n")
@printf("  Active boxes (weight > 1e-4): %d / %d (%.1f%%)\n",
        active_boxes, partition_adaptive.n_boxes,
        100*active_boxes/partition_adaptive.n_boxes)
@printf("  Max weight: %.6f\n", maximum(μ))
@printf("  Min positive weight: %.2e\n", minimum(μ[μ .> 0]))

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
println("\n" * "="^70)
println("Creating Visualizations")
println("="^70)

# Prepare density data
densities = [μ[i] / box_measure(partition_adaptive.boxes[i]) for i in 1:partition_adaptive.n_boxes]

# Create figure
fig = Figure(size=(1200, 500))

# Plot 1: Partition structure (outline only)
ax1 = Axis(fig[1, 1],
    xlabel="x", ylabel="y",
    title="Adaptive Partition ($(partition_adaptive.n_boxes) boxes)",
    aspect=1
)

# Draw partition boxes as outlines (vectorized)
rects_outline = [Rect2f(inf(box[1]), inf(box[2]),
                        sup(box[1]) - inf(box[1]), sup(box[2]) - inf(box[2]))
                 for box in partition_adaptive.boxes]
poly!(ax1, rects_outline, color=:transparent, strokecolor=:gray, strokewidth=0.5)

# Plot 2: Invariant density (filled rectangles)
ax2 = Axis(fig[1, 2],
    xlabel="x", ylabel="y",
    title="Invariant Density",
    aspect=1
)

# Plot all boxes as colored rectangles (vectorized for performance)
rects_density = [Rect2f(inf(box[1]), inf(box[2]),
                        sup(box[1]) - inf(box[1]), sup(box[2]) - inf(box[2]))
                 for box in partition_adaptive.boxes]
poly!(ax2, rects_density, color=densities, colormap=:viridis,
      colorrange=(0, maximum(densities)))

Colorbar(fig[1, 3], colormap=:viridis,
         limits=(0, maximum(densities)), label="Density ρ(x,y)")

# Save figure
output_file = joinpath(@__DIR__, "05_contracting_lorenz_output.png")
save(output_file, fig)
println("\nSaved visualization to ", output_file)

println("\n" * "="^70)
println("Example completed successfully!")
println("="^70)
