# Script to compute and save invariant density plot for contracting Lorenz map
# This generates a publication-quality figure of the stationary measure

using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using RigorousUlam2D
using IntervalArithmetic, IntervalBoxes
using SparseArrays
using LinearAlgebra
using Printf
using ForwardDiff
using CairoMakie

println("=" ^ 70)
println("Computing Invariant Density for Contracting Lorenz Map")
println("=" ^ 70)

# Define contracting Lorenz map
function contracting_lorenz(x, y; s=2.5, r=6.0, ρ=0.7, c0=-0.25, c1=0.25)
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

# Parameters
domain_full = IntervalBox(interval(-0.5, 0.5), interval(-0.5, 0.5))
noise = uniform_noise_box(0.01, 2)
target_diameter = 0.0625

println("\nStep 1: Computing partition with iterative refinement")
println("  Domain: $domain_full")
println("  Noise: ±0.01")
println("  Target diameter: $target_diameter")

partition, info = compute_effective_partition_adaptive(
    contracting_lorenz, domain_full, noise,
    initial_depth=2,
    max_diameter=target_diameter,
    enclosure_method=zeroth_order_enclosure,
    safety_margin=1.2,
    verbose=true
)

println("\nStep 2: Building transition matrix")
println("  Partition size: $(partition.n_boxes) boxes")

P, U = build_noisy_transition_matrix(
    contracting_lorenz, partition, noise,
    enclosure_method=zeroth_order_enclosure,
    maxdepth=8,
    ε=1e-4,
    use_lower_bound=true
)

println("  Matrix constructed: $(size(P))")

# Analyze sparsity
stats = estimate_sparsity(P)
@printf("  Non-zero entries: %d\n", stats.nnz)
@printf("  Sparsity: %.1f%%\n", 100 * stats.sparsity)

# Normalize to column-stochastic
P_norm = copy(P)
col_sums = normalize_columns!(P_norm)

println("\nStep 3: Computing stationary measure (leading eigenvector)")

P_dense = Matrix(P_norm)
λ, v = eigen(P_dense)

# Find leading eigenvalue
idx = argmax(abs.(λ))
λ_max = λ[idx]
stationary_measure = abs.(v[:, idx])
stationary_measure ./= sum(stationary_measure)

@printf("  Leading eigenvalue: %.8f\n", abs(λ_max))
@printf("  Measure sum: %.8f\n", sum(stationary_measure))
@printf("  Measure range: [%.6f, %.6f]\n", minimum(stationary_measure), maximum(stationary_measure))

# Spectral gap
λ_sorted = sort(abs.(λ), rev=true)
if length(λ_sorted) >= 2
    @printf("  Spectral gap: %.6f\n", abs(λ_sorted[1]) - abs(λ_sorted[2]))
    @printf("  Mixing rate: %.6f\n", -log(abs(λ_sorted[2])))
end

println("\nStep 4: Creating visualization")

# Extract box centers
boxes = partition.boxes
n_boxes = length(boxes)
centers_x = [mid(box[1]) for box in boxes]
centers_y = [mid(box[2]) for box in boxes]

# Create high-resolution figure
fig = Figure(resolution=(1400, 600))

# Plot 1: Scatter plot with colorbar
ax1 = Axis(fig[1, 1],
    xlabel="x",
    ylabel="y",
    title="Stationary Measure - Scatter Plot",
    aspect=DataAspect(),
    titlefont=:bold,
    titlesize=16)

sc = scatter!(ax1, centers_x, centers_y,
    color=stationary_measure,
    colormap=:viridis,
    markersize=12,
    colorrange=(0, maximum(stationary_measure)))

Colorbar(fig[1, 2], sc, label="Density", labelsize=14)

# Plot 2: Heatmap with box boundaries
ax2 = Axis(fig[1, 3],
    xlabel="x",
    ylabel="y",
    title="Stationary Measure - Heatmap",
    aspect=DataAspect(),
    titlefont=:bold,
    titlesize=16)

# Draw colored boxes
for i in 1:n_boxes
    box = boxes[i]
    x_min, x_max = inf(box[1]), sup(box[1])
    y_min, y_max = inf(box[2]), sup(box[2])

    rect_x = [x_min, x_max, x_max, x_min, x_min]
    rect_y = [y_min, y_min, y_max, y_max, y_min]

    poly!(ax2, Point2f.(zip(rect_x, rect_y)),
        color=(stationary_measure[i], 0.9),
        colormap=:viridis,
        colorrange=(0, maximum(stationary_measure)))
end

# Add domain boundary
lines!(ax2, [-0.5, 0.5, 0.5, -0.5, -0.5], [-0.5, -0.5, 0.5, 0.5, -0.5],
    color=:white, linewidth=2.5, linestyle=:dash)

# Add title with key info
Label(fig[0, :],
    text=@sprintf("Contracting Lorenz Map: Invariant Density (λ₁ = %.6f, %d boxes, diameter = %.4f)",
                  abs(λ_max), n_boxes, info.partition_diameter),
    fontsize=18,
    font=:bold)

# Save figure
output_file = joinpath(@__DIR__, "lorenz_invariant_density.png")
save(output_file, fig, px_per_unit=2)  # High DPI

println("  Plot saved to: $output_file")

println("\n" * "=" ^ 70)
println("SUCCESS: Invariant density computed and saved!")
println("=" ^ 70)

# Summary statistics
println("\nSummary:")
println("  Method: Iterative refinement with diameter-based stopping")
println("  Partition: $(partition.n_boxes) boxes")
println("  Diameter: $(round(info.partition_diameter, digits=6))")
println("  Domain reduction: $(round(info.reduction_factor, digits=2))×")
println("  Matrix size: $(size(P))")
println("  Matrix sparsity: $(round(100 * stats.sparsity, digits=1))%")
println("  Leading eigenvalue: $(round(abs(λ_max), digits=8))")
println("  Spectral gap: $(round(abs(λ_sorted[1]) - abs(λ_sorted[2]), digits=6))")
println("  Output file: $output_file")
println()
