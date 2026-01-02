# Examples

Complete examples demonstrating the full workflow for various dynamical systems.

## Example 1: Logistic Map (Product System)

The 2D product of logistic maps: `F(x,y) = (4x(1-x), 4y(1-y))`

```julia
using RigorousUlam2D
using IntervalArithmetic, IntervalBoxes
using LinearAlgebra

# Define system
F_logistic(x, y) = (4*x*(1-x), 4*y*(1-y))

# Domain and partition
domain = IntervalBox(interval(0.0, 1.0), interval(0.0, 1.0))

partition, transition_graph = build_adaptive_partition(
    F_logistic, domain, nothing,
    initial_depth=3,
    refinement_levels=2,
    verbose=true
)

# Build matrix (using graph for speedup)
P, U = build_transition_matrix(
    F_logistic, partition,
    transition_graph=transition_graph,
    max_iterations=10,
    ε=1e-5
)

# Normalize
col_sums = normalize_columns!(P)
println("Column sum range: [", minimum(col_sums), ", ", maximum(col_sums), "]")

# Sparsity analysis
stats = estimate_sparsity(P)
println("Matrix size: ", size(P))
println("Non-zeros: ", stats.nnz)
println("Sparsity: ", round(100*stats.sparsity, digits=2), "%")

# Find invariant measure
λ, V = eigen(Matrix(P))
idx = argmax(real.(λ))
μ = abs.(V[:, idx])
μ ./= sum(μ)

println("Leading eigenvalue: ", λ[idx])
println("Invariant measure support: ", count(μ .> 1e-6), " boxes")
```

## Example 2: Contracting Map with Noise

A simple contracting map with additive noise.

```julia
using RigorousUlam2D
using IntervalArithmetic, IntervalBoxes

# Contracting map
F(x, y) = (0.5*x + 0.25, 0.5*y + 0.25)

domain = IntervalBox(interval(0.0, 1.0), interval(0.0, 1.0))

# Compare noise levels
noise_levels = [0.0, 0.01, 0.05, 0.1]

println("Noise Level Comparison")
println("=" ^ 70)

for σ in noise_levels
    noise = σ > 0 ? uniform_noise_box(σ, 2) : nothing

    # Create adaptive partition
    partition, graph = build_adaptive_partition(
        F, domain, noise,
        initial_depth=2,
        refinement_levels=2,
        verbose=false
    )

    # Build matrix (unified interface handles both deterministic and noisy)
    P, U = build_transition_matrix(
        F, partition, noise,
        transition_graph=graph,
        max_iterations=8,
        ε=1e-4
    )

    normalize_columns!(P)
    stats = estimate_sparsity(P)

    println("\nNoise σ = $σ:")
    println("  Partition boxes: ", partition.n_boxes)
    println("  Matrix non-zeros: ", stats.nnz)
    println("  Sparsity: ", round(100*stats.sparsity, digits=2), "%")
end

println("\n" * "=" ^ 70)
```

## Example 3: Hénon Map (Complete Analysis)

Full analysis of the Hénon map with noise and periodic boundary conditions.

```julia
using RigorousUlam2D
using IntervalArithmetic, IntervalBoxes
using LinearAlgebra
using Printf

# Rescaled Hénon map
function henon_rescaled(x, y; a=1.4, b=0.3)
    x_phys = 1.5 * x
    y_phys = 0.5 * y
    x_new = (1 - a * x_phys^2 + y_phys) / 1.5
    y_new = (b * x_phys) / 0.5
    return (x_new, y_new)
end

# Parameters
domain = IntervalBox(interval(-1.0, 1.0), interval(-1.0, 1.0))
noise = uniform_noise_box(0.025, 2)  # Noise: ±0.025

println("=" ^ 70)
println("HÉNON MAP ANALYSIS")
println("=" ^ 70)

# Step 1: Compare partition strategies
println("\n1. PARTITION STRATEGY COMPARISON\n")

# Uniform partition
partition_naive = build_uniform_partition(domain, 4)

# Adaptive with diameter control
partition_opt, graph_opt = build_adaptive_partition(
    henon_rescaled, domain, noise,
    max_diameter=0.02,
    initial_depth=2,
    periodic=true,
    verbose=false
)

naive_diam = maximum(box_width(box) for box in partition_naive.boxes)
opt_diam = maximum(box_width(box) for box in partition_opt.boxes)

@printf("%-20s: %4d boxes, diameter = %.6f\n",
        "Uniform (depth 4)", partition_naive.n_boxes, naive_diam)
@printf("%-20s: %4d boxes, diameter = %.6f\n",
        "Adaptive", partition_opt.n_boxes, opt_diam)
@printf("%-20s: %.1f×\n",
        "Reduction factor", partition_naive.n_boxes / partition_opt.n_boxes)

# Step 2: Build transition matrix
println("\n2. BUILDING TRANSITION MATRIX\n")

P, U = build_transition_matrix(
    henon_rescaled, partition_opt, noise,
    transition_graph=graph_opt,
    periodic=true,
    enclosure_method=first_order_enclosure,
    max_iterations=8,
    ε=1e-4
)

stats = estimate_sparsity(P)
println("Matrix constructed:")
@printf("  Size: %d × %d\n", size(P)...)
@printf("  Non-zeros: %d\n", stats.nnz)
@printf("  Sparsity: %.2f%%\n", 100*stats.sparsity)
@printf("  Avg per column: %.1f\n", stats.avg_per_column)

# Step 3: Normalize and analyze
println("\n3. SPECTRAL ANALYSIS\n")

col_sums = normalize_columns!(P)
@printf("Column sums before: [%.6f, %.6f]\n",
        minimum(col_sums), maximum(col_sums))

new_sums = [sum(P[:, j]) for j in 1:size(P, 2)]
@printf("Column sums after:  [%.6f, %.6f]\n",
        minimum(new_sums), maximum(new_sums))

# Eigenvalues (if small enough)
if size(P, 1) <= 500
    println("\nComputing eigenvalues...")
    λ = eigvals(Matrix(P))
    λ_sorted = sort(abs.(λ), rev=true)

    println("Leading eigenvalues:")
    for i in 1:min(10, length(λ_sorted))
        @printf("  λ_%d = %.6f\n", i, λ_sorted[i])
    end

    if length(λ_sorted) >= 2
        gap = 1.0 - λ_sorted[2]
        rate = -log(λ_sorted[2])
        @printf("\nSpectral gap: %.6f\n", gap)
        @printf("Mixing rate:  %.6f\n", rate)
        @printf("Mixing time:  ~%.1f iterations\n", 1/rate)
    end

    # Invariant measure
    idx = argmax(real.(λ))
    μ = abs.(V[:, idx])
    μ ./= sum(μ)

    active_boxes = count(μ .> 1e-4)
    @printf("\nInvariant measure:\n")
    @printf("  Active boxes (>1e-4): %d / %d\n",
            active_boxes, partition_opt.n_boxes)
    @printf("  Max weight: %.4f\n", maximum(μ))
    @printf("  Min weight: %.4e\n", minimum(μ[μ .> 0]))
else
    println("Partition too large for dense eigenvalue computation")
    println("Consider using iterative methods (Arnoldi, etc.)")
end

println("\n" * "=" ^ 70)
```

## Example 4: Diameter-Based Error Control

Demonstrating the new diameter-based stopping criterion.

```julia
using RigorousUlam2D
using IntervalArithmetic, IntervalBoxes
using Printf

F(x, y) = (0.5*x + 0.25, 0.5*y + 0.25)
domain = IntervalBox(interval(0.0, 1.0), interval(0.0, 1.0))
noise = uniform_noise_box(0.01, 2)

# Target different error tolerances
target_errors = [0.1, 0.05, 0.02, 0.01, 0.005]

println("DIAMETER-BASED ERROR CONTROL")
println("=" ^ 80)
@printf("%-15s | %-15s | %-10s | %-15s | %-15s\n",
        "Target δ", "Achieved δ", "Boxes", "Matrix Entries", "Est. Time")
println("=" ^ 80)

for target in target_errors
    partition, graph = build_adaptive_partition(
        F, domain, noise,
        max_diameter=target,
        initial_depth=2,
        verbose=false
    )

    achieved_diam = maximum(box_width(box) for box in partition.boxes)
    matrix_size = partition.n_boxes^2
    # Rough time estimate (based on typical ~1000 entries/sec)
    est_time_sec = matrix_size / 1000

    @printf("%-15.6f | %-15.6f | %-10d | %-15d | %-15s\n",
            target,
            achieved_diam,
            partition.n_boxes,
            matrix_size,
            est_time_sec < 60 ? "$(round(est_time_sec, digits=1))s" :
            "$(round(est_time_sec/60, digits=1))min")
end

println("=" ^ 80)

println("\nKey Insight:")
println("- Error bound: ‖P - P_δ‖ ≤ C × δ")
println("- To achieve 1% error: set max_diameter = 0.01")
println("- Package automatically finds optimal subdivision!")
```

## Example 5: Operator Decomposition

Demonstrating the noise convolution operator for operator decomposition: P_noisy = N * P_deterministic

```julia
using RigorousUlam2D
using IntervalArithmetic, IntervalBoxes
using Printf

# Simple contracting map
F(x, y) = (0.5*x + 0.25, 0.5*y + 0.25)
domain = IntervalBox(interval(0.0, 1.0), interval(0.0, 1.0))
noise = uniform_noise_box(0.02, 2)

println("OPERATOR DECOMPOSITION: P_noisy = N * P_deterministic")
println("=" ^ 70)

# Build adaptive partition
partition, graph = build_adaptive_partition(
    F, domain, noise,
    max_diameter=0.03,
    initial_depth=2,
    verbose=true
)

# Method 1: Direct noisy transition matrix
println("\n1. Direct method (computing P_noisy directly):")
t1 = @elapsed begin
    P_direct, U_direct = build_transition_matrix(
        F, partition, noise,
        transition_graph=graph,
        max_iterations=8, ε=1e-4
    )
end
println("  Time: $(round(t1, digits=3))s")
println("  Non-zeros: $(nnz(P_direct))")

# Method 2: Operator decomposition
println("\n2. Decomposition method (P_noisy = N * P):")
t2 = @elapsed begin
    # Build deterministic operator
    P_det, U_det = build_transition_matrix(
        F, partition, nothing,  # no noise
        transition_graph=graph,
        max_iterations=8, ε=1e-4
    )

    # Build noise convolution operator
    N = build_noise_convolution_operator(
        noise, partition,
        transition_graph=graph,
        max_iterations=8, ε=1e-4
    )

    # Compose: P_noisy = N * P_deterministic
    P_composed = N * P_det
end
println("  Time: $(round(t2, digits=3))s")
println("  Non-zeros P: $(nnz(P_det))")
println("  Non-zeros N: $(nnz(N))")
println("  Non-zeros P_noisy: $(nnz(P_composed))")

# Compare results
difference = norm(P_direct - P_composed, Inf)
println("\n3. Comparison:")
println("  Max difference: $(difference)")
println("  Speedup: $(round(t1/t2, digits=2))×")

println("\n" * "=" ^ 70)
println("\nKey insight: Operator decomposition allows reusing P for different noise levels!")
```

## Running the Examples

All examples are available as Jupyter notebooks in the `notebooks/` directory:

- `rigorous_ulam_showcase.ipynb` - General tutorial
- `henon_map_periodic.ipynb` - Complete Hénon map analysis
- `henon_comparison.ipynb` - Comparison of optimization strategies

To run:
```bash
jupyter notebook notebooks/
```

## Next Steps

- **API Reference** - Complete function documentation
- **Domain Restriction** - Deep dive into optimization
- **Enclosure Methods** - Advanced enclosure techniques
