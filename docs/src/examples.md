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

partition, _, info = create_restricted_partition(
    F_logistic, domain, nothing,
    partition_depth=5,
    verbose=true
)

# Build matrix
P, U = build_transition_matrix(
    F_logistic, partition,
    maxdepth=10,
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

    # Create partition
    partition, eff_dom, info = create_restricted_partition(
        F, domain, noise,
        partition_depth=4,
        verbose=false
    )

    # Build matrix
    if noise === nothing
        P, U = build_transition_matrix(F, partition, maxdepth=8, ε=1e-4)
    else
        P, U = build_noisy_transition_matrix(F, partition, noise,
                                             maxdepth=8, ε=1e-4)
    end

    normalize_columns!(P)
    stats = estimate_sparsity(P)

    println("\nNoise σ = $σ:")
    println("  Effective domain: ", eff_dom)
    println("  Partition boxes: ", info.n_boxes)
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

# Naive
partition_naive, _, info_naive = create_restricted_partition(
    henon_rescaled, domain, nothing,
    partition_depth=4,
    verbose=false
)

# With domain restriction
partition_opt, _, info_opt = create_restricted_partition(
    henon_rescaled, domain, noise,
    max_diameter=0.02,
    subdivision_depth=2,
    verbose=false
)

@printf("%-20s: %4d boxes, diameter = %.6f\n",
        "Naive (no noise)", info_naive.n_boxes, info_naive.partition_diameter)
@printf("%-20s: %4d boxes, diameter = %.6f\n",
        "Optimized", info_opt.n_boxes, info_opt.partition_diameter)
@printf("%-20s: %.1f×\n",
        "Reduction factor", info_naive.n_boxes / info_opt.n_boxes)

# Step 2: Build transition matrix
println("\n2. BUILDING TRANSITION MATRIX\n")

P, U = build_noisy_transition_matrix_periodic(
    henon_rescaled, partition_opt, noise, domain,
    enclosure_method=first_order_enclosure,
    maxdepth=8,
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
    partition, _, info = create_restricted_partition(
        F, domain, noise,
        max_diameter=target,
        verbose=false
    )

    matrix_size = info.n_boxes^2
    # Rough time estimate (based on typical ~1000 entries/sec)
    est_time_sec = matrix_size / 1000

    @printf("%-15.6f | %-15.6f | %-10d | %-15d | %-15s\n",
            target,
            info.partition_diameter,
            info.n_boxes,
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

## Example 5: Adaptive Enclosure Comparison

Comparing different enclosure strategies for the effective domain.

```julia
using RigorousUlam2D
using IntervalArithmetic, IntervalBoxes
using Printf

# Highly nonlinear map
F(x, y) = (x - y^2, y + x^2)
domain = IntervalBox(interval(-0.5, 0.5), interval(-0.5, 0.5))

println("ADAPTIVE ENCLOSURE STRATEGIES")
println("=" ^ 70)

strategies = [
    ("Standard (depth=0)", nothing, 0),
    ("Standard (depth=2)", nothing, 2),
    ("Adaptive Error", ErrorBased(1e-3, 8), nothing),
    ("Adaptive Expansion", ExpansionBased(2.0, 8), nothing)
]

@printf("%-25s | %-15s | %-15s | %-10s\n",
        "Strategy", "Image Volume", "Eff. Domain", "Time (ms)")
println("=" ^ 70)

for (name, adaptive_method, subdiv_depth) in strategies
    t_start = time()

    if adaptive_method === nothing
        # Standard approach
        eff_dom, img_clean, _ = compute_effective_domain(
            F, domain, nothing,
            subdivision_depth=subdiv_depth
        )
        img_vol = box_measure(img_clean)
    else
        # Adaptive approach
        img_adaptive = adaptive_enclosure(F, domain, adaptive_method)
        img_vol = box_measure(img_adaptive)
        eff_dom = intersect_boxes(
            expand_box(img_adaptive, 1.1),
            domain
        )
    end

    t_elapsed = 1000 * (time() - t_start)
    eff_vol = box_measure(eff_dom)

    @printf("%-25s | %-15.6f | %-15.6f | %-10.3f\n",
            name, img_vol, eff_vol, t_elapsed)
end

println("=" ^ 70)
println("\nResult: Adaptive methods provide tighter bounds → smaller")
println("        effective domain → fewer boxes → faster computation!")
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
