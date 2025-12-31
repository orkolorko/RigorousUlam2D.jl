# Getting Started

## Installation

Install RigorousUlam2D.jl from the Julia package registry:

```julia
using Pkg
Pkg.add("RigorousUlam2D")
```

## Basic Workflow

The typical workflow involves four main steps:

1. **Define the dynamical system**
2. **Create a partition** (with optional domain restriction)
3. **Build the transition matrix**
4. **Analyze the operator**

Let's walk through a complete example.

## Example: Simple Contracting Map

```julia
using RigorousUlam2D
using IntervalArithmetic, IntervalBoxes
using LinearAlgebra

# Step 1: Define the system
F(x, y) = (0.5*x + 0.25, 0.5*y + 0.25)

# Step 2: Setup domain
domain = IntervalBox(interval(0.0, 1.0), interval(0.0, 1.0))

# Step 3: Create partition with domain restriction
partition, eff_domain, info = create_restricted_partition(
    F, domain, nothing,  # no noise
    partition_depth=4,   # 256 boxes
    verbose=true
)

# Step 4: Build transition matrix
P, U = build_transition_matrix(
    F, partition,
    maxdepth=8,
    ε=1e-4
)

# Normalize to stochastic matrix
col_sums = normalize_columns!(P)

# Step 5: Analyze
println("Matrix size: ", size(P))
println("Non-zeros: ", nnz(P))
println("Column sum range: [", minimum(col_sums), ", ", maximum(col_sums), "]")

# Compute invariant measure (leading eigenvector)
λ, v = eigen(Matrix(P))
idx = argmax(abs.(λ))
invariant_measure = abs.(v[:, idx])
invariant_measure ./= sum(invariant_measure)
```

## Example: Noisy System with Periodic BC

```julia
using RigorousUlam2D
using IntervalArithmetic, IntervalBoxes

# Hénon map (rescaled to [-1,1]²)
function henon_rescaled(x, y; a=1.4, b=0.3)
    x_phys = 1.5 * x
    y_phys = 0.5 * y
    x_new = (1 - a * x_phys^2 + y_phys) / 1.5
    y_new = (b * x_phys) / 0.5
    return (x_new, y_new)
end

# Setup
domain = IntervalBox(interval(-1.0, 1.0), interval(-1.0, 1.0))
noise = uniform_noise_box(0.025, 2)  # Square noise of size 0.05

# Create partition with diameter-based stopping (NEW!)
partition, eff_domain, info = create_restricted_partition(
    henon_rescaled, domain, noise,
    max_diameter=0.02,  # Target error tolerance
    subdivision_depth=2,
    verbose=true
)

# Build matrix with periodic BC
P, U = build_noisy_transition_matrix_periodic(
    henon_rescaled, partition, noise, domain,
    enclosure_method=first_order_enclosure,
    maxdepth=8,
    ε=1e-4
)

# Analyze
normalize_columns!(P)
stats = estimate_sparsity(P)
println("Sparsity: ", round(100*stats.sparsity, digits=2), "%")
```

## Understanding Domain Restriction

For contracting maps, computing on the full domain is wasteful:

```julia
# WITHOUT domain restriction
boxes_full = [domain]
for _ in 1:5
    boxes_full = vcat([quadrisect(b) for b in boxes_full]...)
end
n_full = length(boxes_full)  # 1024 boxes
println("Full domain: $n_full boxes")

# WITH domain restriction
partition_opt, _, info = create_restricted_partition(
    F, domain, nothing, partition_depth=5
)
println("Restricted: $(partition_opt.n_boxes) boxes")
println("Reduction: $(round(n_full/partition_opt.n_boxes, digits=2))×")
```

Typical results: **5-10× domain reduction → 25-100× fewer boxes → 100-1000× fewer matrix entries!**

## Diameter-Based Stopping (Error Control)

Instead of guessing the partition depth, specify the target error directly:

```julia
# Traditional: guess depth
partition1, _, _ = create_restricted_partition(
    F, domain, noise,
    partition_depth=5  # Is this enough? Too much?
)

# NEW: specify error tolerance
partition2, _, info = create_restricted_partition(
    F, domain, noise,
    max_diameter=0.01  # Achieve 1% error bound
)

println("Target error: 0.01")
println("Achieved diameter: ", info.partition_diameter)
println("Automatic boxes: ", info.n_boxes)
```

According to the theory, the Ulam approximation error is bounded by `C × diameter`, where `C` depends on the system. This provides direct control over approximation quality!

## Next Steps

- **Enclosure Methods** - Learn about different enclosure strategies
- **Domain Restriction** - Deep dive into optimization techniques
- **Examples** - More complete examples with analysis
- **API Reference** - Complete function documentation
