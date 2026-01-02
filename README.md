# RigorousUlam2D.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://orkolorko.github.io/RigorousUlam2D.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://orkolorko.github.io/RigorousUlam2D.jl/dev/)
[![Build Status](https://github.com/orkolorko/RigorousUlam2D.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/orkolorko/RigorousUlam2D.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/orkolorko/RigorousUlam2D.jl/graph/badge.svg?token=n1T74OYCKL)](https://codecov.io/gh/orkolorko/RigorousUlam2D.jl)

**Rigorous computation of transfer operators for 2D dynamical systems using the Ulam method.**

## Overview

RigorousUlam2D.jl provides mathematically verified bounds for transfer operators of 2D dynamical systems with:
- **Interval arithmetic** for guaranteed enclosures
- **Adaptive partitioning** with forward flagging for optimal efficiency
- **Diameter-based stopping** for direct error control
- **Unified architecture** handling deterministic/noisy systems and absorbing/periodic boundary conditions
- **Graph-based sparsity optimization** for dramatic speedup
- **Operator decomposition** P_noisy = N ∘ P for flexible noise handling

All computations are **rigorously verified** - suitable for computer-assisted proofs.

## Key Features

### Adaptive Partitioning with Forward Flagging

For **contracting maps** and **systems with attractors**, use adaptive partitioning with diameter-based stopping:

```julia
using RigorousUlam2D
using IntervalArithmetic, IntervalBoxes

# Define your dynamical system
F(x, y) = (contracting_map_x(x, y), contracting_map_y(x, y))

# Setup
domain = IntervalBox(interval(-1.0, 1.0), interval(-1.0, 1.0))
noise = uniform_noise_box(0.01, 2)  # Optional additive noise

# Adaptive partition with diameter-based stopping
partition, transition_graph = build_adaptive_partition(
    F, domain, noise,
    initial_depth=2,           # Start with coarse partition (2^4 boxes)
    max_diameter=0.01,         # Target error tolerance
    enclosure_method=first_order_enclosure,
    safety_margin=1.1,
    verbose=true
)

# Build transition matrix (using graph for speedup)
P, U = build_transition_matrix(
    F, partition, noise,
    transition_graph=transition_graph,
    max_iterations=8, ε=1e-4
)
normalize_columns!(P)

# Analyze
stats = estimate_sparsity(P)
println("Partition: $(partition.n_boxes) boxes")
max_diam = maximum(box_width(box) for box in partition.boxes)
println("Diameter: $(max_diam)")
println("Sparsity: $(round(100*stats.sparsity, digits=1))%")
println("Graph sparsity: $(round(100*(1 - nnz(transition_graph)/partition.n_boxes^2), digits=1))%")
```

**Why adaptive partitioning?**

1. **Forward flagging**: Computes F(box) + noise to identify reachable boxes
2. **Greedy subdivision**: Only subdivides boxes violating diameter criterion
3. **Diameter control**: Direct link to theoretical error bounds
4. **Graph optimization**: Tracks potential transitions for 10-100× matrix speedup

**Performance**: For contracting maps, reduces computation by **10-100× compared to uniform partitioning**, while maintaining the same rigorous guarantees.

### Diameter-Based Stopping

The diameter of the partition directly controls the approximation error:

**Theorem B (Error Bound)**:
```
||f - π_δ f||_{L¹} ≤ (δ/2) var(f)
```
where δ is the partition diameter.

By specifying `max_diameter`, you directly control the error tolerance!

## Installation

```julia
using Pkg
Pkg.add("RigorousUlam2D")
```

## Quick Start

### Example 1: Contracting Lorenz Map

```julia
using RigorousUlam2D
using IntervalArithmetic, IntervalBoxes

# Contracting Lorenz map
function contracting_lorenz(x, y; s=2.5, r=6.0, ρ=0.8)
    ax = abs(x)
    if x > 0
        return (-ρ * ax^s + 0.5, y * ax^r)
    else
        return (ρ * ax^s - 0.5, -y * ax^r)
    end
end

domain = IntervalBox(interval(-0.5, 0.5), interval(-0.5, 0.5))
noise = uniform_noise_box(0.01, 2)

# Adaptive partitioning: automatically finds effective domain
partition, graph = build_adaptive_partition(
    contracting_lorenz, domain, noise,
    initial_depth=2,
    max_diameter=0.05,
    verbose=true
)

# Build transition matrix with graph optimization
P, U = build_transition_matrix(
    contracting_lorenz, partition, noise,
    transition_graph=graph,
    max_iterations=8, ε=1e-4
)

# Results:
# - Adaptive: ~50-100 boxes (vs 256+ uniform)
# - Diameter ≤ 0.05 (error control)
# - Graph provides 10-100× speedup for matrix construction
```

### Example 2: Hénon Map with Periodic BC

For maps on the torus:

```julia
using RigorousUlam2D
using IntervalArithmetic, IntervalBoxes
using LinearAlgebra

# Hénon map (rescaled)
function henon_rescaled(x, y; a=1.4, b=0.3)
    x_phys = 1.5 * x
    y_phys = 0.5 * y
    x_new = (1 - a * x_phys^2 + y_phys) / 1.5
    y_new = (b * x_phys) / 0.5
    return (x_new, y_new)
end

domain = IntervalBox(interval(-1.0, 1.0), interval(-1.0, 1.0))
noise = uniform_noise_box(0.025, 2)

# Adaptive partition with periodic BC
partition, graph = build_adaptive_partition(
    henon_rescaled, domain, noise,
    max_diameter=0.02,
    initial_depth=2,
    periodic=true,  # Enable periodic boundary conditions
    verbose=true
)

# Build matrix
P, U = build_transition_matrix(
    henon_rescaled, partition, noise,
    transition_graph=graph,
    periodic=true,
    max_iterations=8, ε=1e-4
)
normalize_columns!(P)

# Compute leading eigenvalues
λ = eigvals(Matrix(P))
λ_sorted = sort(abs.(λ), rev=true)
println("Spectral gap: ", 1.0 - λ_sorted[2])
```

## Algorithm: Adaptive Partitioning with Forward Flagging

The adaptive partitioning algorithm works as follows:

1. **Initialize**: Start with coarse partition (e.g., `initial_depth=2` → 2^(N·2) boxes)

2. **Forward Flagging**: For each box B in current partition:
   - Compute rigorous enclosure: `img = F(B) ⊕ noise ⊕ margin`
   - Flag all boxes that intersect with `img`
   - Track transition: if box j intersects img(B_i), set `graph[i,j] = true`

3. **Greedy Refinement**: Only subdivide flagged boxes where `box_width(B) > max_diameter`
   - Uses `bisect_box` (splits along widest dimension)
   - Boxes already meeting diameter criterion are kept as-is

4. **Stopping Criterion**:
   - If all flagged boxes have `diameter ≤ max_diameter` → DONE
   - If max iterations reached → DONE
   - Otherwise, go to step 2

5. **Output**:
   - Partition focused on reachable set (effective domain)
   - Transition graph encoding potential mass transfers

**Key Insights**:
- Boxes outside the reachable set are never flagged, so they're never refined
- Greedy subdivision ensures efficient convergence
- Transition graph provides 10-100× speedup for matrix construction

## API Reference

### Adaptive Partitioning

```julia
# RECOMMENDED: Adaptive partition with forward flagging
build_adaptive_partition(F, domain, noise=nothing;
    initial_depth=2,           # Starting partition depth
    max_diameter=nothing,      # Diameter-based stopping (recommended)
    refinement_levels=nothing, # Or fixed number of iterations
    enclosure_method=first_order_enclosure,
    safety_margin=1.1,
    periodic=false,            # Enable periodic boundary conditions
    verbose=false)
# Returns: (partition, transition_graph)

# Uniform partition (baseline)
build_uniform_partition(domain, depth)
# Returns: partition
```

### Unified Transition Matrix Builder

```julia
# Handles both deterministic/noisy and absorbing/periodic
build_transition_matrix(F, partition, noise=nothing;
    transition_graph=nothing,  # Optional speedup from build_adaptive_partition
    periodic=false,            # Periodic boundary conditions
    enclosure_method=first_order_enclosure,
    max_iterations=30,
    ε=1e-6,
    use_lower_bound=true)
# Returns: (P, U) where P is transition matrix, U is uncertainty
```

### Operator Decomposition

```julia
# Build noise convolution operator: P_noisy = N * P_deterministic
build_noise_convolution_operator(noise, partition;
    transition_graph=nothing,
    periodic=false,
    enclosure_method=first_order_enclosure,
    max_iterations=30,
    ε=1e-6)
# Returns: N (noise operator)
```

### Enclosure Methods

```julia
zeroth_order_enclosure(F, box)         # Pure interval arithmetic
first_order_enclosure(F, box)          # Mean value form (tighter, recommended)
```

### Utilities

```julia
normalize_columns!(P)                  # Convert to stochastic matrix
estimate_sparsity(P)                   # Matrix statistics
box_measure(box)                       # Lebesgue measure
box_width(box)                         # Maximum diameter
bisect_box(box)                        # Split along widest dimension
uniform_noise_box(σ, N)               # Uniform noise [-σ, σ]^N
```

## Documentation

- **Complete implementation guide**: See [COMPLETE_IMPLEMENTATION.md](COMPLETE_IMPLEMENTATION.md)
- **Implementation summary**: See [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)
- **Tutorial notebook**: [notebooks/rigorous_ulam_showcase.ipynb](notebooks/rigorous_ulam_showcase.ipynb)
- **Performance comparison**: [notebooks/contracting_lorenz_comparison.ipynb](notebooks/contracting_lorenz_comparison.ipynb)

## Testing

All 218 tests passing:

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

Test coverage:
- Enclosure methods: 5 tests
- Partition utilities: 37 tests
- Adaptive refinement: 13 tests
- Sparse matrices: 20 tests
- Noise convolution: 29 tests
- **Domain restriction**: 128 tests (including iterative refinement)
- Boundary conditions: Integrated

## Performance

### Contracting Maps

For maps where the forward image is strictly contained in the domain:

| Method | Boxes | Graph NNZ | Speedup |
|--------|-------|-----------|---------|
| Uniform partition (depth 5) | 1024 | 1,048,576 | 1× |
| **Adaptive (diameter 0.03)** | ~50-100 | ~2,500-10,000 | **10-100×** |

### Speedup Sources

1. **Fewer boxes** (10-20× reduction): Only refines reachable set
2. **Greedy subdivision**: Efficient convergence to target diameter
3. **Graph sparsity** (10-100× reduction): Only tracks actual transitions
4. **Combined effect**: 100-1000× fewer rel_size computations!

### Typical Timings

- Adaptive partitioning + matrix: 1-10 minutes (contracting map, diameter 0.02-0.05)
- Uniform partition + matrix: 1-10 hours (same resolution)
- **ROI**: 10-100× speedup with same rigorous guarantees!

## Citation

If you use this package in research, please cite:

```bibtex
@software{RigorousUlam2D,
  title = {RigorousUlam2D.jl: Rigorous Transfer Operators for 2D Dynamical Systems},
  author = {Nisoli, Isaia},
  year = {2025},
  url = {https://github.com/orkolorko/RigorousUlam2D.jl}
}
```

## License

MIT License - see LICENSE file for details.

## References

- **Ulam's method**: Ulam (1960), "A Collection of Mathematical Problems"
- **Rigorous computation**: Dellnitz & Junge (2002), "Set oriented numerical methods"
- **Error bounds**: Galatolo, Nisoli (2014), "Rigorous computation of invariant measures"
- **Contracting Lorenz map**: Galatolo, Nisoli, Pacifico (2018), "Decay of Correlations for Contracting Lorenz Attractors"
