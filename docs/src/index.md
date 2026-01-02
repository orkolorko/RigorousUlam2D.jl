```@meta
CurrentModule = RigorousUlam2D
```

# RigorousUlam2D.jl

**Rigorous computation of transfer operators for 2D dynamical systems using the Ulam method**

## Overview

RigorousUlam2D.jl provides mathematically rigorous tools for computing transfer (Perron-Frobenius) operators of 2D dynamical systems using interval arithmetic and the Ulam discretization method.

### Key Features

- ✅ **Mathematical Rigor**: All computations provide verified bounds using interval arithmetic
- ✅ **Computational Efficiency**: Domain restriction achieves 100-1000× speedup for contracting maps
- ✅ **Sparse Matrices**: Efficient storage and computation for large partitions
- ✅ **Additive Noise**: Built-in support for stochastic perturbations via Minkowski sums
- ✅ **Periodic Boundary Conditions**: Support for torus topology
- ✅ **Adaptive Methods**: Multiple adaptive strategies for tighter enclosures
- ✅ **Error Control**: Diameter-based stopping criteria directly tied to theoretical error bounds

## Quick Start

```julia
using RigorousUlam2D
using IntervalArithmetic, IntervalBoxes

# Define a dynamical system
F(x, y) = (1 - 1.4*x^2 + y, 0.3*x)  # Hénon map

# Setup domain and noise
domain = IntervalBox(interval(-1.0, 1.0), interval(-1.0, 1.0))
noise = uniform_noise_box(0.01, 2)

# Create optimized partition (with domain restriction)
partition, eff_domain, info = create_restricted_partition(
    F, domain, noise,
    max_diameter=0.01  # Error-based stopping
)

# Build rigorous transition matrix
P, U = build_noisy_transition_matrix_periodic(
    F, partition, noise, domain,
    max_iterations=8, ε=1e-4
)

# Normalize to stochastic matrix
normalize_columns!(P)

# Analyze spectrum
using LinearAlgebra
λ = eigvals(Matrix(P))
spectral_gap = 1 - abs(λ[2])
```

## Installation

```julia
using Pkg
Pkg.add("RigorousUlam2D")
```

Or from the Julia REPL:
```julia
] add RigorousUlam2D
```

## Package Structure

```
RigorousUlam2D.jl
├── Enclosure Methods        # Interval-based image computation
├── Partition Utilities      # Box operations and subdivision
├── Adaptive Refinement      # Core rel_size algorithm
├── Sparse Matrices          # Transition operator construction
├── Noise Convolution        # Minkowski sums for additive noise
├── Domain Restriction       # Major optimization for contracting maps
├── Boundary Conditions      # Periodic BC for torus
└── Adaptive Enclosure       # Multiple subdivision strategies
```

## Documentation Contents

```@contents
Pages = [
    "getting_started.md",
    "enclosures.md",
    "domain_restriction.md",
    "noise_and_boundaries.md",
    "examples.md",
    "api.md"
]
Depth = 2
```

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

## Index

```@index
```
