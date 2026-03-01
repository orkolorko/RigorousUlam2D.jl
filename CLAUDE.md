# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RigorousUlam2D.jl is a Julia package for computing mathematically verified bounds on transfer operators of 2D dynamical systems using the Ulam method with interval arithmetic. It provides guaranteed enclosures via `IntervalArithmetic.jl` and `ForwardDiff.jl`.

## Common Commands

```bash
# Run tests
julia --project=. -e 'using Pkg; Pkg.test()'

# Run a single test file
julia --project=. test/test_spectral_analysis.jl

# Run an example (examples use their own Project.toml)
julia --project=examples examples/01_contracting_map.jl

# Interactive REPL
julia --project=.

# Format code (uses SciML style)
julia -e 'using JuliaFormatter; format("src/")'
```

## Code Style

Uses JuliaFormatter with `style = "sciml"`. Key conventions:
- 4-space indentation
- No trailing whitespace
- Matrix convention: `P[i,j]` = transition probability from box j to box i (column-stochastic)

## Architecture

### Operator Composition Pattern

The central design is **P_noisy = N * P_det** (operator decomposition):

1. **Adaptive partition** (`build_adaptive_partition`) — returns `(partition, dynamical_graph, topological_graph)`. The graphs encode sparsity: which box pairs can have nonzero transitions.
2. **Deterministic transition matrix** (`build_transition_matrix`) — uses `dynamical_graph` to skip impossible pairs; calls `rel_size` for each candidate pair to get rigorous lower/upper bounds.
3. **Noise convolution operator** (`build_noise_convolution_operator` or `build_grid_aligned_noise_operator`) — uses `topological_graph`; grid-aligned version is 10-100x faster when noise width divides domain evenly.
4. **Compose**: `P = N * P_det` then `normalize_columns!(P)`.

### Enclosure Methods

- `zeroth_order_enclosure(F, box)` — pure interval arithmetic evaluation
- `first_order_enclosure(F, box)` — mean value form via ForwardDiff (tighter bounds, preferred default)

### Key Type

`UlamPartition{N,T,I}` wraps a `Vector{IntervalBox{N,T,I}}` with the domain. All matrices are `SparseMatrixCSC`.

### Module Organization (src/)

| File | Responsibility |
|------|---------------|
| `partition_utils.jl` | Box ops, bisection, `build_uniform_partition`, `build_adaptive_partition` |
| `adaptive_refinement.jl` | `rel_size` algorithm (priority queue refinement for transition bounds) |
| `sparse_ulam.jl` | `build_transition_matrix`, `normalize_columns!`, `UlamPartition` type |
| `noise_convolution.jl` | Minkowski sum, noise operators (general + grid-aligned) |
| `boundary_conditions.jl` | `wrap_to_torus`, `apply_periodic_bc` for periodic domains |
| `spectral_analysis.jl` | `operator_norm_on_V`, `rigorous_error_bound`, spectral gap estimation |

### Test Organization

8 test files in `test/` (~218 tests) covering all modules. `test/runtests.jl` includes them all.

### Examples

5 examples in `examples/` (01-05), each standalone. They use CairoMakie for visualization and Arpack for eigenvalues — these deps live in `examples/Project.toml`.

## Important Implementation Details

- `build_adaptive_partition` returns **3 values** (partition, dynamical_graph, topological_graph), not 2.
- `build_transition_matrix` raises an error if noise is passed — use operator composition instead.
- Grid-aligned noise requires `noise_width[i] = domain_width[i] / 2^k[i]` (validated at construction).
- `wrap_to_torus` returns full domain interval when wrapping is ambiguous (conservative).
- Plotting large partitions: use vectorized `Rect2f` + single `poly!` call, not per-box loops (hangs for >1000 boxes).
- Julia compat: `>= 1.12`.
