# RigorousUlam2D.jl - Implementation Notes

Complete implementation of rigorous transfer operator computation for 2D dynamical systems.

## Architecture

### Module Structure
```
src/
├── RigorousUlam2D.jl          # Main module with exports
├── partition_utils.jl          # Box operations (bisect, quadrisect, measures)
├── adaptive_refinement.jl      # Core rel_size algorithm
├── sparse_ulam.jl              # Transition matrix construction
├── noise_convolution.jl        # Noise convolution operator
└── boundary_conditions.jl      # Periodic BC support
```

### Core Algorithm

The Ulam method discretizes the transfer operator using interval arithmetic:

1. **Partition** domain into boxes
2. **Compute rel_size** for each box pair via adaptive subdivision
3. **Build matrix** P where P[i,j] = probability j→i
4. **Normalize** columns to stochastic matrix
5. **Analyze** eigenvalues for invariant measure

### Operator Composition

For noisy systems `x' = F(x) + ξ`:
```julia
P_det, U = build_transition_matrix(F, partition, ...)
N = build_noise_convolution_operator(noise, partition, ...)
P_noisy = N * P_det
```

## Key Features

### 1. Adaptive Partitioning with Forward Flagging

Only refines boxes reachable by dynamics:
- Start with coarse partition (initial_depth=2)
- For each box, compute F(box) ⊕ noise ⊕ margin
- Flag boxes intersecting the image
- Refine flagged boxes exceeding diameter criterion
- Repeat until convergence

Typical reduction: 10-100× fewer boxes, 100-1000× fewer matrix entries.

### 2. Transition Graph Optimization

Sparse boolean matrix tracking nonzero transitions:
- Built during adaptive partitioning
- Passed to matrix builder to skip zero entries
- Speedup: 10-100× for matrix construction

### 3. Rigorous Noise Convolution

Noise convolution uses adaptive subdivision (not direct Minkowski sum):
```julia
function compute_noise_overlap(source, target, noise; ε, max_iterations)
    noisy = source ⊕ noise
    if noisy ⊆ target: return 1.0  # Fully verified
    if noisy ∩ target = ∅: return 0.0  # No transition
    # Otherwise: subdivide source and recurse
end
```

Returns lower bound (verified transitions only) for rigorous approximation.

### 4. Periodic Boundary Conditions

Add `periodic=true` to wrap coordinates to torus:
```julia
build_adaptive_partition(F, domain, noise, periodic=true, ...)
build_transition_matrix(F, partition, periodic=true, ...)
build_noise_convolution_operator(noise, partition, periodic=true, ...)
```

Conservative: if wrapping creates disconnected set, return full domain.

## New Features (2026-01-03)

### 1. Grid-Aligned Noise Operator

**High-performance exact computation** when noise matches partition grid structure.

**Constraint**: `noise_width[i] = domain_width[i] / 2^k[i]` for integer k per dimension

**Benefits**:
- 10-100× faster than adaptive refinement (no subdivision needed)
- Exact geometric computation using grid alignment
- Supports anisotropic noise (different k per dimension)
- **Exact periodic BC**: column sums = 1.0 exactly (no numerical drift)

**Usage**:
```julia
# Grid-aligned noise: width = domain_width / 2^k
domain = IntervalBox(interval(0.0, 1.0), interval(0.0, 1.0))
noise = uniform_noise_box(1/64, 2)  # width = 1.0/64 = 1.0/2^6 ✓

partition, dyn_graph, topo_graph = build_adaptive_partition(F, domain, noise, ...)

# Fast exact computation
N = build_grid_aligned_noise_operator(noise, partition,
                                      topological_graph=topo_graph,
                                      periodic=true)
```

**How it works**:
- Validates grid alignment (errors if constraint violated)
- For periodic BC with wraparound: enumerates wrapped components explicitly
- Each component processed separately for exact overlap computation
- No subdivision needed - pure geometry

### 2. Incremental Topological Graph Building

**API Change**: `build_adaptive_partition` now returns **3 values** (breaking change)

**New signature**:
```julia
partition, dynamical_graph, topological_graph = build_adaptive_partition(...)
```

**Two types of graphs**:
- **Dynamical graph**: tracks transitions via F(x) - used by transition matrix builder
- **Topological graph**: tracks spatial proximity via noise - used by noise operator

**Topological graph**:
- Box i connected to j if `j ⊕ noise` intersects i
- Automatically includes multi-hop neighbors (2nd, 3rd order via Minkowski sum)
- Built incrementally during adaptive partitioning (no post-processing)

**Algorithm**:
```julia
# During adaptive partitioning:
1. Build initial topological_graph on coarse partition
2. For each refinement iteration:
   - Prune graph when removing unflagged boxes
   - Rebuild after subdivision (efficient on effective boxes only)
3. Return both graphs
```

**Migration**:
```julia
# OLD (2 returns):
partition, graph = build_adaptive_partition(F, domain, noise, ...)
N = build_noise_convolution_operator(noise, partition, transition_graph=graph)  # WRONG GRAPH!

# NEW (3 returns):
partition, dyn_graph, topo_graph = build_adaptive_partition(F, domain, noise, ...)
P, U = build_transition_matrix(F, partition, transition_graph=dyn_graph)  # ✓
N = build_noise_convolution_operator(noise, partition, transition_graph=topo_graph)  # ✓
```

## API Summary

### Enclosure Methods
```julia
zeroth_order_enclosure(F, box)      # Pure interval arithmetic
first_order_enclosure(F, box)        # Mean value form with AD
```

### Partition Building
```julia
# Uniform partition
partition = build_uniform_partition(domain, depth)

# Adaptive partition (returns 3 values!)
partition, dynamical_graph, topological_graph = build_adaptive_partition(
    F, domain, noise,
    max_diameter=0.01,
    initial_depth=2,
    safety_margin=1.1,
    enclosure_method=first_order_enclosure,
    periodic=false  # or true for torus topology
)
```

### Matrix Construction
```julia
# Deterministic operator
P_det, U = build_transition_matrix(
    F, partition,
    transition_graph=dynamical_graph,
    enclosure_method=first_order_enclosure,
    max_iterations=8, ε=1e-4,
    periodic=false
)

# Noise convolution operator (adaptive method)
N = build_noise_convolution_operator(
    noise, partition,
    transition_graph=topological_graph,
    ε=1e-4,
    periodic=false
)

# OR: Grid-aligned noise operator (fast exact method)
N = build_grid_aligned_noise_operator(
    noise, partition,
    topological_graph=topological_graph,
    periodic=false
)

# Compose
P_noisy = N * P_det
normalize_columns!(P_noisy)
```

## Removed Deprecated Methods

### Annealed Operator Functions (Removed 2026-01-02)

**Why removed**: Violated operator decomposition architecture.

Old approach (WRONG):
```julia
# Computed F(x) + noise together - annealed operator
P, U = build_transition_matrix(F, partition, noise, ...)
```

New approach (CORRECT):
```julia
# Separate dynamics and noise operators
P_det, U = build_transition_matrix(F, partition, ...)
N = build_noise_convolution_operator(noise, partition, ...)
P_noisy = N * P_det
```

**Benefits**:
- Mathematically correct operator composition
- Enables reuse: compute P once, compose with different N
- Rigorous: each operator has verified lower bounds
- Clear separation of concerns

Removed functions:
- `noisy_rel_size` (from `noise_convolution.jl`)
- `noisy_rel_size_periodic` (from `boundary_conditions.jl`)

### Domain Restriction Functions (Removed during refactoring)

Forward flagging in `build_adaptive_partition` superseded separate domain restriction:

Old workflow:
```julia
eff_domain = compute_effective_domain(F, domain, noise, ...)
partition = build_uniform_partition(eff_domain, depth)
P, U = build_transition_matrix(F, partition, ...)
```

New workflow (better):
```julia
partition, graph = build_adaptive_partition(F, domain, noise, max_diameter=0.01)
P, U = build_transition_matrix(F, partition, transition_graph=graph, ...)
```

Removed functions:
- `compute_effective_domain`
- `compute_effective_domain_boxes`
- `compute_effective_partition`
- `create_restricted_partition`
- `create_partition_from_boxes`

## Performance Tips

1. **Use adaptive partitioning** for contracting/dissipative systems
2. **Pass correct graphs** to matrix builders:
   - `dynamical_graph` → `build_transition_matrix`
   - `topological_graph` → `build_noise_convolution_operator` or `build_grid_aligned_noise_operator`
3. **Use grid-aligned noise** when possible for 10-100× speedup:
   - Choose `noise_width = domain_width / 2^k`
   - Supports anisotropic noise (different k per dimension)
   - Exact periodic BC (column sums = 1.0 exactly)
4. **Use diameter control** for direct error bounds
5. **Start with initial_depth=2** and let refinement adapt
6. **Use first_order_enclosure** for nonlinear maps

## Examples

See `examples/` directory for complete workflows:
- `01_contracting_map.jl` - Simple contracting map
- `02_henon_periodic.jl` - Hénon map with periodic BC
- `03_operator_decomposition.jl` - Demonstrates P_noisy = N * P
- `04_diameter_error_control.jl` - Error control via diameter
- `05_contracting_lorenz.jl` - Complex contracting map with visualization
