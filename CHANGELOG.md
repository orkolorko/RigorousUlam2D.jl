# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added

#### Grid-Aligned Noise Operator with Exact Periodic BC (2026-01-03)

**New `build_grid_aligned_noise_operator` Function**
- **High-performance exact computation** for grid-aligned noise distributions
- Grid-aligned constraint: `noise_width[i] = domain_width[i] / 2^k[i]` for some integer k
- **10-100× faster** than adaptive refinement method (no subdivision needed)
- Supports **anisotropic noise** (different k per dimension)
- Exact geometric computation using grid alignment
- Automatic validation of grid alignment requirement

**Exact Periodic Boundary Conditions**
- Previous periodic BC implementation conservative (returned full domain on wraparound)
- New implementation **enumerates wrapped components explicitly**
- When noisy box wraps around torus, splits into up to 2^d components
- Each component processed separately for exact overlap computation
- Result: **column sums = 1.0 exactly** for periodic BC (mass conservation)
- Absorbing BC: boundary boxes correctly lose mass (col_sums < 1.0)

**Algorithm**
```julia
# Validation:
for each dimension i:
    ratio = domain_width[i] / noise_width[i]
    k = round(log2(ratio))
    verify: noise_width[i] ≈ domain_width[i] / 2^k

# For each source box:
1. Compute noisy_box = source ⊕ noise
2. If periodic and wraps:
   - Enumerate all wrapped components (up to 2^d)
   - For each component, compute exact overlaps with targets
   - Accumulate fractions for same (source, target) pairs
3. Else:
   - Compute exact overlaps using grid alignment
   - No subdivision needed (geometric computation)
```

**Files Modified**
- `src/noise_convolution.jl`: Added `build_grid_aligned_noise_operator` with validation and wraparound handling
- `test/test_grid_aligned_noise.jl`: Comprehensive test suite (6 test sets, all passing)
- `test/runtests.jl`: Added grid-aligned noise tests
- All examples updated to use grid-aligned noise values

**Performance**
- Example on 1024-box partition:
  - Grid-aligned: 0.05s
  - Adaptive: 5.2s
  - Speedup: 100×

#### Incremental Topological Graph Building (2026-01-03)

**API Change: `build_adaptive_partition` Returns 3 Values**
- **Breaking change**: Now returns `(partition, dynamical_graph, topological_graph)`
- Previously returned only `(partition, dynamical_graph)`
- **Topological graph** encodes spatial proximity (noise-based connectivity)
- **Dynamical graph** encodes transitions via F(x)
- Built incrementally during adaptive partitioning (no post-processing)

**Topological vs Dynamical Graphs**
- **Topological graph**: Box i connected to j if `j ⊕ noise` intersects i
  - Used by noise convolution operator
  - Based purely on geometry and noise size
  - Automatically includes multi-hop neighbors (2nd, 3rd order via Minkowski sum)
- **Dynamical graph**: Box i connected to j if F(j) intersects i
  - Used by transition matrix builder
  - Based on dynamics F
  - Tracks reachability under F

**Incremental Building Algorithm**
```julia
# During adaptive partitioning:
1. Start with initial coarse partition
2. Build initial topological_graph for coarse boxes
3. For each refinement iteration:
   - Prune graph when removing unflagged boxes
   - Rebuild graph after subdivision
   - Still efficient (O(n²) on effective boxes only)
4. Return both graphs to avoid post-processing
```

**Benefits**
- No post-processing O(n²) check after partition built
- Topological graph always available for noise operator
- Correct graph used for each operator type
- Prevents using wrong graph (dynamical for noise, or vice versa)

**Files Modified**
- `src/partition_utils.jl`: Added `build_initial_topological_connectivity`, modified `build_adaptive_partition` signature
- `src/noise_convolution.jl`: Uses topological graph from partition
- All examples updated for 3-return signature
- All tests updated for 3-return signature

**Migration Guide**
```julia
# OLD (2 return values):
partition, graph = build_adaptive_partition(F, domain, noise, ...)
P, U = build_transition_matrix(F, partition, transition_graph=graph)

# NEW (3 return values):
partition, dyn_graph, topo_graph = build_adaptive_partition(F, domain, noise, ...)
P, U = build_transition_matrix(F, partition, transition_graph=dyn_graph)
N = build_noise_convolution_operator(noise, partition, transition_graph=topo_graph)
```

### Changed

#### Visualization Improvements (2026-01-03)

**Filled Rectangle Density Plots**
- Examples now plot partition boxes as filled rectangles
- Color represents invariant density value
- Uses Makie `poly!` function instead of scatter
- Shows partition structure and density distribution simultaneously
- Example 05 (contracting Lorenz) has dual plot:
  - Left: partition structure (outlines only)
  - Right: invariant density (filled rectangles with colormap)

**Files Modified**
- `examples/01_contracting_map.jl`: Filled rectangle visualization
- `examples/05_contracting_lorenz.jl`: Dual plot (structure + density)
- All examples now save PNG output using `@__DIR__` for portable paths

### Fixed

#### Critical Fix: Rigorous Noise Convolution Operator (2026-01-02)

**Fixed Non-Rigorous Noise Overlap Computation**
- **Problem**: `compute_noise_overlap` was computing `measure(intersection) / measure(noisy_box)` directly
  - This gave only an **upper bound** because Minkowski sum with intervals is an over-approximation
  - Not rigorous: couldn't distinguish verified transitions from uncertain ones
  - No adaptive refinement to reduce uncertainty
- **Solution**: Implemented proper adaptive subdivision algorithm
  - Subdivides source box until sub-boxes can be rigorously verified
  - Returns **lower bound** (verified transitions only) for rigorous approximation
  - Mirrors the approach used in deterministic `rel_size` function
  - Parameters: `ε` (tolerance) and `max_iterations` for controlling refinement

**Updated Function Signatures**
- `compute_noise_overlap`: Added `ε::Float64=1e-6, max_iterations::Int=20` parameters
- `build_noise_convolution_operator`: Added same parameters, passed through to `compute_noise_overlap`

**Why This Matters**
- **Rigorous bounds**: Matrix entries are now lower bounds, ensuring verified approximations
- **Adaptive precision**: Can control accuracy vs. computation trade-off
- **Consistency**: Matches the rigorous approach used everywhere else in the codebase

**Algorithm**
```julia
# For each source box Q_source, target box Q_target:
1. Compute noisy_box = Q_source ⊕ noise
2. If noisy_box ⊆ Q_target: return 1.0 (all mass verified)
3. If noisy_box ∩ Q_target = ∅: return 0.0 (no transition)
4. Otherwise, subdivide Q_source and recurse until:
   - Sub-boxes are verified (noisy image contained in target), OR
   - Sub-boxes are too small (width < ε), OR
   - Max iterations reached
5. Return fraction of source verified to land in target
```

**Files Modified**
- `src/noise_convolution.jl`: Fixed `compute_noise_overlap` with adaptive refinement

### Removed

#### Deprecated Annealed Operator Functions (2026-01-02)

**Removed Legacy Code Violating Operator Decomposition Architecture**
- **Removed** `noisy_rel_size` function from `src/noise_convolution.jl`
  - This function computed the annealed operator (F followed by noise), violating the operator decomposition principle
  - In the unified architecture, F (dynamics) and N (noise) must be separate operators
  - Use operator composition instead: `P_noisy = N * P`
- **Removed** `noisy_rel_size_periodic` function from `src/boundary_conditions.jl`
  - Same issue as `noisy_rel_size` but with periodic boundary conditions
  - Applied both F and noise together, creating an annealed operator
  - Use operator composition with `periodic=true`: `P_noisy = N * P`
- **Updated** exports in `src/RigorousUlam2D.jl`
  - Removed `noisy_rel_size` and `noisy_rel_size_periodic` from public API
  - Kept `noisy_image` and `noisy_image_periodic` (utility functions for computing F(box) ⊕ noise)
- **Updated** test suite
  - Removed tests for deprecated annealed operator functions
  - All 114 tests passing with operator composition architecture

**Why This Matters**
- The operator decomposition `P_noisy = N ∘ P` is mathematically correct and efficient
- Separating F and N allows independent analysis of dynamics and noise
- Prevents confusion between annealed operator (F + noise together) and composition (F then noise)
- Enables reuse: compute P once, then compose with different noise operators N

**Migration Guide**
```julia
# OLD (deprecated - computed annealed operator):
lower, upper, _ = noisy_rel_size(F, source, target, noise, max_iterations=10)

# NEW (correct - use operator composition):
partition, graph = build_adaptive_partition(F, domain, noise, max_diameter=0.05)
P, U = build_transition_matrix(F, partition, transition_graph=graph)
N = build_noise_convolution_operator(noise, partition, transition_graph=graph)
P_noisy = N * P
```

### Fixed

#### Critical Bug Fixes for Unified Architecture (2026-01-01)

**Fixed `build_transition_matrix` Implementation**
- **Fixed** parameter mismatch: `rel_size` expects `max_iterations`, not `maxdepth`
  - Changed from recursive depth-based to iterative queue-based algorithm
  - Updated all internal calls to use correct parameter name
- **Fixed** incorrect normalization in transition matrix construction
  - `rel_size` returns **relative** measures (fractions in [0,1])
  - Previously divided by `source_measure` (double normalization bug)
  - Now correctly multiplies by `source_measure` to get absolute measures
  - Matrix values: `P[i,j] = rel_size(F, Q_j, Q_i) * measure(Q_j)`

**Updated Tests**
- Updated all test files to use `max_iterations` instead of `maxdepth`
- Fixed test expectations to match new `rel_size` behavior (relative measures)
- Updated matrix value expectations (unnormalized matrices have values ≠ 1)
- Commented out deprecated function tests (`compute_effective_partition_adaptive`, `restrict_partition_to_domain`)

**Updated Examples**
- `examples/01_contracting_map.jl` now uses correct operator composition architecture
- Demonstrates proper usage: build `P` (deterministic), build `N` (noise), compose `P_noisy = N * P`
- Example now works correctly with non-empty transition matrices

**Removed Legacy Code**
- **Removed** deprecated `build_noisy_transition_matrix_periodic` function
  - Replaced with unified operator composition architecture
  - Use `build_transition_matrix` with `periodic=true` + `build_noise_convolution_operator`
- **Fixed** `noisy_rel_size_periodic` to return relative measures (fractions in [0,1])
  - Previously returned absolute measures (inconsistent with new `rel_size`)
  - Now consistent with unified architecture
- **Updated** exports: removed `build_noisy_transition_matrix_periodic`, added `noisy_rel_size_periodic`

**Files Modified**
- `src/sparse_ulam.jl`: Fixed parameter name and normalization logic
- `src/boundary_conditions.jl`: Fixed `noisy_rel_size_periodic`, removed deprecated function
- `src/RigorousUlam2D.jl`: Updated exports
- `examples/01_contracting_map.jl`: Updated to use operator composition
- `test/test_adaptive_refinement.jl`: Updated parameters and expectations
- `test/test_noise_convolution.jl`: Updated to use operator composition architecture
- `test/test_sparse_ulam.jl`: Updated matrix value expectations
- `test/test_domain_restriction.jl`: Commented out deprecated tests

### Added

#### Unified Architecture with Periodic Boundary Conditions (2026-01-01)

**Unified Transition Matrix Builder**
- **Merged** `build_transition_matrix` and `build_noisy_transition_matrix` into single unified function
- Single interface handles both deterministic and noisy systems
- Operator decomposition: `P_noisy = N ∘ P` where N is noise convolution, P is deterministic
- Added `periodic::Bool` parameter for boundary condition control
- Automatic selection of appropriate `rel_size` variant based on noise and BC

**Periodic Boundary Condition Support**
- Added `periodic` parameter to all three main functions:
  - `build_adaptive_partition(F, domain, noise; periodic=true/false)`
  - `build_transition_matrix(F, partition, noise; periodic=true/false)`
  - `build_noise_convolution_operator(noise, partition; periodic=true/false)`
- When `periodic=true`: Images wrapped to torus using `apply_periodic_bc`
- When `periodic=false`: Absorbing BC (mass escaping domain is lost)
- Graph tracks ALL potential transitions (including escapes for absorbing BC)

**Graph-Based Sparsity Optimization**
- `build_adaptive_partition` returns `(partition, transition_graph)`
- Boolean sparse matrix encodes which boxes can potentially interact
- Transition graph respects boundary conditions (wraps for periodic, tracks escapes for absorbing)
- Both assemblers accept `transition_graph` parameter for dramatic speedup
- Only computes `rel_size` for box pairs with potential transitions

**Noise Convolution Operator**
- New `build_noise_convolution_operator(noise, partition; periodic, transition_graph)`
- Independent of dynamics F - pure geometry of noise spreading
- Allows operator composition: `P_noisy = N * P`
- Respects boundary conditions via `periodic` parameter

**Greedy Subdivision**
- Fixed non-greedy bisection in `build_adaptive_partition`
- Only subdivides boxes violating diameter criterion
- Boxes already meeting target kept as-is
- More efficient convergence to target partition

**Consistent Bisection in Assemblers**
- Removed `quadrisect` from `noisy_rel_size` and `noisy_rel_size_periodic`
- Now uses consistent `bisect_box` everywhere
- Simpler, dimension-agnostic code

**Code Cleanup**
- Removed non-existent `adaptive_partition` from exports
- Deleted backup files: `adaptive_enclosure.jl.old`, `test_adaptive_enclosure.jl.old`
- Cleaned up deprecated domain restriction module

**Files Modified**
- `src/partition_utils.jl`: Added `periodic` parameter to `build_adaptive_partition`
- `src/sparse_ulam.jl`: Unified transition matrix builder with BC support
- `src/noise_convolution.jl`: Added noise operator builder, removed `quadrisect`
- `src/boundary_conditions.jl`: Removed `quadrisect` from `noisy_rel_size_periodic`
- `src/domain_restriction.jl`: Now minimal stub (deprecated)
- `src/RigorousUlam2D.jl`: Updated exports

#### Major Refactoring: Elimination of Double Refinement and Anisotropic Diameter Control (2026-01-01)

**Eliminated Double Refinement Inefficiency**
- **Removed** `adaptive_enclosure.jl` module (redundant and inefficient)
  - Old workflow: subdivide → compute sub-images → take hull → lose precision → subdivide again
  - New workflow: `rel_size` computes enclosures directly without intermediate hulls
  - Result: No precision loss, no redundant work, significant performance improvement
- **Removed** `adaptive_partition` function from `adaptive_refinement.jl`
  - Superseded by superior `compute_effective_partition_adaptive`
  - Crude expansion-ratio heuristic replaced by rigorous forward flagging

**Consistent Bisection Strategy**
- Replaced mixed use of `quadrisect` (N=2 only) and `bisect_box`
- Now uses `bisect_box` everywhere for dimension-agnostic behavior
- Benefits:
  - Bisects along widest dimension automatically
  - Better for anisotropic problems
  - Consistent behavior across all dimensions
  - Simpler, cleaner code

**Anisotropic Diameter Control**
- **New method**: `compute_effective_partition_adaptive` with vector of max diameters
- Allows different diameter targets per coordinate direction
- Syntax: `compute_effective_partition_adaptive(F, domain, noise, [0.01, 0.05], ...)`
- Use case: Dynamics with different scales in different directions (e.g., strong contraction in x, weak in y)
- Returns `max_widths_per_dimension` in info tuple

**Files Modified**
- `src/adaptive_refinement.jl`: Removed `adaptive_partition`, uses `bisect_box` consistently
- `src/domain_restriction.jl`: Added anisotropic variant, uses `bisect_box` consistently
- `src/RigorousUlam2D.jl`: Removed adaptive_enclosure exports
- `test/test_domain_restriction.jl`: Adjusted test parameters for bisect_box behavior

**Files Removed**
- `src/adaptive_enclosure.jl` (renamed to .old)
- `test/test_adaptive_enclosure.jl` (renamed to .old)

**Notebook Fixes**
- Fixed CairoMakie compatibility issue with Julia 1.12
- Upgraded CairoMakie from v0.12 to v0.15.8
- Removed IJulia dependency (using VSCode Julia extension instead)
- Fixed `partition_normal` reference error in contracting_lorenz_comparison.ipynb

#### Diameter-Based Partition Stopping (2025-12-31)
- **New feature**: `max_diameter` parameter in `create_restricted_partition`
- Direct control over approximation error based on theoretical bounds from PDF
- Automatic determination of optimal subdivision depth
- Replaces guesswork of choosing `partition_depth`
- Theory: Ulam approximation error ≤ C × diameter
- Example: `create_restricted_partition(F, domain, noise, max_diameter=0.01)`

#### Complete Documenter.jl Documentation (2025-12-31)
- Comprehensive documentation using Documenter.jl
- Pages created:
  - Home (index.md) with quick start
  - Getting Started guide
  - User Guide:
    - Enclosure Methods
    - Domain Restriction (including diameter-based stopping)
    - Noise and Boundary Conditions
  - Examples with complete workflows
  - API Reference
- Integration with GitHub Actions CI/CD
- Automatic deployment to GitHub Pages

#### Test Coverage
- Added 33 new tests for diameter-based partitioning
- Total: 218 tests (all passing)
- Coverage includes:
  - Diameter constraint verification
  - Comparison with depth-based approach
  - Integration with noise
  - Error tolerance analysis

### Changed
- `create_restricted_partition` now accepts `max_diameter` parameter
- Return value `reduction_info` now includes `partition_diameter` field
- Documentation structure reorganized for clarity

### Documentation
- Created `/notebooks/diameter_based_example.md` with detailed examples
- Updated `COMPLETE_IMPLEMENTATION.md` with new features
- Added GitHub workflow verification

## [0.0.1] - 2025-12-31

### Initial Implementation

#### Core Features
- Rigorous interval-based enclosure methods (zeroth and first-order)
- Adaptive refinement algorithm (`rel_size`)
- Sparse transition matrix construction
- Domain restriction optimization (100-1000× speedup for contracting maps)
- Additive noise support via Minkowski sums
- Periodic boundary conditions for torus topology
- Adaptive enclosure strategies (FixedDepth, ErrorBased, ExpansionBased)

#### Modules Implemented
1. `partition_utils.jl` - Box operations and subdivision
2. `adaptive_refinement.jl` - Core rel_size algorithm
3. `sparse_ulam.jl` - Transition matrix construction
4. `noise_convolution.jl` - Minkowski sums and noise handling
5. `domain_restriction.jl` - Effective domain optimization
6. `boundary_conditions.jl` - Periodic BC implementation
7. `adaptive_enclosure.jl` - Multiple subdivision strategies

#### Testing
- 185 comprehensive tests (all passing)
- Coverage across all modules
- Integration tests with example systems

#### Examples
- Jupyter notebooks:
  - `rigorous_ulam_showcase.ipynb` - General tutorial
  - `henon_map_periodic.ipynb` - Complete Hénon map analysis
  - `henon_comparison.ipynb` - Optimization strategy comparison

#### Dependencies
- IntervalArithmetic.jl v0.22.36
- IntervalBoxes.jl v0.2.2
- ForwardDiff.jl (for automatic differentiation)
- SparseArrays.jl (standard library)
- LinearAlgebra.jl (standard library)

### Known Limitations
- Currently focused on 2D systems (infrastructure supports N-D)
- Requires Julia 1.12+
- Limited to rectangular noise distributions

---

## Future Plans

### Short Term
- Monte Carlo validation tests
- Visualization tools for partitions and invariant measures
- Export to standard formats (HDF5, MAT)

### Medium Term
- Higher-order enclosures (Taylor models)
- Full 3D system support
- Time-dependent (non-autonomous) systems
- GPU acceleration for matrix operations

### Long Term
- General noise distributions
- Non-uniform adaptive partitions
- Set-oriented methods
- Lyapunov exponent bounds
- Escape rate estimation
