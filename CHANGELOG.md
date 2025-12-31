# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added

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
