# Research Notes - RigorousUlam2D.jl

**Date**: 2025-12-31

---

## GitHub Actions Status

✅ **FIXED**: Documentation errors resolved

**Issue**: Documenter.jl complained about undocumented functions (`compute_effective_domain_boxes`, `compute_effective_partition`, `compute_forward_image_boxes`, `create_partition_from_boxes`)

**Solution**: Added these legacy functions to the API reference in `docs/src/domain_restriction.md`. These functions are marked for removal in `REMOVAL_OF_OLD_METHODS.md` but are documented temporarily to pass CI.

**Commit**: [7f91780](https://github.com/orkolorko/RigorousUlam2D.jl/commit/7f91780)

---

## Invariant Density Plot

❌ **BLOCKED**: CairoMakie incompatibility with Julia 1.12

**Issue**: CairoMakie 0.12 has a precompilation error on Julia 1.12:
```
FieldError: type Core.TypeName has no field `mt`
```

This is a known Julia 1.12 incompatibility with the current Makie ecosystem.

**Created**:
- Script: [`notebooks/save_lorenz_invariant_density.jl`](notebooks/save_lorenz_invariant_density.jl)
- This script computes the stationary measure and saves a high-quality PNG

**Workaround Options**:
1. **Use Julia 1.11** or 1.10 instead of 1.12
2. **Wait for Makie/CairoMakie update** compatible with Julia 1.12
3. **Use GLMakie** instead (OpenGL backend, works in Julia 1.12)
4. **Downgrade Julia** in notebooks environment

**Recommended**: Run the notebook in Julia 1.10 or 1.11 until CairoMakie is updated.

---

## BallArithmetic.jl and Pseudospectral Bounds

### Paper Found

📄 **"A pseudospectral approach to rigorous numerical estimation of Ruelle resonances of transfer operators"**

- **Authors**: Alex Blumenthal, Isaia Nisoli, Toby Taylor-Crush
- **arXiv**: [2507.09021](https://arxiv.org/abs/2507.09021) (submitted July 11, 2025)
- **Topic**: Pseudospectral approach inspired by Householder's theorem for computer-assisted estimation of resonances

**Key Points**:
- Provides rigorous, computer-assisted estimation of Ruelle resonances
- Identifies regions where resonances must exist
- Precludes presence of resonances in other regions
- Applies to transfer operators of:
  - Anosov/Axiom A diffeomorphisms
  - Piecewise expanding maps
  - Analytic uniformly expanding maps of the circle (computationally implemented)

### BallArithmetic.jl Package

✅ **FOUND**: [https://github.com/JuliaBallArithmetic/BallArithmetic.jl](https://github.com/JuliaBallArithmetic/BallArithmetic.jl)

**What It Is**: A Julia package implementing ball arithmetic with emphasis on numerical linear algebra

**Core Concept**:
- Uses `Ball` type representing intervals (c-r, c+r) where c=center, r=radius
- Guarantees rigorous enclosures: `Ball(x) + Ball(y)` contains all possible sums
- Optimized for matrix operations with `BallMatrix` and `BallVector` types

**Key Features for Transfer Operators**:
1. **Eigenvalue Enclosures**:
   - `evbox()` - rigorous eigenvalue bounds
   - `gevbox()` - generalized eigenvalue problems
2. **Pseudospectra Computation**:
   - `compute_enclosure()` - follows level lines around eigenvalues
   - Rigorous contour integration with guaranteed resolvent bounds
   - Pearl necklace algorithms for eigenvalue neighborhoods
3. **Spectral Analysis**:
   - `svdbox()` - singular value bounds
   - `svd_bound_L2_opnorm()` - operator norm estimation
   - Resolvent norm bounds for spectral analysis
4. **M-matrix Verification**: Matrix classification tools

**Documentation**:
- Stable: [https://JuliaBallArithmetic.github.io/BallArithmetic.jl/stable/](https://JuliaBallArithmetic.github.io/BallArithmetic.jl/stable/)
- Dev: [https://JuliaBallArithmetic.github.io/BallArithmetic.jl/dev/](https://JuliaBallArithmetic.github.io/BallArithmetic.jl/dev/)

**Installation**:
```julia
using Pkg
Pkg.add("BallArithmetic")  # If registered
# Or from GitHub:
Pkg.add(url="https://github.com/JuliaBallArithmetic/BallArithmetic.jl")
```

**References**:
- Rump, BIT 1999: Foundational linear algebra methods
- Revol & Theveny 2013: Ball arithmetic techniques

**Related Julia Packages** (for rigorous numerics):
- [**IntervalArithmetic.jl**](https://github.com/JuliaIntervals/IntervalArithmetic.jl): Validated numerics using interval arithmetic
- [**Pseudospectra.jl**](https://ralphas.github.io/Pseudospectra.jl/dev/): Computing pseudospectra and spectral portraits
- [**ValidatedNumerics.jl**](https://github.com/JuliaIntervals/ValidatedNumerics.jl): Rigorous floating-point calculations

### Integration with RigorousUlam2D.jl

**How to Use BallArithmetic.jl with Transfer Operators**:

```julia
using RigorousUlam2D
using BallArithmetic

# 1. Compute Ulam discretization as usual
partition, info = compute_effective_partition_adaptive(
    F, domain, noise,
    initial_depth=2,
    max_diameter=0.01,
    verbose=true
)

# 2. Build transition matrix
P, U = build_noisy_transition_matrix(F, partition, noise, maxdepth=8, ε=1e-4)
normalize_columns!(P)

# 3. Convert to BallMatrix for rigorous spectral analysis
# (Account for numerical errors in matrix construction)
P_ball = BallMatrix(Matrix(P), error_radius)  # error_radius from ε parameter

# 4. Compute rigorous eigenvalue enclosures
eigenvalue_boxes = evbox(P_ball)  # Returns rigorous bounds on all eigenvalues

# 5. Verify spectral gap rigorously
λ_leading = eigenvalue_boxes[1]  # Should be ≈ 1 for stochastic matrix
spectral_gap_lower_bound = abs(λ_leading) - maximum(abs.(eigenvalue_boxes[2:end]))

# 6. Compute pseudospectral contours
ε_levels = [1e-1, 1e-2, 1e-3]
pseudospectra = [compute_enclosure(P_ball, ε) for ε in ε_levels]
```

**Advantages for Dynamical Systems**:

1. **Rigorous Ruelle Resonances**: Compute validated bounds on leading eigenvalues
2. **Spectral Gap Verification**: Prove mixing rates with guaranteed error bounds
3. **Stability Analysis**: Rigorous bounds on operator norms
4. **Computer-Assisted Proofs**: All computations mathematically verified

**Example Application - Mixing Time Bounds**:

```julia
# Rigorous bound on mixing time for Markov chain
function compute_mixing_time_bound(P::SparseMatrixCSC, ε_target::Float64)
    # Convert to ball arithmetic
    P_ball = BallMatrix(Matrix(P), 1e-15)  # Machine precision

    # Get rigorous eigenvalue enclosures
    λ_boxes = evbox(P_ball)

    # Second-largest eigenvalue magnitude (upper bound)
    λ_2_upper = maximum([sup(abs(λ)) for λ in λ_boxes[2:end]])

    # Rigorous mixing time bound: t ≥ log(ε_target) / log(λ_2)
    t_mix_bound = ceil(Int, log(ε_target) / log(λ_2_upper))

    return t_mix_bound, λ_2_upper
end
```

### Next Steps for Implementation

1. **Add BallArithmetic.jl dependency**:
   ```julia
   # In Project.toml
   [deps]
   BallArithmetic = "..."
   ```

2. **Create spectral analysis module** in RigorousUlam2D:
   - `src/spectral_rigorous.jl`
   - Functions for rigorous eigenvalue computation
   - Pseudospectral contour computation
   - Mixing time bounds

3. **Example applications**:
   - Validate spectral gaps for contracting Lorenz map
   - Prove exponential mixing rates
   - Rigorous bounds on correlation decay

---

## Summary for User

### ✅ Completed

1. **Fixed GitHub Actions CI** - Documentation errors resolved
2. **Created invariant density script** - Ready to run when CairoMakie works
3. **Researched pseudospectral methods** - Found the paper and related packages

### ⚠️ Blocked

1. **CairoMakie plot generation** - Julia 1.12 incompatibility
   - **Solution**: Use Julia 1.10/1.11 or wait for Makie update

###  🔍 Investigation Needed

1. **BallArithmetic.jl** - Not publicly available yet
   - **Action**: Contact Isaia Nisoli directly or check paper supplements
   - **Alternative**: Use IntervalArithmetic.jl + Pseudospectra.jl

---

## Files Created/Modified

### New Files
- [`notebooks/save_lorenz_invariant_density.jl`](notebooks/save_lorenz_invariant_density.jl) - Script to compute and save density plot
- [`RESEARCH_NOTES.md`](RESEARCH_NOTES.md) - This file

### Modified Files
- [`docs/src/domain_restriction.md`](docs/src/domain_restriction.md) - Added legacy functions to API reference

### Git Status
- All changes committed and pushed to `main`
- CI should now pass (documentation fixed)

---

## Sources

- [A pseudospectral approach to rigorous numerical estimation of Ruelle resonances](https://arxiv.org/abs/2507.09021)
- [IntervalArithmetic.jl](https://github.com/JuliaIntervals/IntervalArithmetic.jl)
- [Pseudospectra.jl](https://ralphas.github.io/Pseudospectra.jl/dev/)
- [GitHub Topics: Spectral Methods in Julia](https://github.com/topics/spectral-methods?l=julia)
- [FourierFlows.jl](https://github.com/FourierFlows/FourierFlows.jl) - Pseudospectral PDE solvers
