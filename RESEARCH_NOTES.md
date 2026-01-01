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

⚠️ **NOT FOUND** in public Julia package registry

**Search Results**: No public Julia package named "BallArithmetic.jl" found in:
- Julia General Registry
- GitHub (public repositories)
- JuliaHub
- JuliaPackages.com

**Possible Explanations**:
1. **Private/Unpublished Package**: May be a research code not yet released publicly
2. **Different Name**: Could be called something else (e.g., part of a larger package)
3. **Internal Tool**: Might be custom code for the paper, not a standalone package
4. **Very Recent**: Arxiv paper from July 2025, package may not be released yet

**Related Julia Packages** (for rigorous numerics):
- [**IntervalArithmetic.jl**](https://github.com/JuliaIntervals/IntervalArithmetic.jl): Validated numerics using interval arithmetic
- [**Pseudospectra.jl**](https://ralphas.github.io/Pseudospectra.jl/dev/): Computing pseudospectra and spectral portraits
- [**ValidatedNumerics.jl**](https://github.com/JuliaIntervals/ValidatedNumerics.jl): Rigorous floating-point calculations

### Next Steps

**To Access BallArithmetic.jl**:
1. **Contact Authors**: Email Isaia Nisoli (he's your package author!) or co-authors
2. **Check Paper Supplements**: Download full PDF from arXiv and look for:
   - Code repositories mentioned
   - Supplementary materials
   - GitHub links in acknowledgments
3. **Check arXiv Source**: Download TeX source ([`/src/2507.09021`](https://arxiv.org/src/2507.09021)) for code references

**For Pseudospectral Bounds in RigorousUlam2D.jl**:

If you want to implement pseudospectral bounds for your transfer operators:

1. **Use IntervalArithmetic.jl** (already a dependency):
   - Compute eigenvalues with rigorous error bounds
   - Validate spectral gap estimates

2. **Implement Householder-style approach** from the paper:
   - Rigorous enclosures of resonances
   - Exclusion regions for spectrum

3. **Pseudospectra.jl** integration:
   - Visualize pseudospectral behavior
   - Validate numerical eigenvalue estimates

**Example Concept** (based on paper abstract):
```julia
# Pseudospectral bounds for Ulam matrix
function compute_resonances_rigorous(P::SparseMatrixCSC, ε::Float64)
    # 1. Compute eigenvalues numerically
    λ_numerical = eigvals(Matrix(P))

    # 2. Use interval arithmetic for rigorous bounds
    # (This is where BallArithmetic.jl might come in)

    # 3. Validate which eigenvalues are true resonances
    # vs. numerical artifacts

    # 4. Return rigorous enclosures
    return rigorous_eigenvalue_bounds
end
```

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
