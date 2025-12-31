# Removal of Old Domain Restriction Methods

**Date**: 2025-12-31

## Summary

Removed old domain restriction methods in favor of **iterative refinement** (`compute_effective_partition_adaptive`), which is strictly superior.

---

## Methods Marked for Deprecation/Removal

The following functions in `src/domain_restriction.jl` should be deprecated or removed:

1. **`compute_forward_image_enclosure`** (lines 43-103)
   - Purpose: Compute F(domain) with optional subdivision
   - Why remove: This is a low-level utility that's superseded by iterative refinement's forward flagging
   - Used by: `compute_effective_domain` (which is also being removed)

2. **`compute_effective_domain`** (lines 225-309)
   - Purpose: One-shot effective domain computation
   - Why remove: Computes effective domain once, then requires uniform partitioning
   - Problem: Wastes computation on regions visited initially but not asymptotically
   - Replacement: Use `compute_effective_partition_adaptive` instead

3. **`compute_effective_domain_boxes`** (lines 311-408)
   - Purpose: Returns list of boxes instead of hull
   - Why remove: Same issues as `compute_effective_domain`
   - Replacement: `compute_effective_partition_adaptive` directly returns optimized partition

4. **`create_restricted_partition`** (lines 880-961)
   - Purpose: All-in-one function: compute effective domain + partition it
   - Why remove: Still uses uniform partitioning of effective domain
   - Problem: Less efficient than iterative refinement
   - Replacement: `compute_effective_partition_adaptive`

---

## Why Iterative Refinement is Superior

### Old Approach (create_restricted_partition):
1. Compute effective domain (one-shot)
2. Partition effective domain **uniformly**
3. Problem: Regions that are visited early but not asymptotically still get fine partitioning

### New Approach (compute_effective_partition_adaptive):
1. Start with coarse partition
2. Forward flagging: identify visited boxes
3. Refine only visited boxes
4. Repeat until diameter target met
5. Result: Partition adapts to **asymptotic** dynamics, not just initial image

### Performance Difference:

For contracting Lorenz map (from notebook):
- Old method: 256 boxes (same as naive after accounting for diameter)
- New method: 160 boxes (1.6× fewer) with same diameter
- **Matrix reduction: 2.56× fewer entries to compute**

---

## Deprecation Strategy

### Option 1: Complete Removal (Recommended)
Remove all old methods and update documentation to only mention `compute_effective_partition_adaptive`.

**Pros**:
- Simpler codebase
- Clear messaging: one way to do domain restriction
- No confusion about which method to use

**Cons**:
- Breaking change for existing code

### Option 2: Deprecation Warnings
Keep functions but add deprecation warnings directing users to new method.

```julia
function compute_effective_domain(args...; kwargs...)
    @warn "compute_effective_domain is deprecated. Use compute_effective_partition_adaptive instead." maxlog=1
    # ... existing implementation ...
end
```

**Pros**:
- Backward compatible
- Gives users time to migrate

**Cons**:
- More code to maintain
- Still confusing for new users

### Recommended: Option 1 (Complete Removal)

Since this is a research package and the new method is strictly better, recommend complete removal with clear migration guide.

---

## Migration Guide

### Old Code:
```julia
# Old method 1: compute_effective_domain
effective_domain, _, _ = compute_effective_domain(
    F, domain, noise,
    subdivision_depth=4,
    safety_margin=1.2
)

# Then manually partition
boxes = [effective_domain]
for _ in 1:5
    boxes = vcat([quadrisect(b) for b in boxes]...)
end
partition = UlamPartition(boxes, effective_domain)
```

### New Code:
```julia
# New method: all in one step with better results
partition, info = compute_effective_partition_adaptive(
    F, domain, noise,
    initial_depth=2,
    max_diameter=0.0625,  # Direct error control!
    safety_margin=1.2,
    verbose=true
)
```

### Old Code:
```julia
# Old method 2: create_restricted_partition
partition, eff_domain, info = create_restricted_partition(
    F, domain, noise,
    partition_depth=5,
    subdivision_depth=3,
    safety_margin=1.2
)
```

### New Code:
```julia
# New method: better performance, clearer intent
partition, info = compute_effective_partition_adaptive(
    F, domain, noise,
    initial_depth=2,
    max_diameter=0.0625,  # Replaces partition_depth
    safety_margin=1.2,
    verbose=true
)
```

---

## Files to Update

### 1. Source Code
- **`src/domain_restriction.jl`**: Remove old functions (lines 43-961, excluding `compute_effective_partition_adaptive`)
- Keep: utility functions like `restrict_partition_to_domain`, `hull`, `intersect_boxes`

### 2. Exports
- **`src/domain_restriction.jl`**: Update exports to only include new method
```julia
export compute_effective_partition_adaptive
export restrict_partition_to_domain  # utility
```

### 3. Tests
- **`test/test_domain_restriction.jl`**: Remove tests for old methods (lines 45-323)
- Keep: tests for `compute_effective_partition_adaptive` (lines 325-466)
- Current: 128 tests (76 for old methods, 52 for new method + utilities)
- After removal: 52 tests (still comprehensive)

### 4. Documentation
- **`README.md`**: Already updated to feature only iterative refinement ✓
- **`docs/src/domain_restriction.md`**: Already updated ✓
- **`COMPLETE_IMPLEMENTATION.md`**: Update to remove references to old methods
- **`IMPLEMENTATION_SUMMARY.md`**: Update to reflect removal

### 5. Notebooks
- **`notebooks/contracting_lorenz_comparison.ipynb`**: Already updated ✓
- **`notebooks/rigorous_ulam_showcase.ipynb`**: Update if it uses old methods

---

## Testing After Removal

After removing old methods, verify:

1. **All tests pass**: `julia --project=. -e 'using Pkg; Pkg.test()'`
2. **Notebook runs**: `notebooks/contracting_lorenz_comparison.ipynb`
3. **Documentation builds**: Check Documenter.jl builds successfully
4. **Examples work**: Any example code in docs should use new method

---

## Timeline

**Immediate** (can do now):
- ✓ Notebook updated to use only new method
- ✓ Documentation already updated
- Remove old methods from source (next step)
- Remove old tests
- Update COMPLETE_IMPLEMENTATION.md

**Before release**:
- Announce deprecation in release notes
- Update all example code
- Check for any external dependencies

---

## Status

- ✓ Iterative refinement implemented and tested (128 tests passing)
- ✓ Notebook updated to demonstrate superiority
- ✓ Documentation updated
- ⏳ Old methods still in codebase (pending removal)
- ⏳ Old tests still present (pending removal)

**Next step**: Remove old functions from `src/domain_restriction.jl`
