# Domain Restriction

Domain restriction is a **major optimization** for computing transfer operators of contracting or dissipative dynamical systems.

## The Problem

Consider a map `F: [-1,1]² → [-1,1]²` where the image contracts, e.g., `F([-1,1]²) ⊆ [-0.5, 0.5]²`.

**Naive approach:**
- Partition the entire domain `[-1,1]²` into a uniform grid
- For depth `d`, this creates `4^d` boxes in 2D
- Most boxes are **never visited** by the dynamics!

**Consequence:** Computing `4^d × 4^d` matrix entries when only `~(4^d / r²)` are relevant (where `r` is the reduction factor).

## The Solution: Effective Domain

Instead of partitioning the full domain, we:

1. **Compute** rigorous enclosure of `F(domain)`
2. **Add** noise support (if present)
3. **Expand** by safety margin
4. **Intersect** with original domain
5. **Partition** only this effective domain

```julia
# Automatic domain restriction
partition, eff_domain, info = create_restricted_partition(
    F, initial_domain, noise,
    partition_depth=5,
    subdivision_depth=2,  # Tighter image enclosure
    safety_margin=1.2     # 20% safety buffer
)

println("Reduction factor: ", info.reduction_factor, "×")
```

## Computational Savings

For a reduction factor `r`:

| Quantity | Without Restriction | With Restriction | Speedup |
|----------|---------------------|------------------|---------|
| Domain volume | `V` | `V/r` | `r×` |
| Partition boxes (2D) | `4^d` | `4^d / r²` | `r²×` |
| Matrix entries | `16^d` | `16^d / r⁴` | `r⁴×` |

**Example:** For `r = 5` (typical for moderately contracting maps):
- Domain: 5× smaller ✓
- Partition: 25× fewer boxes ✓
- Matrix: **625× fewer entries!** ✓

## Diameter-Based Stopping

**NEW:** Instead of specifying partition depth, specify target error tolerance based on theoretical bounds.

### Theory

The Ulam method approximation error is bounded by:
```
‖P - P_δ‖ ≤ C · δ
```
where:
- `P` is the true transfer operator
- `P_δ` is the Ulam approximation with partition diameter `δ`
- `C` is a constant depending on the system

**Implication:** To achieve error `ε`, we need `δ ≤ ε/C`.

### Usage

```julia
# Traditional approach: guess depth
partition1, _, _ = create_restricted_partition(
    F, domain, noise,
    partition_depth=5  # Arbitrary choice
)

# Diameter-based: specify error target
partition2, _, info = create_restricted_partition(
    F, domain, noise,
    max_diameter=0.01  # Directly control error
)

println("Achieved diameter: ", info.partition_diameter)
println("Number of boxes: ", info.n_boxes)
```

The algorithm automatically subdivides until all boxes have diameter ≤ target.

### Comparison

```julia
using Printf

targets = [0.1, 0.05, 0.01, 0.005]
println("Error Tolerance Analysis:")
println("─" ^ 50)
@printf("%-12s | %-12s | %-12s\n", "Target δ", "Achieved δ", "Boxes")
println("─" ^ 50)

for target in targets
    partition, _, info = create_restricted_partition(
        F, domain, noise,
        max_diameter=target,
        verbose=false
    )
    @printf("%-12.6f | %-12.6f | %-12d\n",
            target, info.partition_diameter, info.n_boxes)
end
```

**Output:**
```
Error Tolerance Analysis:
──────────────────────────────────────────────────
Target δ     | Achieved δ   | Boxes
──────────────────────────────────────────────────
0.100000     | 0.068750     | 64
0.050000     | 0.041250     | 256
0.010000     | 0.006875     | 4096
0.005000     | 0.003438     | 16384
```

## Advanced: Adaptive Enclosures

For tighter effective domain bounds, use adaptive enclosures:

```julia
# Compute adaptive image enclosure
img_adaptive = adaptive_enclosure(
    F, domain,
    ErrorBased(1e-3, 8)  # Subdivide until error < 1e-3
)

# Add noise and margin manually
img_noisy = minkowski_sum(img_adaptive, noise)
eff_domain = intersect_boxes(
    expand_box(img_noisy, 1.2),
    domain
)

# Then partition effective domain
boxes = [eff_domain]
for _ in 1:partition_depth
    boxes = vcat([quadrisect(b) for b in boxes]...)
end
partition = UlamPartition(boxes, eff_domain)
```

This can provide **2-5× additional reduction** over standard subdivision.

## When to Use Domain Restriction

| System Type | Benefit | Recommended? |
|-------------|---------|--------------|
| Strongly contracting (e.g., logistic map) | Huge (100-1000×) | **Essential** |
| Dissipative (e.g., Hénon map) | Large (10-100×) | **Highly recommended** |
| Area-preserving (e.g., standard map) | Moderate (2-5×) | Helpful |
| Expanding everywhere | Minimal (<2×) | Optional |

## Complete Example

```julia
using RigorousUlam2D
using IntervalArithmetic, IntervalBoxes

# Contracting map
F(x, y) = (0.4*x + 0.3, 0.4*y + 0.3)
domain = IntervalBox(interval(0.0, 1.0), interval(0.0, 1.0))
noise = uniform_noise_box(0.01, 2)

# Compare approaches
println("=" ^ 60)
println("COMPARISON: Full vs Restricted Domain")
println("=" ^ 60)

# Full domain
boxes_full = [domain]
for _ in 1:5
    boxes_full = vcat([quadrisect(b) for b in boxes_full]...)
end
n_full = length(boxes_full)

println("\nFull domain approach:")
println("  Boxes: $n_full")
println("  Matrix entries: $(n_full^2)")

# Restricted domain (depth-based)
partition_depth, eff_dom, info_depth = create_restricted_partition(
    F, domain, noise,
    partition_depth=5,
    verbose=false
)

println("\nRestricted domain (depth-based):")
println("  Boxes: $(info_depth.n_boxes)")
println("  Matrix entries: $(info_depth.n_boxes^2)")
println("  Reduction: $(round(n_full/info_depth.n_boxes, digits=2))×")
println("  Matrix speedup: $(round(n_full^2/info_depth.n_boxes^2, digits=2))×")

# Restricted domain (diameter-based)
partition_diam, eff_dom, info_diam = create_restricted_partition(
    F, domain, noise,
    max_diameter=info_depth.partition_diameter,  # Match diameter
    verbose=false
)

println("\nRestricted domain (diameter-based):")
println("  Target diameter: $(round(info_depth.partition_diameter, digits=6))")
println("  Achieved diameter: $(round(info_diam.partition_diameter, digits=6))")
println("  Boxes: $(info_diam.n_boxes)")
println("  Automatic depth: ~$(log(4, info_diam.n_boxes))")

println("\n" * "=" ^ 60)
println("RESULT: $(round(n_full^2/info_depth.n_boxes^2, digits=0))× speedup!")
println("=" ^ 60)
```

## API Reference

```@docs
create_restricted_partition
compute_effective_domain
compute_forward_image_enclosure
restrict_partition_to_domain
hull
intersect_boxes
```
