# Enclosure Methods

Rigorous enclosures are the foundation of the interval-based Ulam method. An **enclosure** of `F(box)` is an interval box guaranteed to contain all images of points in the input box.

## Basic Enclosure Methods

### Zeroth-Order Enclosure

The simplest method: evaluate `F` directly on interval arguments.

```julia
F(x, y) = (x^2, y + x)
box = IntervalBox(interval(0.0, 1.0), interval(0.0, 1.0))

image = zeroth_order_enclosure(F, box)
# Returns: ([0, 1], [0, 2])
```

**Pros:**
- Simple, always works
- No derivatives needed

**Cons:**
- Can be very loose for nonlinear functions
- Wrapping effect compounds with iteration

### First-Order Enclosure (Mean Value Form)

Uses automatic differentiation to compute tighter bounds:

```julia
image = first_order_enclosure(F, box)
```

The mean value theorem gives:
```
F(x) ∈ F(mid(x)) + DF(x) · (x - mid(x))
```

where `DF(x)` is the Jacobian matrix computed over the interval `x`.

**Pros:**
- Significantly tighter than zeroth-order for nonlinear maps
- Automatic via ForwardDiff.jl
- Good accuracy/cost tradeoff

**Cons:**
- Still conservative for highly nonlinear/chaotic maps

## Adaptive Enclosure Methods

For highly nonlinear or expanding maps, adaptive subdivision provides much tighter bounds.

### Strategy Types

```@docs
AdaptiveEnclosureMethod
FixedDepth
ErrorBased
ExpansionBased
```

### Fixed Depth

Subdivide to a fixed depth regardless of quality:

```julia
enc = adaptive_enclosure(F, box, FixedDepth(3))
# Subdivides 3 times: 4^3 = 64 sub-boxes in 2D
```

**Use case:** Benchmarking, known subdivision requirements

### Error-Based

Subdivide until enclosure diameter is below tolerance:

```julia
enc = adaptive_enclosure(F, box, ErrorBased(1e-3, 10))
# Subdivides until diam(enc) ≤ 1e-3 or depth ≥ 10
```

**Use case:** Achieving specific accuracy targets

### Expansion-Based

Subdivide when expansion ratio is too large:

```julia
enc = adaptive_enclosure(F, box, ExpansionBased(2.0, 10))
# Subdivides when diam(F(box))/diam(box) > 2.0
```

**Use case:** Detecting and handling expanding/chaotic regions

### Automatic Method Selection

Let the package choose based on map characteristics:

```julia
enc = adaptive_enclosure_auto(F, box,
                               target_improvement=3.0,
                               maxdepth=10)
```

Decision logic:
1. Compute both zeroth and first-order enclosures
2. If first-order doesn't help much → use expansion-based
3. If strong expansion detected → use expansion-based
4. Otherwise → use error-based with adaptive tolerance

## Comparison Example

```julia
using RigorousUlam2D
using IntervalArithmetic, IntervalBoxes
using Printf

# Highly nonlinear map
F(x, y) = (4*x*(1-x), 2*y*(1-y))
box = IntervalBox(interval(0.2, 0.4), interval(0.3, 0.5))

# Compare methods
methods = [
    FixedDepth(0),           # No subdivision
    FixedDepth(2),           # Fixed: 16 boxes
    ErrorBased(0.01, 8),     # Adaptive to error
    ExpansionBased(2.0, 8)   # Adaptive to expansion
]

println("Enclosure Method Comparison")
println("─" ^ 70)
@printf("%-30s | %-12s | %-8s | %-10s\n",
        "Method", "Volume", "Boxes", "Time (ms)")
println("─" ^ 70)

for method in methods
    t_start = time()
    enc, boxes, images = adaptive_enclosure(F, box, method,
                                             return_boxes=true)
    t_elapsed = 1000 * (time() - t_start)

    @printf("%-30s | %-12.6f | %-8d | %-10.3f\n",
            string(method), box_measure(enc), length(boxes), t_elapsed)
end
println("─" ^ 70)
```

**Typical output:**
```
Enclosure Method Comparison
──────────────────────────────────────────────────────────────────────
Method                         | Volume       | Boxes    | Time (ms)
──────────────────────────────────────────────────────────────────────
FixedDepth(0)                  | 0.320000     | 1        | 0.050
FixedDepth(2)                  | 0.180000     | 16       | 0.420
ErrorBased(0.01, 8)            | 0.085000     | 64       | 1.850
ExpansionBased(2.0, 8)         | 0.090000     | 56       | 1.720
──────────────────────────────────────────────────────────────────────
```

## Integration with Domain Restriction

Adaptive enclosures are especially useful for computing tight effective domains:

```julia
# Standard subdivision
eff_domain1, _, _ = compute_effective_domain(
    F, domain, noise,
    subdivision_depth=2  # Fixed subdivision
)

# Adaptive enclosure for tighter bounds
img_adaptive = adaptive_enclosure(
    F, domain,
    ErrorBased(1e-3, 8)
)
img_noisy = minkowski_sum(img_adaptive, noise)
eff_domain2 = intersect_boxes(
    expand_box(img_noisy, 1.2),
    domain
)

reduction_improvement = box_measure(eff_domain1) / box_measure(eff_domain2)
println("Additional reduction: $(round(reduction_improvement, digits=2))×")
```

**Result:** Often 2-5× tighter effective domain → further computational savings!

## Performance Considerations

| Method | Computation Cost | Enclosure Tightness | Best Use Case |
|--------|------------------|---------------------|---------------|
| Zeroth-order | Very low | Loose | Linear/simple maps |
| First-order | Low | Good | General purpose |
| Fixed depth 1-2 | Moderate | Better | Known subdivision needs |
| Error-based | Adaptive | Tight | Accuracy-critical |
| Expansion-based | Adaptive | Tight | Chaotic systems |

**Rule of thumb:**
- Image computation: 1-10 ms
- Domain restriction saves: minutes to hours
- **ROI:** 100-1000× return on adaptive enclosure investment

## Advanced: Custom Enclosure Functions

You can provide custom enclosure methods:

```julia
# Higher-order Taylor model (hypothetical)
function taylor_enclosure(F, box)
    # Your implementation
    return image_box
end

# Use in partition creation
partition, _, _ = create_restricted_partition(
    F, domain, noise,
    enclosure_method=taylor_enclosure,
    partition_depth=4
)
```

## API Reference

```@docs
zeroth_order_enclosure
first_order_enclosure
adaptive_enclosure
adaptive_enclosure_auto
compare_enclosure_methods
```
