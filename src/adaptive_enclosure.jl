"""
Adaptive enclosure methods for computing tighter rigorous bounds on function images.

When a map has significant nonlinearity or expansion, evaluating F(box) directly
can give very loose bounds. Adaptive subdivision refines the enclosure by:
1. Subdividing the input box
2. Computing enclosures of sub-boxes
3. Taking the hull of sub-images

This trades computation time for tighter bounds.
"""

using IntervalArithmetic, IntervalBoxes

export adaptive_enclosure, adaptive_enclosure_auto
export AdaptiveEnclosureMethod, FixedDepth, ErrorBased, ExpansionBased

"""
    AdaptiveEnclosureMethod

Abstract type for adaptive enclosure strategies.
"""
abstract type AdaptiveEnclosureMethod end

"""
    FixedDepth(depth::Int)

Subdivide to a fixed depth regardless of enclosure quality.

Simple and predictable, good for benchmarking.
"""
struct FixedDepth <: AdaptiveEnclosureMethod
    depth::Int
end

"""
    ErrorBased(tol::Real, maxdepth::Int)

Subdivide until enclosure diameter is below tolerance or max depth reached.

Adapts to local nonlinearity: refines more where needed.
"""
struct ErrorBased <: AdaptiveEnclosureMethod
    tol::Real
    maxdepth::Int
end

"""
    ExpansionBased(max_expansion::Real, maxdepth::Int)

Subdivide when image expansion is too large.

Measures ratio: diam(F(box)) / diam(box), refines when > max_expansion.
Good for detecting and handling expanding/chaotic regions.
"""
struct ExpansionBased <: AdaptiveEnclosureMethod
    max_expansion::Real
    maxdepth::Int
end

"""
    adaptive_enclosure(F, box::IntervalBox{N}, method::AdaptiveEnclosureMethod;
                      base_enclosure=first_order_enclosure,
                      return_boxes=false) where {N}

Compute an adaptive rigorous enclosure of F(box).

# Arguments
- `F`: The function to enclose
- `box`: Input box
- `method`: Adaptive strategy (FixedDepth, ErrorBased, or ExpansionBased)
- `base_enclosure`: Base enclosure method (default: first_order_enclosure)
- `return_boxes`: If true, return (enclosure, sub_boxes, sub_images)

# Returns
- If `return_boxes=false`: IntervalBox enclosing F(box)
- If `return_boxes=true`: (enclosure, sub_boxes, sub_images)

# Example
```julia
F(x, y) = (4*x*(1-x), 2*y*(1-y))  # Expanding map

# Fixed depth subdivision
enc1 = adaptive_enclosure(F, box, FixedDepth(3))

# Error-based adaptive
enc2 = adaptive_enclosure(F, box, ErrorBased(1e-3, 10))

# Expansion-based (good for chaotic maps)
enc3 = adaptive_enclosure(F, box, ExpansionBased(2.0, 10))
```
"""
function adaptive_enclosure(F, box::IntervalBox{N}, method::FixedDepth;
                           base_enclosure=first_order_enclosure,
                           return_boxes=false) where {N}

    sub_boxes, sub_images = _subdivide_fixed_depth(F, box, method.depth, base_enclosure)

    # Compute hull of all sub-images
    enclosure = hull(sub_images...)

    if return_boxes
        return enclosure, sub_boxes, sub_images
    else
        return enclosure
    end
end

function adaptive_enclosure(F, box::IntervalBox{N}, method::ErrorBased;
                           base_enclosure=first_order_enclosure,
                           return_boxes=false) where {N}

    sub_boxes, sub_images = _subdivide_error_based(F, box, method.tol, method.maxdepth,
                                                   base_enclosure)

    enclosure = hull(sub_images...)

    if return_boxes
        return enclosure, sub_boxes, sub_images
    else
        return enclosure
    end
end

function adaptive_enclosure(F, box::IntervalBox{N}, method::ExpansionBased;
                           base_enclosure=first_order_enclosure,
                           return_boxes=false) where {N}

    sub_boxes, sub_images = _subdivide_expansion_based(F, box, method.max_expansion,
                                                       method.maxdepth, base_enclosure)

    enclosure = hull(sub_images...)

    if return_boxes
        return enclosure, sub_boxes, sub_images
    else
        return enclosure
    end
end

"""
    adaptive_enclosure_auto(F, box::IntervalBox{N};
                           base_enclosure=first_order_enclosure,
                           target_improvement=2.0,
                           maxdepth=10) where {N}

Automatically choose subdivision strategy based on map properties.

Compares zeroth-order vs first-order enclosures to detect nonlinearity,
then applies appropriate adaptive method.

# Arguments
- `F`: Function to enclose
- `box`: Input box
- `base_enclosure`: Base method (default: first_order)
- `target_improvement`: Desired tightness ratio (default: 2.0)
- `maxdepth`: Maximum subdivision depth

# Returns
Adaptive enclosure of F(box)

# Example
```julia
# Automatically adapts to map characteristics
enc = adaptive_enclosure_auto(F, box, target_improvement=3.0)
```
"""
function adaptive_enclosure_auto(F, box::IntervalBox{N};
                                base_enclosure=first_order_enclosure,
                                target_improvement=2.0,
                                maxdepth=10) where {N}

    # Compute both enclosures
    enc_zeroth = zeroth_order_enclosure(F, box)
    enc_first = base_enclosure(F, box)

    # Convert to boxes for measurement
    if enc_zeroth isa Tuple
        enc_zeroth_box = IntervalBox(enc_zeroth...)
        enc_first_box = IntervalBox(enc_first...)
    else
        enc_zeroth_box = enc_zeroth
        enc_first_box = enc_first
    end

    # Measure improvement from first-order
    improvement = box_measure(enc_zeroth_box) / box_measure(enc_first_box)

    # Measure expansion
    expansion = box_measure(enc_first_box) / box_measure(box)

    # Decision logic
    if improvement < 1.2
        # First-order doesn't help much → use expansion-based
        # (map is likely linear or first-order is poor)
        method = ExpansionBased(max(expansion, 2.0), maxdepth)
    elseif expansion > 3.0
        # Strong expansion → use expansion-based
        method = ExpansionBased(2.5, maxdepth)
    else
        # Moderate case → use error-based
        # Target: achieve target_improvement over current best
        current_width = box_width(enc_first_box)
        tol = current_width / target_improvement
        method = ErrorBased(tol, maxdepth)
    end

    return adaptive_enclosure(F, box, method; base_enclosure=base_enclosure)
end

# ============================================================================
# Internal subdivision routines
# ============================================================================

function _subdivide_fixed_depth(F, box::IntervalBox{N}, depth::Int,
                               base_enclosure) where {N}
    if depth == 0
        image = _compute_image_as_box(F, box, base_enclosure)
        return [box], [image]
    end

    # Subdivide
    sub_boxes = N == 2 ? quadrisect(box) : collect(bisect_box(box))

    # Recurse
    all_boxes = IntervalBox{N}[]
    all_images = IntervalBox{N}[]

    for sub_box in sub_boxes
        boxes, images = _subdivide_fixed_depth(F, sub_box, depth - 1, base_enclosure)
        append!(all_boxes, boxes)
        append!(all_images, images)
    end

    return all_boxes, all_images
end

function _subdivide_error_based(F, box::IntervalBox{N}, tol::Real, maxdepth::Int,
                               base_enclosure, depth::Int=0) where {N}

    # Compute image
    image = _compute_image_as_box(F, box, base_enclosure)

    # Check stopping criteria
    if depth >= maxdepth || box_width(image) <= tol
        return [box], [image]
    end

    # Subdivide and recurse
    sub_boxes = N == 2 ? quadrisect(box) : collect(bisect_box(box))

    all_boxes = IntervalBox{N}[]
    all_images = IntervalBox{N}[]

    for sub_box in sub_boxes
        boxes, images = _subdivide_error_based(F, sub_box, tol, maxdepth,
                                               base_enclosure, depth + 1)
        append!(all_boxes, boxes)
        append!(all_images, images)
    end

    return all_boxes, all_images
end

function _subdivide_expansion_based(F, box::IntervalBox{N}, max_expansion::Real,
                                   maxdepth::Int, base_enclosure,
                                   depth::Int=0) where {N}

    # Compute image
    image = _compute_image_as_box(F, box, base_enclosure)

    # Measure expansion
    expansion_ratio = box_measure(image) / box_measure(box)

    # Check stopping criteria
    if depth >= maxdepth || expansion_ratio <= max_expansion
        return [box], [image]
    end

    # Subdivide and recurse
    sub_boxes = N == 2 ? quadrisect(box) : collect(bisect_box(box))

    all_boxes = IntervalBox{N}[]
    all_images = IntervalBox{N}[]

    for sub_box in sub_boxes
        boxes, images = _subdivide_expansion_based(F, sub_box, max_expansion,
                                                   maxdepth, base_enclosure,
                                                   depth + 1)
        append!(all_boxes, boxes)
        append!(all_images, images)
    end

    return all_boxes, all_images
end

function _compute_image_as_box(F, box::IntervalBox{N}, base_enclosure) where {N}
    image = base_enclosure(F, box)

    if image isa Tuple
        return IntervalBox(image...)
    elseif image isa IntervalBox
        return image
    elseif image isa Interval
        error("Scalar output but N=$N dimensional box")
    else
        error("Unexpected image type: $(typeof(image))")
    end
end

"""
    compare_enclosure_methods(F, box::IntervalBox{N};
                              methods=[FixedDepth(0), FixedDepth(1), FixedDepth(2),
                                      ErrorBased(1e-2, 5), ExpansionBased(2.0, 5)],
                              base_enclosure=first_order_enclosure) where {N}

Compare multiple enclosure methods on the same input.

Returns a NamedTuple with statistics for each method.

# Example
```julia
results = compare_enclosure_methods(F, box)
for (method, stats) in zip(methods, results)
    println("\$method: volume = \$(stats.volume), time = \$(stats.time)")
end
```
"""
function compare_enclosure_methods(F, box::IntervalBox{N};
                                  methods=[FixedDepth(0), FixedDepth(1), FixedDepth(2),
                                          ErrorBased(1e-2, 5), ExpansionBased(2.0, 5)],
                                  base_enclosure=first_order_enclosure) where {N}

    results = []

    for method in methods
        # Time the computation
        t_start = time()
        enclosure, boxes, images = adaptive_enclosure(F, box, method;
                                                      base_enclosure=base_enclosure,
                                                      return_boxes=true)
        t_elapsed = time() - t_start

        stats = (
            method = method,
            enclosure = enclosure,
            volume = box_measure(enclosure),
            width = box_width(enclosure),
            n_boxes = length(boxes),
            time = t_elapsed
        )

        push!(results, stats)
    end

    return results
end
