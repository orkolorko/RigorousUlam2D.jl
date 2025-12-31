"""
Domain restriction utilities for efficient Ulam discretization.

When a map contracts the domain (e.g., maps [0,1]² strictly inside itself),
computing the transfer operator on the entire domain is wasteful. This module
provides tools to:
1. Compute rigorous enclosures of the forward image
2. Enlarge by noise support
3. Create restricted partitions of the effective domain
"""

using IntervalArithmetic, IntervalBoxes

export compute_forward_image_enclosure, compute_effective_domain
export restrict_partition_to_domain, create_restricted_partition

"""
    compute_forward_image_enclosure(F, domain::IntervalBox{N};
                                    enclosure_method=first_order_enclosure,
                                    subdivision_depth=0) where {N}

Compute a rigorous enclosure of the forward image F(domain).

For better accuracy, the domain can be subdivided before computing the image.

# Arguments
- `F`: The dynamical system
- `domain`: The domain box
- `enclosure_method`: Method for computing image (default: first_order_enclosure)
- `subdivision_depth`: Number of times to subdivide domain (0 = no subdivision)

# Returns
An IntervalBox enclosing F(domain)

# Example
```julia
F(x, y) = (0.5*x + 0.25, 0.5*y + 0.25)  # Contracting map
domain = IntervalBox(interval(0.0, 1.0), interval(0.0, 1.0))
image = compute_forward_image_enclosure(F, domain, subdivision_depth=2)
# image ≈ [0.25, 0.75] × [0.25, 0.75], much smaller than domain
```
"""
function compute_forward_image_enclosure(F, domain::IntervalBox{N};
                                        enclosure_method=first_order_enclosure,
                                        subdivision_depth=0) where {N}

    if subdivision_depth == 0
        # No subdivision - compute image directly
        image = enclosure_method(F, domain)

        # Convert to IntervalBox if needed
        if image isa Tuple
            return IntervalBox(image...)
        elseif image isa IntervalBox
            return image
        elseif image isa Interval
            error("Map returns scalar but domain is $N-dimensional")
        else
            error("Unexpected image type: $(typeof(image))")
        end
    else
        # Subdivide domain and compute union of images
        if N == 2
            boxes = quadrisect(domain)
        else
            left, right = bisect_box(domain)
            boxes = [left, right]
        end

        # Recursively compute images of sub-boxes
        sub_images = [compute_forward_image_enclosure(F, box;
                                                      enclosure_method=enclosure_method,
                                                      subdivision_depth=subdivision_depth-1)
                     for box in boxes]

        # Compute hull (smallest box containing all images)
        return hull(sub_images...)
    end
end

"""
    hull(boxes::IntervalBox{N}...) where {N}

Compute the smallest IntervalBox containing all input boxes.
"""
function hull(boxes::IntervalBox{N}...) where {N}
    if length(boxes) == 0
        error("Cannot compute hull of empty collection")
    end

    # Compute hull interval-by-interval
    hull_intervals = [hull([box[i] for box in boxes]...) for i in 1:N]
    return IntervalBox(hull_intervals...)
end

"""
    hull(intervals::Interval...)

Compute the smallest interval containing all input intervals.
"""
function hull(intervals::Interval...)
    if length(intervals) == 0
        error("Cannot compute hull of empty collection")
    end

    min_inf = minimum(inf(I) for I in intervals)
    max_sup = maximum(sup(I) for I in intervals)
    return interval(min_inf, max_sup)
end

"""
    compute_effective_domain(F, initial_domain::IntervalBox{N},
                            noise::Union{IntervalBox{N}, Nothing}=nothing;
                            enclosure_method=first_order_enclosure,
                            subdivision_depth=2,
                            safety_margin=1.1) where {N}

Compute the effective domain for Ulam discretization.

The effective domain is:
1. Forward image: F(initial_domain)
2. Plus noise support (if noise provided)
3. Plus safety margin
4. Intersected with initial_domain (to stay in bounds)

# Arguments
- `F`: The dynamical system
- `initial_domain`: The initial domain box
- `noise`: Noise support box (optional)
- `enclosure_method`: Method for image computation
- `subdivision_depth`: Subdivision for tighter enclosure
- `safety_margin`: Multiplicative expansion factor (default: 1.1 = 10% larger)

# Returns
`(effective_domain, image_no_noise, image_with_noise)`

# Example
```julia
F(x, y) = (0.5*x + 0.25, 0.5*y + 0.25)
domain = IntervalBox(interval(0.0, 1.0), interval(0.0, 1.0))
noise = uniform_noise_box(0.05, 2)

eff_domain, img, img_noisy = compute_effective_domain(F, domain, noise)
# eff_domain is much smaller than domain, focused on attractor region
```
"""
function compute_effective_domain(F, initial_domain::IntervalBox{N},
                                 noise::Union{IntervalBox{N}, Nothing}=nothing;
                                 enclosure_method=first_order_enclosure,
                                 subdivision_depth=2,
                                 safety_margin=1.1) where {N}

    # Compute forward image
    image_no_noise = compute_forward_image_enclosure(F, initial_domain;
                                                     enclosure_method=enclosure_method,
                                                     subdivision_depth=subdivision_depth)

    # Add noise if present
    if noise !== nothing
        image_with_noise = minkowski_sum(image_no_noise, noise)
    else
        image_with_noise = image_no_noise
    end

    # Apply safety margin (expand each interval by margin factor)
    if safety_margin != 1.0
        expanded_intervals = [expand_interval(image_with_noise[i], safety_margin)
                             for i in 1:N]
        effective_domain = IntervalBox(expanded_intervals...)
    else
        effective_domain = image_with_noise
    end

    # Intersect with initial domain to stay in bounds
    effective_domain = intersect_boxes(effective_domain, initial_domain)

    return effective_domain, image_no_noise, image_with_noise
end

"""
    expand_interval(I::Interval, factor::Real)

Expand interval I around its midpoint by multiplicative factor.

# Example
```julia
I = interval(0.4, 0.6)  # width = 0.2, center = 0.5
expand_interval(I, 1.2)  # → [0.38, 0.62], width = 0.24
```
"""
function expand_interval(I::Interval, factor::Real)
    center = mid(I)
    half_width = diam(I) / 2
    new_half_width = half_width * factor
    return interval(center - new_half_width, center + new_half_width)
end

"""
    intersect_boxes(A::IntervalBox{N}, B::IntervalBox{N}) where {N}

Compute the intersection A ∩ B of two boxes.

Returns the largest box contained in both A and B, or an empty box if disjoint.
"""
function intersect_boxes(A::IntervalBox{N}, B::IntervalBox{N}) where {N}
    intervals = [intersect_interval(A[i], B[i]) for i in 1:N]
    return IntervalBox(intervals...)
end

"""
    create_restricted_partition(F, initial_domain::IntervalBox{N},
                               noise::Union{IntervalBox{N}, Nothing}=nothing;
                               partition_depth=4,
                               max_diameter=nothing,
                               enclosure_method=first_order_enclosure,
                               subdivision_depth=2,
                               safety_margin=1.1,
                               verbose=true) where {N}

Create an optimized partition focused on the effective domain.

This is the main convenience function for domain restriction.

# Arguments
- `F`: The dynamical system
- `initial_domain`: The initial domain box
- `noise`: Noise support (optional)
- `partition_depth`: Number of uniform subdivisions of effective domain (ignored if max_diameter is set)
- `max_diameter`: Maximum allowed partition diameter (stopping criterion from PDF error bounds)
- Other arguments: see `compute_effective_domain`

# Returns
`(partition, effective_domain, reduction_info)`

Where `reduction_info` is a NamedTuple with:
- `initial_volume`: Volume of initial domain
- `effective_volume`: Volume of effective domain
- `reduction_factor`: How much smaller (initial/effective)
- `n_boxes`: Number of boxes in partition
- `partition_diameter`: Maximum diameter in final partition

# Example
```julia
F(x, y) = (0.5*x + 0.25, 0.5*y + 0.25)
domain = IntervalBox(interval(0.0, 1.0), interval(0.0, 1.0))
noise = uniform_noise_box(0.01, 2)

# Fixed depth approach
partition, eff_dom, info = create_restricted_partition(F, domain, noise,
                                                       partition_depth=4)

# Diameter-based approach (adaptive to error tolerance from PDF)
partition, eff_dom, info = create_restricted_partition(F, domain, noise,
                                                       max_diameter=0.01)
println("Achieved partition diameter: ", info.partition_diameter)
```
"""
function create_restricted_partition(F, initial_domain::IntervalBox{N},
                                    noise::Union{IntervalBox{N}, Nothing}=nothing;
                                    partition_depth=4,
                                    max_diameter=nothing,
                                    enclosure_method=first_order_enclosure,
                                    subdivision_depth=2,
                                    safety_margin=1.1,
                                    verbose=true) where {N}

    # Compute effective domain
    effective_domain, img_clean, img_noisy = compute_effective_domain(
        F, initial_domain, noise;
        enclosure_method=enclosure_method,
        subdivision_depth=subdivision_depth,
        safety_margin=safety_margin
    )

    # Compute reduction statistics
    initial_volume = box_measure(initial_domain)
    effective_volume = box_measure(effective_domain)
    reduction_factor = initial_volume / effective_volume

    if verbose
        println("Domain Restriction Analysis:")
        println("  Initial domain:    $initial_domain")
        println("  Effective domain:  $effective_domain")
        println("  Initial volume:    $initial_volume")
        println("  Effective volume:  $effective_volume")
        println("  Reduction factor:  $(round(reduction_factor, digits=2))×")
        println("  Image (no noise):  $img_clean")
        if noise !== nothing
            println("  Image (with noise): $img_noisy")
            println("  Noise expansion:    $(round(box_measure(img_noisy)/box_measure(img_clean), digits=2))×")
        end
    end

    # Create partition of effective domain
    # Two modes: fixed depth or diameter-based (adaptive to error tolerance)
    boxes = [effective_domain]

    if max_diameter !== nothing
        # Diameter-based stopping criterion (from PDF error bounds)
        # Subdivide until all boxes have diameter ≤ max_diameter
        level = 0
        while true
            current_diameter = maximum(box_width(box) for box in boxes)

            if current_diameter <= max_diameter
                if verbose
                    println("  Reached target diameter after $level subdivisions")
                end
                break
            end

            # Subdivide all boxes
            new_boxes = eltype(boxes)[]
            for box in boxes
                if N == 2
                    append!(new_boxes, quadrisect(box))
                else
                    left, right = bisect_box(box)
                    push!(new_boxes, left, right)
                end
            end
            boxes = new_boxes
            level += 1

            # Safety check to prevent infinite loops
            if level > 20
                @warn "Reached maximum subdivision depth (20) before achieving target diameter"
                break
            end
        end
    else
        # Fixed depth subdivision (original behavior)
        for level in 1:partition_depth
            new_boxes = eltype(boxes)[]
            for box in boxes
                if N == 2
                    append!(new_boxes, quadrisect(box))
                else
                    left, right = bisect_box(box)
                    push!(new_boxes, left, right)
                end
            end
            boxes = new_boxes
        end
    end

    partition = UlamPartition(boxes, effective_domain)

    # Compute final partition diameter
    final_diameter = maximum(box_width(box) for box in boxes)

    if verbose
        println("  Partition boxes:   $(partition.n_boxes)")
        println("  Partition diameter: $(round(final_diameter, digits=6))")
        println("  Compared to full partition: $(round(reduction_factor^N, digits=2))× fewer boxes")
    end

    reduction_info = (
        initial_volume = initial_volume,
        effective_volume = effective_volume,
        reduction_factor = reduction_factor,
        n_boxes = partition.n_boxes,
        partition_diameter = final_diameter,
        image_clean = img_clean,
        image_noisy = img_noisy,
        effective_domain = effective_domain
    )

    return partition, effective_domain, reduction_info
end

"""
    restrict_partition_to_domain(boxes::Vector{IntervalBox{N}},
                                 domain::IntervalBox{N}) where {N}

Filter a partition to only include boxes that intersect the given domain.

Useful for restricting an existing partition to an effective domain.

# Returns
Vector of boxes that intersect domain
"""
function restrict_partition_to_domain(boxes::Vector{<:IntervalBox{N}},
                                     domain::IntervalBox{N}) where {N}
    return [box for box in boxes if !isdisjoint_box(box, domain)]
end
