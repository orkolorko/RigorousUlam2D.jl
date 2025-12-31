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

    if length(boxes) == 1
        return boxes[1]
    end

    # Compute hull interval-by-interval
    # Extract the i-th interval from each box and compute their hull
    hull_intervals = Interval[]
    for i in 1:N
        intervals_i = [box[i] for box in boxes]
        push!(hull_intervals, hull(intervals_i...))
    end
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
    compute_forward_image_boxes(F, domain::IntervalBox{N};
                               enclosure_method=first_order_enclosure,
                               subdivision_depth=0) where {N}

Compute the forward image as a **list of boxes** (not their hull).

This is more precise than `compute_forward_image_enclosure` because it preserves
the structure of the image rather than enclosing it in a single box.

# Arguments
- `F`: The dynamical system
- `domain`: Domain box to compute image of
- `enclosure_method`: Enclosure method (default: first_order_enclosure)
- `subdivision_depth`: Number of subdivisions (0 = no subdivision)

# Returns
Vector of IntervalBoxes representing F(domain)

# Example
```julia
F(x, y) = (0.5*x + 0.25, 0.5*y + 0.25)
domain = IntervalBox(interval(0.0, 1.0), interval(0.0, 1.0))
boxes = compute_forward_image_boxes(F, domain, subdivision_depth=2)
# Returns 4^2 = 16 boxes (tighter than single hull)
```
"""
function compute_forward_image_boxes(F, domain::IntervalBox{N};
                                    enclosure_method=first_order_enclosure,
                                    subdivision_depth=0) where {N}

    if subdivision_depth == 0
        # No subdivision - return single box
        image = enclosure_method(F, domain)

        # Convert to IntervalBox if needed
        if image isa Tuple
            return [IntervalBox(image...)]
        elseif image isa IntervalBox
            return [image]
        elseif image isa Interval
            error("Map returns scalar but domain is $N-dimensional")
        else
            error("Unexpected image type: $(typeof(image))")
        end
    else
        # Subdivide domain
        if N == 2
            boxes = quadrisect(domain)
        else
            left, right = bisect_box(domain)
            boxes = [left, right]
        end

        # Recursively compute images and concatenate lists
        all_image_boxes = IntervalBox{N, Float64, Interval{Float64}}[]
        for box in boxes
            sub_images = compute_forward_image_boxes(F, box;
                                                     enclosure_method=enclosure_method,
                                                     subdivision_depth=subdivision_depth-1)
            append!(all_image_boxes, sub_images)
        end

        return all_image_boxes
    end
end

"""
    compute_effective_domain(F, initial_domain::IntervalBox{N},
                            noise::Union{IntervalBox{N}, Nothing}=nothing;
                            enclosure_method=first_order_enclosure,
                            subdivision_depth=2,
                            safety_margin=1.1,
                            verbose=false) where {N}

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
- `verbose`: Print progress information (default: false)

# Returns
`(effective_domain, image_no_noise, image_with_noise)`

# Example
```julia
F(x, y) = (0.5*x + 0.25, 0.5*y + 0.25)
domain = IntervalBox(interval(0.0, 1.0), interval(0.0, 1.0))
noise = uniform_noise_box(0.05, 2)

eff_domain, img, img_noisy = compute_effective_domain(F, domain, noise, verbose=true)
# eff_domain is much smaller than domain, focused on attractor region
```
"""
function compute_effective_domain(F, initial_domain::IntervalBox{N},
                                 noise::Union{IntervalBox{N}, Nothing}=nothing;
                                 enclosure_method=first_order_enclosure,
                                 subdivision_depth=2,
                                 safety_margin=1.1,
                                 verbose=false) where {N}

    if verbose
        println("Computing effective domain...")
        println("  Initial domain: $initial_domain")
    end

    # Compute forward image
    image_no_noise = compute_forward_image_enclosure(F, initial_domain;
                                                     enclosure_method=enclosure_method,
                                                     subdivision_depth=subdivision_depth)

    if verbose
        println("  Image (no noise): $image_no_noise")
    end

    # Add noise if present
    if noise !== nothing
        image_with_noise = minkowski_sum(image_no_noise, noise)
        if verbose
            println("  Noise: $noise")
            println("  Image (with noise): $image_with_noise")
        end
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

    if verbose
        println("  Effective domain: $effective_domain")
        println("  Reduction factor: $(round(box_measure(initial_domain)/box_measure(effective_domain), digits=2))×")
    end

    return effective_domain, image_no_noise, image_with_noise
end

"""
    compute_effective_domain_boxes(F, initial_domain::IntervalBox{N},
                                   noise::Union{IntervalBox{N}, Nothing}=nothing;
                                   enclosure_method=first_order_enclosure,
                                   subdivision_depth=2,
                                   safety_margin=1.1,
                                   verbose=false) where {N}

Compute the effective domain as a **list of boxes** (more precise than single hull).

This is superior to `compute_effective_domain` when noise is present because:
1. Noise is added to each image box individually (not to the hull)
2. Preserves the structure of the image
3. Avoids enclosing "empty space" in the hull

# Arguments
Same as `compute_effective_domain`

# Returns
`(effective_boxes::Vector{IntervalBox}, total_measure::Float64)`
- `effective_boxes`: List of boxes covering the effective domain
- `total_measure`: Sum of measures of all boxes

# Example
```julia
F(x, y) = (0.5*x + 0.25, 0.5*y + 0.25)
domain = IntervalBox(interval(0.0, 1.0), interval(0.0, 1.0))
noise = uniform_noise_box(0.05, 2)

boxes, measure = compute_effective_domain_boxes(F, domain, noise,
                                                subdivision_depth=3, verbose=true)
# Returns list of boxes, much tighter than single hull!
```
"""
function compute_effective_domain_boxes(F, initial_domain::IntervalBox{N},
                                       noise::Union{IntervalBox{N}, Nothing}=nothing;
                                       enclosure_method=first_order_enclosure,
                                       subdivision_depth=2,
                                       safety_margin=1.1,
                                       verbose=false) where {N}

    if verbose
        println("Computing effective domain (as list of boxes)...")
        println("  Initial domain: $initial_domain")
    end

    # Compute forward image as list of boxes
    image_boxes = compute_forward_image_boxes(F, initial_domain;
                                              enclosure_method=enclosure_method,
                                              subdivision_depth=subdivision_depth)

    if verbose
        println("  Image boxes: $(length(image_boxes)) boxes")
        image_measure = sum(box_measure(box) for box in image_boxes)
        println("  Image measure (no noise): $(round(image_measure, digits=6))")
    end

    # Add noise to each box individually (Minkowski sum)
    if noise !== nothing
        noisy_boxes = [minkowski_sum(box, noise) for box in image_boxes]
        if verbose
            println("  Noise: $noise")
            noisy_measure = sum(box_measure(box) for box in noisy_boxes)
            println("  Image measure (with noise): $(round(noisy_measure, digits=6))")
        end
    else
        noisy_boxes = image_boxes
    end

    # Apply safety margin to each box
    if safety_margin != 1.0
        expanded_boxes = IntervalBox{N, Float64, Interval{Float64}}[]
        for box in noisy_boxes
            expanded_intervals = [expand_interval(box[i], safety_margin) for i in 1:N]
            push!(expanded_boxes, IntervalBox(expanded_intervals...))
        end
    else
        expanded_boxes = noisy_boxes
    end

    # Intersect each box with initial domain to stay in bounds
    effective_boxes = IntervalBox{N, Float64, Interval{Float64}}[]
    for box in expanded_boxes
        intersected = intersect_boxes(box, initial_domain)
        # Only add non-empty boxes
        if !is_empty_box(intersected)
            push!(effective_boxes, intersected)
        end
    end

    total_measure = sum(box_measure(box) for box in effective_boxes)

    if verbose
        println("  Effective domain: $(length(effective_boxes)) boxes")
        println("  Total measure: $(round(total_measure, digits=6))")
        println("  Reduction factor: $(round(box_measure(initial_domain)/total_measure, digits=2))×")
    end

    # Note: total_measure may overcount due to box overlaps
    # The boxes represent a covering of the effective domain, not a partition
    return effective_boxes, total_measure
end

"""
    create_partition_from_boxes(boxes::Vector{IntervalBox{N}},
                               partition_depth::Int=0) where {N}

Create a partition by subdividing a list of boxes.

This is more precise than first taking the hull and then partitioning,
because it preserves the structure of the domain.

# Arguments
- `boxes`: List of boxes to partition
- `partition_depth`: Depth of quadrisection for each box

# Returns
`UlamPartition` covering the union of input boxes

# Example
```julia
# boxes is a list of effective domain boxes
partition = create_partition_from_boxes(boxes, partition_depth=3)
# Each box is subdivided to depth 3, total = length(boxes) * 4^3
```
"""
function create_partition_from_boxes(boxes::Vector{<:IntervalBox},
                                    partition_depth::Int=0)

    if isempty(boxes)
        error("Cannot create partition from empty box list")
    end

    # Get dimension from first box
    first_box = boxes[1]
    N = length(first_box)

    all_partition_boxes = typeof(first_box)[]

    for box in boxes
        if partition_depth == 0
            push!(all_partition_boxes, box)
        else
            # Recursively subdivide this box
            current_boxes = [box]
            for _ in 1:partition_depth
                new_boxes = typeof(first_box)[]
                for b in current_boxes
                    if N == 2
                        append!(new_boxes, quadrisect(b))
                    else
                        left, right = bisect_box(b)
                        push!(new_boxes, left, right)
                    end
                end
                current_boxes = new_boxes
            end
            append!(all_partition_boxes, current_boxes)
        end
    end

    # Compute the hull of all boxes for the partition domain
    partition_domain = hull(all_partition_boxes...)

    return UlamPartition(all_partition_boxes, partition_domain)
end

"""
    compute_effective_partition(F, initial_domain::IntervalBox{N},
                                noise::Union{IntervalBox{N}, Nothing}=nothing;
                                partition_depth::Int=4,
                                enclosure_method=first_order_enclosure,
                                safety_margin=1.1,
                                verbose=false) where {N}

Compute effective domain partition using the **forward flagging algorithm**.

Algorithm:
1. Create initial partition of full domain
2. For each box B in partition:
   - Compute F(B) + noise + safety margin
   - Flag all partition boxes touched by this image
3. Return partition of flagged boxes only

This is more efficient than computing image → partitioning because:
- Works directly with target partition resolution
- No hull computation needed
- Exact boxes for Ulam discretization

# Arguments
- `F`: The dynamical system
- `initial_domain`: The full domain box
- `noise`: Noise support box (optional)
- `partition_depth`: Depth of initial partition (2D: gives 4^depth boxes)
- `enclosure_method`: Method for computing F(box)
- `safety_margin`: Expansion factor for images
- `verbose`: Print progress

# Returns
`(partition::UlamPartition, info::NamedTuple)`

# Example
```julia
F(x, y) = (0.5*x + 0.25, 0.5*y + 0.25)
domain = IntervalBox(interval(0.0, 1.0), interval(0.0, 1.0))
noise = uniform_noise_box(0.01, 2)

partition, info = compute_effective_partition(F, domain, noise,
                                              partition_depth=5,
                                              verbose=true)
# partition contains only boxes touched by forward images
# Dramatic reduction for contracting maps!
```
"""
function compute_effective_partition(F, initial_domain::IntervalBox{N},
                                    noise::Union{IntervalBox{N}, Nothing}=nothing;
                                    partition_depth::Int=4,
                                    enclosure_method=first_order_enclosure,
                                    safety_margin=1.1,
                                    verbose=false) where {N}

    if verbose
        println("Computing effective partition (forward flagging)...")
        println("  Initial domain: $initial_domain")
        println("  Partition depth: $partition_depth")
    end

    # Step 1: Create initial partition of full domain
    boxes = [initial_domain]
    for _ in 1:partition_depth
        new_boxes = typeof(initial_domain)[]
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

    n_boxes = length(boxes)
    if verbose
        println("  Initial partition: $n_boxes boxes")
    end

    # Step 2: Flag boxes touched by forward images
    flagged = falses(n_boxes)  # Initially all false

    for (i, box) in enumerate(boxes)
        # Compute forward image of this box
        image = enclosure_method(F, box)

        # Convert to IntervalBox
        if image isa Tuple
            image_box = IntervalBox(image...)
        elseif image isa IntervalBox
            image_box = image
        else
            error("Unexpected image type: $(typeof(image))")
        end

        # Add noise
        if noise !== nothing
            image_box = minkowski_sum(image_box, noise)
        end

        # Apply safety margin
        if safety_margin != 1.0
            expanded_intervals = [expand_interval(image_box[i], safety_margin) for i in 1:N]
            image_box = IntervalBox(expanded_intervals...)
        end

        # Intersect with domain (stay in bounds)
        image_box = intersect_boxes(image_box, initial_domain)

        # Flag all boxes that intersect with this image
        for (j, target_box) in enumerate(boxes)
            if !flagged[j] && intersects(image_box, target_box)
                flagged[j] = true
            end
        end
    end

    # Step 3: Extract flagged boxes
    effective_boxes = [boxes[i] for i in 1:n_boxes if flagged[i]]
    n_effective = length(effective_boxes)

    if verbose
        println("  Effective boxes: $n_effective")
        println("  Reduction: $(round(n_boxes/n_effective, digits=2))×")
        total_measure = sum(box_measure(box) for box in effective_boxes)
        println("  Total measure: $(round(total_measure, digits=6))")
        println("  Domain reduction: $(round(box_measure(initial_domain)/total_measure, digits=2))×")
    end

    # Create partition
    partition = UlamPartition(effective_boxes, initial_domain)

    info = (
        n_boxes_initial = n_boxes,
        n_boxes_effective = n_effective,
        reduction_factor = n_boxes / n_effective,
        partition_depth = partition_depth
    )

    return partition, info
end

"""
    compute_effective_partition_adaptive(F, initial_domain::IntervalBox{N},
                                         noise::Union{IntervalBox{N}, Nothing}=nothing;
                                         initial_depth::Int=3,
                                         refinement_levels::Union{Int,Nothing}=nothing,
                                         max_diameter::Union{Float64,Nothing}=nothing,
                                         enclosure_method=first_order_enclosure,
                                         safety_margin=1.1,
                                         verbose=false) where {N}

Compute effective partition using **iterative refinement** of the forward flagging algorithm.

Algorithm:
1. Start with coarse partition (depth = initial_depth)
2. Run forward flagging → get effective boxes
3. Subdivide effective boxes
4. Repeat flagging on refined partition
5. Continue until stopping criterion is met

Stopping criteria (use one):
- `refinement_levels`: Fixed number of refinement iterations (original behavior)
- `max_diameter`: Subdivide until partition diameter ≤ max_diameter (error-based, recommended)

This gives better resolution in the effective domain while keeping full domain coarse.

# Arguments
- `F`: The dynamical system
- `initial_domain`: The full domain box
- `noise`: Noise support box (optional)
- `initial_depth`: Initial partition depth
- `refinement_levels`: Number of refinement iterations (ignored if max_diameter is set)
- `max_diameter`: Maximum allowed partition diameter (from PDF error bounds)
- `enclosure_method`: Method for computing F(box)
- `safety_margin`: Expansion factor
- `verbose`: Print progress

# Returns
`(partition::UlamPartition, info::NamedTuple)`

# Example
```julia
F(x, y) = (0.5*x + 0.25, 0.5*y + 0.25)
domain = IntervalBox(interval(0.0, 1.0), interval(0.0, 1.0))
noise = uniform_noise_box(0.01, 2)

# Fixed refinement levels (original behavior)
partition, info = compute_effective_partition_adaptive(F, domain, noise,
                                                       initial_depth=3,
                                                       refinement_levels=2,
                                                       verbose=true)

# Diameter-based stopping (recommended - follows PDF error bounds)
partition, info = compute_effective_partition_adaptive(F, domain, noise,
                                                       initial_depth=3,
                                                       max_diameter=0.01,
                                                       verbose=true)
# Subdivides until all boxes have diameter ≤ 0.01
```
"""
function compute_effective_partition_adaptive(F, initial_domain::IntervalBox{N},
                                             noise::Union{IntervalBox{N}, Nothing}=nothing;
                                             initial_depth::Int=3,
                                             refinement_levels::Union{Int,Nothing}=nothing,
                                             max_diameter::Union{Float64,Nothing}=nothing,
                                             enclosure_method=first_order_enclosure,
                                             safety_margin=1.1,
                                             verbose=false) where {N}

    # Determine stopping criterion
    if max_diameter !== nothing && refinement_levels !== nothing
        @warn "Both max_diameter and refinement_levels specified; using max_diameter"
    end

    use_diameter_stopping = (max_diameter !== nothing)

    if !use_diameter_stopping && refinement_levels === nothing
        error("Must specify either max_diameter or refinement_levels")
    end

    if verbose
        println("Computing effective partition (adaptive refinement)...")
        println("  Initial domain: $initial_domain")
        println("  Initial depth: $initial_depth")
        if use_diameter_stopping
            println("  Stopping criterion: diameter ≤ $max_diameter")
        else
            println("  Refinement levels: $refinement_levels")
        end
    end

    # Start with initial partition
    boxes = [initial_domain]
    for _ in 1:initial_depth
        new_boxes = typeof(initial_domain)[]
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

    if verbose
        println("\nIteration 0: $(length(boxes)) boxes")
    end

    # Iterative refinement
    level = 0
    max_iterations = use_diameter_stopping ? 20 : refinement_levels

    while level < max_iterations
        n_boxes = length(boxes)
        flagged = falses(n_boxes)

        # Forward flagging
        for (i, box) in enumerate(boxes)
            image = enclosure_method(F, box)

            if image isa Tuple
                image_box = IntervalBox(image...)
            elseif image isa IntervalBox
                image_box = image
            else
                error("Unexpected image type: $(typeof(image))")
            end

            if noise !== nothing
                image_box = minkowski_sum(image_box, noise)
            end

            if safety_margin != 1.0
                expanded_intervals = [expand_interval(image_box[i], safety_margin) for i in 1:N]
                image_box = IntervalBox(expanded_intervals...)
            end

            image_box = intersect_boxes(image_box, initial_domain)

            for (j, target_box) in enumerate(boxes)
                if !flagged[j] && intersects(image_box, target_box)
                    flagged[j] = true
                end
            end
        end

        # Extract and subdivide flagged boxes
        effective_boxes = [boxes[i] for i in 1:n_boxes if flagged[i]]
        n_effective = length(effective_boxes)

        if verbose
            println("Iteration $(level+1): $n_effective effective boxes ($(round(100*n_effective/n_boxes, digits=1))% of partition)")
        end

        # Check stopping criterion
        current_diameter = maximum(box_width(box) for box in effective_boxes)

        if use_diameter_stopping && current_diameter <= max_diameter
            if verbose
                println("  Reached target diameter: $current_diameter ≤ $max_diameter")
            end
            boxes = effective_boxes
            break
        end

        # Subdivide effective boxes for next iteration
        new_boxes = typeof(initial_domain)[]
        for box in effective_boxes
            if N == 2
                append!(new_boxes, quadrisect(box))
            else
                left, right = bisect_box(box)
                push!(new_boxes, left, right)
            end
        end
        boxes = new_boxes
        level += 1

        if verbose && use_diameter_stopping
            println("  Current diameter: $(round(current_diameter, digits=6))")
        end
    end

    # Final partition
    n_final = length(boxes)
    total_measure = sum(box_measure(box) for box in boxes)
    final_diameter = maximum(box_width(box) for box in boxes)

    if verbose
        println("\nFinal partition:")
        println("  Boxes: $n_final")
        println("  Partition diameter: $(round(final_diameter, digits=6))")
        println("  Total measure: $(round(total_measure, digits=6))")
        println("  Domain reduction: $(round(box_measure(initial_domain)/total_measure, digits=2))×")
        if N == 2
            final_depth_equivalent = log(4, n_final)  # Equivalent uniform depth in 2D
            println("  Equivalent uniform depth: ~$(round(final_depth_equivalent, digits=1))")
        end
    end

    partition = UlamPartition(boxes, initial_domain)

    info = (
        n_boxes_final = n_final,
        initial_depth = initial_depth,
        refinement_levels = level,  # Actual number of refinements performed
        partition_diameter = final_diameter,
        total_measure = total_measure,
        reduction_factor = box_measure(initial_domain) / total_measure
    )

    return partition, info
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
