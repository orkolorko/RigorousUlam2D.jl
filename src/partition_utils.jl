"""
Partition utilities for adaptive box refinement in the rigorous Ulam method.

This module provides functions for:
- Bisecting and quadrisecting interval boxes
- Testing box relationships (subset, disjoint, intersection)
- Computing box measures (area, volume)
- Managing adaptive partitions
"""

using IntervalArithmetic, IntervalBoxes
using SparseArrays

export bisect_interval, bisect_box, quadrisect
export issubset_box, isdisjoint_box, intersects, is_empty_box
export box_measure, box_width, box_radius
export expand_interval, intersect_boxes
export build_uniform_partition, build_adaptive_partition

"""
    bisect_interval(I::Interval)

Split an interval I into two equal halves.
Returns a tuple (I_left, I_right).
"""
function bisect_interval(I::Interval)
    m = mid(I)
    return (interval(inf(I), m), interval(m, sup(I)))
end

"""
    bisect_box(box::IntervalBox, dimension::Int)

Bisect a box along the specified dimension (1-indexed).
Returns a tuple of two boxes.
"""
function bisect_box(box::IntervalBox{N}, dimension::Int) where {N}
    intervals = collect(box)
    left_int, right_int = bisect_interval(intervals[dimension])

    left_intervals = copy(intervals)
    left_intervals[dimension] = left_int

    right_intervals = copy(intervals)
    right_intervals[dimension] = right_int

    return (IntervalBox(left_intervals...), IntervalBox(right_intervals...))
end

"""
    bisect_box(box::IntervalBox)

Bisect a box along its widest dimension.
Returns a tuple of two boxes.
"""
function bisect_box(box::IntervalBox{N}) where {N}
    widths = [diam(interval) for interval in box]
    dimension = argmax(widths)
    return bisect_box(box, dimension)
end

"""
    quadrisect(box::IntervalBox{2})

Split a 2D box into 4 equal quadrants.
Returns a vector of 4 boxes ordered as: [SW, SE, NW, NE]
(southwest, southeast, northwest, northeast).
"""
function quadrisect(box::IntervalBox{2})
    x_left, x_right = bisect_interval(box[1])
    y_bottom, y_top = bisect_interval(box[2])

    return [
        IntervalBox(x_left, y_bottom),   # SW
        IntervalBox(x_right, y_bottom),  # SE
        IntervalBox(x_left, y_top),      # NW
        IntervalBox(x_right, y_top)      # NE
    ]
end

"""
    issubset_box(A::IntervalBox, B::IntervalBox)

Check if box A ⊆ B (A is a subset of B).
Returns true if every point in A is also in B.
"""
function issubset_box(A::IntervalBox{N}, B::IntervalBox{N}) where {N}
    if is_empty_box(A)
        return true
    end
    if is_empty_box(B)
        return is_empty_box(A)
    end

    for i in 1:N
        if !(issubset_interval(A[i], B[i]))
            return false
        end
    end
    return true
end

"""
    isdisjoint_box(A::IntervalBox, B::IntervalBox)

Check if boxes A and B are disjoint (have no intersection).
"""
function isdisjoint_box(A::IntervalBox{N}, B::IntervalBox{N}) where {N}
    for i in 1:N
        # Boxes are disjoint if they're separated in any dimension
        if sup(A[i]) < inf(B[i]) || sup(B[i]) < inf(A[i])
            return true
        end
    end
    return false
end

"""
    intersects(A::IntervalBox, B::IntervalBox)

Check if boxes A and B have a non-empty intersection.
"""
intersects(A::IntervalBox, B::IntervalBox) = !isdisjoint_box(A, B)

"""
    is_empty_box(box::IntervalBox)

Check if a box is empty (has zero or negative width in any dimension).
"""
function is_empty_box(box::IntervalBox{N}) where {N}
    for i in 1:N
        if diam(box[i]) <= 0 || isempty_interval(box[i])
            return true
        end
    end
    return false
end

"""
    box_measure(box::IntervalBox)

Compute the Lebesgue measure (area/volume) of a box.
For 2D boxes, this is the area. For nD boxes, this is the n-dimensional volume.
"""
function box_measure(box::IntervalBox{N}) where {N}
    if is_empty_box(box)
        return 0.0
    end
    measure = 1.0
    for i in 1:N
        measure *= diam(box[i])
    end
    return measure
end

"""
    box_width(box::IntervalBox)

Return the maximum width (diameter) across all dimensions.
"""
function box_width(box::IntervalBox{N}) where {N}
    return maximum(diam(interval) for interval in box)
end

"""
    box_radius(box::IntervalBox)

Return the radius of the smallest ball containing the box.
This is half the diagonal length.
"""
function box_radius(box::IntervalBox{N}) where {N}
    sum_sq = sum(diam(interval)^2 for interval in box)
    return sqrt(sum_sq) / 2
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
    build_uniform_partition(domain::IntervalBox{N}, depth::Int) where {N}

Build a uniform partition by subdividing the domain to the specified depth.

For 2D domains, this creates 4^depth boxes via repeated quadrisection.
For higher dimensions, this creates 2^(N*depth) boxes via repeated bisection.

# Arguments
- `domain`: The domain box to partition
- `depth`: Number of subdivision levels (0 = no subdivision, just the domain)

# Returns
- `partition::UlamPartition`: The uniform partition

# Example
```julia
domain = IntervalBox(interval(0.0, 1.0), interval(0.0, 1.0))
partition = build_uniform_partition(domain, 3)
# Creates 4^3 = 64 boxes
```
"""
function build_uniform_partition(domain::IntervalBox{N}, depth::Int) where {N}
    if depth < 0
        error("Depth must be non-negative")
    end

    boxes = [domain]

    for _ in 1:depth
        new_boxes = typeof(domain)[]
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

    return UlamPartition(boxes, domain)
end

"""
    build_adaptive_partition(F, domain::IntervalBox{N},
                            noise::Union{IntervalBox{N}, Nothing}=nothing;
                            initial_depth::Int=2,
                            max_diameter::Union{Float64,Nothing}=nothing,
                            refinement_levels::Union{Int,Nothing}=nothing,
                            enclosure_method=first_order_enclosure,
                            safety_margin=1.1,
                            periodic::Bool=false,
                            verbose=false) where {N}

Build an adaptive partition using iterative forward flagging.

This function performs greedy refinement:
1. Start with a coarse partition
2. For each box, compute F(box) + noise + margin
3. Flag all boxes that can be reached
4. Refine flagged boxes and repeat

Returns both the partition and a boolean transition graph indicating which boxes
can potentially send mass to which other boxes.

# Arguments
- `F`: The dynamical system
- `domain`: The domain box
- `noise`: Noise support box (optional)
- `initial_depth`: Initial partition depth before refinement
- `max_diameter`: Maximum partition diameter (stopping criterion)
- `refinement_levels`: Fixed number of refinement iterations (alternative stopping criterion)
- `enclosure_method`: Method for computing F(box)
- `safety_margin`: Expansion factor for images
- `periodic`: If true, apply periodic boundary conditions (wrap images to torus)
- `verbose`: Print progress information

# Returns
- `partition::UlamPartition`: The adaptive partition
- `transition_graph::SparseMatrixCSC{Bool}`: Boolean matrix where [i,j]=true means box i can send mass to box j

# Example
```julia
F(x, y) = (0.5*x + 0.25, 0.5*y + 0.25)
domain = IntervalBox(interval(0.0, 1.0), interval(0.0, 1.0))
noise = uniform_noise_box(0.01, 2)

# Absorbing boundary conditions (default)
partition, graph = build_adaptive_partition(F, domain, noise,
                                            initial_depth=2,
                                            max_diameter=0.05,
                                            verbose=true)

# Periodic boundary conditions
partition, graph = build_adaptive_partition(F, domain, noise,
                                            initial_depth=2,
                                            max_diameter=0.05,
                                            periodic=true,
                                            verbose=true)
```
"""
function build_adaptive_partition(F, domain::IntervalBox{N},
                                  noise::Union{IntervalBox{N}, Nothing}=nothing;
                                  initial_depth::Int=2,
                                  max_diameter::Union{Float64,Nothing}=nothing,
                                  refinement_levels::Union{Int,Nothing}=nothing,
                                  enclosure_method=first_order_enclosure,
                                  safety_margin=1.1,
                                  periodic::Bool=false,
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
        println("Building adaptive partition using forward flagging...")
        println("  Domain: $domain")
        println("  Initial depth: $initial_depth")
        if use_diameter_stopping
            println("  Stopping criterion: diameter ≤ $max_diameter")
        else
            println("  Refinement levels: $refinement_levels")
        end
    end

    # Start with initial uniform partition
    boxes = [domain]
    for _ in 1:initial_depth
        new_boxes = typeof(domain)[]
        for box in boxes
            left, right = bisect_box(box)
            push!(new_boxes, left, right)
        end
        boxes = new_boxes
    end

    if verbose
        println("  Initial partition: $(length(boxes)) boxes")
    end

    # Iterative refinement with flagging
    level = 0
    max_iterations = use_diameter_stopping ? 20 : refinement_levels

    # Track transition graph during refinement
    I_rows = Int[]  # Source box indices
    J_cols = Int[]  # Target box indices

    while level < max_iterations
        n_boxes = length(boxes)
        flagged = falses(n_boxes)

        # Clear previous iteration's edges
        empty!(I_rows)
        empty!(J_cols)

        # Forward flagging: compute images and track transitions
        for (i, box) in enumerate(boxes)
            # Compute image
            image = enclosure_method(F, box)

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

            # Apply boundary conditions
            if periodic
                # Wrap to torus for periodic BC
                image_box = apply_periodic_bc(image_box, domain)
            end
            # For absorbing BC: don't intersect with domain (graph tracks escapes)

            # Flag all boxes that intersect and record transitions
            for (j, target_box) in enumerate(boxes)
                if intersects(image_box, target_box)
                    flagged[j] = true
                    push!(I_rows, i)
                    push!(J_cols, j)
                end
            end
        end

        # Extract flagged boxes
        effective_boxes = [boxes[i] for i in 1:n_boxes if flagged[i]]
        n_effective = length(effective_boxes)

        if verbose
            println("  Iteration $(level+1): $n_effective effective boxes ($(round(100*n_effective/n_boxes, digits=1))% flagged)")
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

        # Greedy subdivision: only subdivide boxes that violate diameter criterion
        new_boxes = typeof(domain)[]
        for box in effective_boxes
            if use_diameter_stopping && box_width(box) > max_diameter
                # Box still too large, subdivide it
                left, right = bisect_box(box)
                push!(new_boxes, left, right)
            else
                # Box already satisfies criterion (or using refinement_levels), keep as-is
                push!(new_boxes, box)
            end
        end
        boxes = new_boxes
        level += 1

        if verbose && use_diameter_stopping
            println("  Current diameter: $(round(current_diameter, digits=6))")
        end
    end

    # Build final partition
    n_final = length(boxes)
    partition = UlamPartition(boxes, domain)

    # Build transition graph for final partition
    # We need to recompute for the final boxes
    empty!(I_rows)
    empty!(J_cols)

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

        # Apply boundary conditions
        if periodic
            # Wrap to torus for periodic BC
            image_box = apply_periodic_bc(image_box, domain)
        end
        # For absorbing BC: don't intersect with domain (graph tracks escapes)

        for (j, target_box) in enumerate(boxes)
            if intersects(image_box, target_box)
                push!(I_rows, i)
                push!(J_cols, j)
            end
        end
    end

    # Create sparse boolean matrix
    transition_graph = sparse(J_cols, I_rows, trues(length(I_rows)), n_final, n_final)

    if verbose
        nnz_graph = nnz(transition_graph)
        sparsity = 100 * (1 - nnz_graph / (n_final^2))
        println("\nFinal partition:")
        println("  Boxes: $n_final")
        println("  Transition graph: $nnz_graph non-zeros ($(round(sparsity, digits=1))% sparse)")
        println("  Refinement levels: $level")
    end

    return partition, transition_graph
end
