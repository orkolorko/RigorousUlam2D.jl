"""
Adaptive refinement algorithms for computing rigorous transition probabilities.

This module implements the core algorithm for the rigorous Ulam method:
computing the relative size (measure) of points in one partition element
that map into another partition element under a given dynamical system.
"""

using IntervalArithmetic, IntervalBoxes

# Import partition utilities - will be included by main module
# These will be available when this file is included

export rel_size, adaptive_partition

"""
    rel_size(F, Q_source::IntervalBox, Q_target::IntervalBox;
             enclosure_method=first_order_enclosure,
             ε=1e-6, maxdepth=30, depth=0)

Compute a rigorous enclosure of the measure of points in `Q_source`
that map under `F` into `Q_target`.

Returns `(lower_bound, upper_bound, uncertain_boxes)` where:
- `lower_bound`: Guaranteed lower bound on the measure
- `upper_bound`: Guaranteed upper bound on the measure
- `uncertain_boxes`: List of boxes that couldn't be decided (at maxdepth)

# Algorithm
Uses adaptive bisection with three cases at each step:
1. If F(box) ⊆ Q_target: box definitely maps in → add to lower bound
2. If F(box) ∩ Q_target = ∅: box definitely maps out → discard
3. Otherwise: bisect the box and recurse (if depth < maxdepth)

# Arguments
- `F`: The dynamical system (function)
- `Q_source`: Source partition element to analyze
- `Q_target`: Target partition element
- `enclosure_method`: Function to compute image enclosures (default: first_order)
- `ε`: Tolerance for box size (boxes smaller than ε won't be bisected)
- `maxdepth`: Maximum recursion depth
- `depth`: Current recursion depth (internal parameter)

# Example
```julia
F(x, y) = (2*x*(1-x), 0.5*y)  # Logistic × contraction
source = IntervalBox(interval(0.0, 0.5), interval(0.0, 1.0))
target = IntervalBox(interval(0.25, 0.75), interval(0.0, 0.5))
lower, upper, uncertain = rel_size(F, source, target)
```
"""
function rel_size(F, Q_source::IntervalBox{N}, Q_target::IntervalBox{N};
                  enclosure_method=first_order_enclosure,
                  ε=1e-6, maxdepth=30, depth=0) where {N}

    # Base case: empty source
    if box_measure(Q_source) == 0
        return (0.0, 0.0, IntervalBox{N}[])
    end

    # Compute rigorous enclosure of F(Q_source)
    image = enclosure_method(F, Q_source)

    # Convert to IntervalBox if needed (handles scalar and tuple outputs)
    if image isa Tuple
        image_box = IntervalBox(image...)
    elseif image isa Interval
        # For scalar maps, need to know target dimension
        # This shouldn't happen for N-dimensional systems
        error("Scalar output from map, but expected $N-dimensional output")
    else
        image_box = image
    end

    # Case 1: Image definitely contained in target
    if issubset_box(image_box, Q_target)
        measure = box_measure(Q_source)
        return (measure, measure, IntervalBox{N}[])
    end

    # Case 2: Image definitely disjoint from target
    if isdisjoint_box(image_box, Q_target)
        return (0.0, 0.0, IntervalBox{N}[])
    end

    # Case 3: Uncertain - need to refine
    # Check stopping criteria
    if depth >= maxdepth || box_width(Q_source) < ε
        # Can't refine further - add to uncertain belt
        measure = box_measure(Q_source)
        return (0.0, measure, [Q_source])
    end

    # Bisect the source box and recurse
    if N == 2
        sub_boxes = quadrisect(Q_source)
    else
        # For general N, bisect along widest dimension
        sub_boxes = collect(bisect_box(Q_source))
    end

    # Accumulate results from sub-boxes
    total_lower = 0.0
    total_upper = 0.0
    all_uncertain = IntervalBox{N}[]

    for sub_box in sub_boxes
        lower, upper, uncertain = rel_size(F, sub_box, Q_target;
                                           enclosure_method=enclosure_method,
                                           ε=ε, maxdepth=maxdepth, depth=depth+1)
        total_lower += lower
        total_upper += upper
        append!(all_uncertain, uncertain)
    end

    return (total_lower, total_upper, all_uncertain)
end

"""
    adaptive_partition(domain::IntervalBox{N}, F;
                      initial_depth=3, max_refinement_depth=10,
                      enclosure_method=first_order_enclosure) where {N}

Create an adaptive partition of the domain based on the dynamics of F.

Returns a vector of IntervalBox elements representing the partition.

# Algorithm
1. Start with a uniform grid at `initial_depth`
2. Refine boxes where the image enclosure is significantly larger than the box
3. Stop at `max_refinement_depth`

# Arguments
- `domain`: The domain to partition
- `F`: The dynamical system
- `initial_depth`: Initial uniform subdivision depth (2^depth boxes per dimension)
- `max_refinement_depth`: Maximum refinement depth
- `enclosure_method`: Method for computing image enclosures

# Example
```julia
domain = IntervalBox(interval(0.0, 1.0), interval(0.0, 1.0))
F(x, y) = (2*x*(1-x), 0.5*y)
partition = adaptive_partition(domain, F, initial_depth=4)
```
"""
function adaptive_partition(domain::IntervalBox{N}, F;
                           initial_depth=3, max_refinement_depth=10,
                           enclosure_method=first_order_enclosure) where {N}

    # Create initial uniform partition
    partition = IntervalBox{N}[domain]

    for level in 1:initial_depth
        new_partition = IntervalBox{N}[]
        for box in partition
            if N == 2
                append!(new_partition, quadrisect(box))
            else
                left, right = bisect_box(box)
                push!(new_partition, left, right)
            end
        end
        partition = new_partition
    end

    # Adaptive refinement phase
    for level in (initial_depth+1):max_refinement_depth
        refined = false
        new_partition = IntervalBox{N}[]

        for box in partition
            # Compute image enclosure
            image = enclosure_method(F, box)

            # Convert to box for measure computation
            if image isa Tuple
                image_box = IntervalBox(image...)
            else
                image_box = image
            end

            # Refinement criterion: image much larger than source
            source_measure = box_measure(box)
            image_measure = box_measure(image_box)
            expansion_ratio = image_measure / source_measure

            # Refine if expansion is large (heuristic)
            if expansion_ratio > 2.0 && box_width(box) > 1e-6
                refined = true
                if N == 2
                    append!(new_partition, quadrisect(box))
                else
                    left, right = bisect_box(box)
                    push!(new_partition, left, right)
                end
            else
                push!(new_partition, box)
            end
        end

        partition = new_partition

        # Stop if no refinement occurred
        if !refined
            break
        end
    end

    return partition
end
