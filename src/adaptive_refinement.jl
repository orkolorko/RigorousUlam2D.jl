"""
Adaptive refinement algorithms for computing rigorous transition probabilities.

This module implements the core algorithm for the rigorous Ulam method:
computing the relative size (measure) of points in one partition element
that map into another partition element under a given dynamical system.

The implementation uses an iterative greedy algorithm with a priority queue,
processing largest boxes first to avoid stack overflow and provide better
control over the refinement process.
"""

using IntervalArithmetic, IntervalBoxes
using DataStructures  # For PriorityQueue

export rel_size

"""
    rel_size(F, Q_source::IntervalBox, Q_target::IntervalBox;
             enclosure_method=first_order_enclosure,
             ε=1e-6, max_iterations=100000)

Compute a rigorous enclosure of the relative measure of points in `Q_source`
that map under `F` into `Q_target`.

Returns `(lower_bound, upper_bound, uncertain_boxes)` where:
- `lower_bound`: Guaranteed lower bound on the relative measure (in [0,1])
- `upper_bound`: Guaranteed upper bound on the relative measure (in [0,1])
- `uncertain_boxes`: List of boxes that couldn't be decided

# Algorithm
Uses an iterative greedy algorithm with a priority queue:
1. Start with Q_source in the queue
2. Always process the largest box first (greedy)
3. For each box, compute F(box):
   - If F(box) ⊆ Q_target: add measure to lower bound
   - If F(box) ∩ Q_target = ∅: discard
   - If box width < ε: add to uncertain (upper bound only)
   - Otherwise: bisect and add sub-boxes to queue
4. Stop when queue is empty or max_iterations reached

# Arguments
- `F`: The dynamical system (function)
- `Q_source`: Source partition element to analyze
- `Q_target`: Target partition element
- `enclosure_method`: Function to compute image enclosures (default: first_order)
- `ε`: Tolerance for box size (boxes smaller than ε are added to uncertain)
- `max_iterations`: Maximum number of boxes to process

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
                  ε=1e-6, max_iterations=100000) where {N}

    # Base case: empty source
    source_measure = box_measure(Q_source)
    if source_measure == 0
        return (0.0, 0.0, IntervalBox{N}[])
    end

    # Initialize accumulators
    lower_measure = 0.0  # Absolute measure that definitely maps in
    uncertain_boxes = IntervalBox{N}[]  # Boxes we couldn't decide

    # Priority queue: process largest boxes first
    # PriorityQueue is a min-heap, so we use negative measures to get max-heap behavior
    # (larger boxes have more negative priorities, so they're dequeued first)
    queue = PriorityQueue{IntervalBox{N}, Float64}()
    push!(queue, Q_source => -box_measure(Q_source))

    iterations = 0

    while !isempty(queue) && iterations < max_iterations
        iterations += 1

        # Get the largest box
        box = popfirst!(queue).first
        box_vol = box_measure(box)

        # Compute rigorous enclosure of F(box)
        image = enclosure_method(F, box)

        # Convert to IntervalBox if needed
        if image isa Tuple
            image_box = IntervalBox(image...)
        elseif image isa Interval
            error("Scalar output from map, but expected $N-dimensional output")
        else
            image_box = image
        end

        # Case 1: Image definitely contained in target
        if issubset_box(image_box, Q_target)
            lower_measure += box_vol
            continue
        end

        # Case 2: Image definitely disjoint from target
        if isdisjoint_box(image_box, Q_target)
            continue  # Discard this box
        end

        # Case 3: Uncertain - need to refine or give up
        if box_width(box) < ε
            # Box too small to refine - add to uncertain belt
            push!(uncertain_boxes, box)
            continue
        end

        # Bisect the box and add sub-boxes to queue
        left, right = bisect_box(box)
        push!(queue, left => -box_measure(left))
        push!(queue, right => -box_measure(right))
    end

    # Handle remaining boxes in queue (if max_iterations reached)
    while !isempty(queue)
        box = popfirst!(queue).first
        push!(uncertain_boxes, box)
    end

    # Compute uncertain measure
    uncertain_measure = sum(box_measure(box) for box in uncertain_boxes; init=0.0)

    # Return relative measures
    lower_rel = lower_measure / source_measure
    upper_rel = (lower_measure + uncertain_measure) / source_measure

    return (lower_rel, upper_rel, uncertain_boxes)
end
