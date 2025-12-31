"""
Partition utilities for adaptive box refinement in the rigorous Ulam method.

This module provides functions for:
- Bisecting and quadrisecting interval boxes
- Testing box relationships (subset, disjoint, intersection)
- Computing box measures (area, volume)
- Managing adaptive partitions
"""

using IntervalArithmetic, IntervalBoxes

export bisect_interval, bisect_box, quadrisect
export issubset_box, isdisjoint_box, intersects, is_empty_box
export box_measure, box_width, box_radius

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
