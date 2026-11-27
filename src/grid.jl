"""
The uniform dyadic grids on a rectangle and the two orders in which their cells are indexed.

`build_uniform_partition(domain, d)` quadrisects `domain` `d` times and keeps the cells in the
order of the recursion; the matrices of `ulam_matrices` and the blocks of `bench/tree_deepen.jl`
are indexed in that order.  `grid_order(domain, d)` is the permutation to the grid order of the
noisy operator (`noisy_operator.jl`), `v[(ix - 1) k + iy]` for the cell in column `ix` and row
`iy` of the `k × k` grid, `k = 2^d`.
"""

export UlamPartition, bisect_interval, bisect_box, quadrisect, build_uniform_partition,
    grid_order, wrap_to_torus

"""
    UlamPartition(boxes, domain)

The cells `boxes` (a vector of `IntervalBox`) of a partition of `domain`, with their number in
`n_boxes`.
"""
struct UlamPartition{N,T,I}
    boxes::Vector{IntervalBox{N,T,I}}
    domain::IntervalBox{N,T,I}
    n_boxes::Int

    function UlamPartition(boxes::Vector{IntervalBox{N,T,I}}, domain::IntervalBox{N,T,I}) where {N,T,I}
        new{N,T,I}(boxes, domain, length(boxes))
    end
end

"""
    bisect_interval(I::Interval) -> (I_left, I_right)

The two halves of `I`, split at `mid(I)`.
"""
function bisect_interval(I::Interval)
    m = mid(I)
    return (interval(inf(I), m), interval(m, sup(I)))
end

"""
    bisect_box(box, dimension) -> (left, right)
    bisect_box(box) -> (left, right)

The two halves of `box` along the coordinate `dimension`, or along its widest coordinate.
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

function bisect_box(box::IntervalBox{N}) where {N}
    widths = [diam(interval) for interval in box]
    return bisect_box(box, argmax(widths))
end

"""
    quadrisect(box::IntervalBox{2}) -> [SW, SE, NW, NE]

The four quadrants of a rectangle, in that order.
"""
function quadrisect(box::IntervalBox{2})
    x_left, x_right = bisect_interval(box[1])
    y_bottom, y_top = bisect_interval(box[2])
    return [IntervalBox(x_left, y_bottom), IntervalBox(x_right, y_bottom),
            IntervalBox(x_left, y_top), IntervalBox(x_right, y_top)]
end

"""
    build_uniform_partition(domain, depth) -> UlamPartition

The uniform partition of `domain` into `4^depth` cells by `depth` quadrisections (or `2^(N depth)`
by bisection along the widest coordinate when `N ≠ 2`), in the order of the recursion.  The cells
of a dyadic domain have exact floating-point ends.
"""
function build_uniform_partition(domain::IntervalBox{N}, depth::Int) where {N}
    depth < 0 && error("Depth must be non-negative")
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

# box_to_grid[k] = (ix, iy) and grid_to_box[iy, ix] = k for the cells k of a uniform partition
# on an nx × ny grid, read off the midpoints of the cells.  (Moved from noise_convolution.jl,
# where it was build_grid_index_maps; unchanged.)
function _grid_index_maps(partition::UlamPartition{2}, grid_dims::Tuple{Int,Int})
    nx, ny = grid_dims
    n = nx * ny
    domain = partition.domain
    dx = diam(domain[1]) / nx
    dy = diam(domain[2]) / ny
    box_to_grid = Vector{Tuple{Int,Int}}(undef, n)
    grid_to_box = zeros(Int, ny, nx)
    for k in 1:n
        box = partition.boxes[k]
        cx = (inf(box[1]) + sup(box[1])) / 2
        cy = (inf(box[2]) + sup(box[2])) / 2
        ix = clamp(Int(floor((cx - inf(domain[1])) / dx)) + 1, 1, nx)
        iy = clamp(Int(floor((cy - inf(domain[2])) / dy)) + 1, 1, ny)
        box_to_grid[k] = (ix, iy)
        grid_to_box[iy, ix] = k
    end
    return box_to_grid, grid_to_box
end

"""
    grid_order(domain, depth) -> perm

The permutation from the order of `build_uniform_partition(domain, depth)` to the grid order:
`v[perm]` and `L[perm, perm]` are in grid order when `v` and `L` are indexed as the partition
(`perm[(ix - 1) k + iy]` is the index in the partition of the cell `(ix, iy)`).
"""
function grid_order(domain::IntervalBox{2}, depth::Int)
    k = 2^depth
    _, grid_to_box = _grid_index_maps(build_uniform_partition(domain, depth), (k, k))
    return vec(grid_to_box)                 # grid_to_box[iy, ix], column major
end

"""
    wrap_to_torus(x::Real, xmin, xmax)

`x` translated by a multiple of `xmax − xmin` into `[xmin, xmax)`, in floating point: for the
pictures of the maps (`paper/dynamics.jl`), not for enclosures.
"""
function wrap_to_torus(x::Real, domain_min::Real, domain_max::Real)
    period = domain_max - domain_min
    return mod(x - domain_min, period) + domain_min
end
