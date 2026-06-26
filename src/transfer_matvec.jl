"""
Pure matrix-free transfer operator by forward-image scatter.

The deterministic Ulam operator is built WITHOUT a precomputed transition graph and
WITHOUT an all-pairs scan: each source cell is subdivided, the rigorous image of every
leaf is enclosed, and the leaf's mass is scattered (by area overlap) onto the target
cells the image covers. The reachable targets are discovered FROM the images — exactly
the information the dynamical graph used to memoize. Iterating the uniform density makes
this concrete: the support of Pᵏ𝟙 spreads along the reachable set on its own, so the
graph is redundant once we work matvec-only.

The noisy operator is then applied as a matvec  v ↦ N(P_det v)  with N a convolution
(`noise_matvec`); the composed P = N·P_det and the dense noise operator N are never
formed. This is what lets the partition get fine without the storage blowing up: P_det
is sparse (O(n)), N is a stencil, and nothing O(n²) or O(n·(ξ/δ)²) is ever materialized.

This is the Float64 *candidate* operator (forward-scatter by image-box overlap). The
rigorous lower/upper version layers on top of the same scatter — deferred.
"""

using IntervalArithmetic, IntervalBoxes
using SparseArrays

export build_transfer_forward, transfer_operator_matvec

# overlap length of [a,b] ∩ [c,d]
_overlap1d(a, b, c, d) = max(0.0, min(b, d) - max(a, c))

"""
    build_transfer_forward(F, partition, grid_dims; subdiv, enclosure_method, periodic)

Build the deterministic Ulam transfer matrix `P_det` (sparse, column-stochastic after
`normalize_columns!`) on the uniform `nx×ny` grid by forward-image scatter. No transition
graph, no all-pairs scan: cost is `O(n · 2^subdiv · cells_touched)`.

- `subdiv`  : number of bisections of each source cell (more ⇒ sharper images; `2^subdiv`
              leaves per source cell). Use a few extra over the grid resolution.
- `periodic`: fold target cell indices onto the torus (else absorbing — mass leaving the
              domain is lost, then columns are renormalized).
"""
function build_transfer_forward(F, partition::UlamPartition{2}, grid_dims::Tuple{Int, Int};
        subdiv::Int = 4,
        enclosure_method = first_order_enclosure,
        periodic::Bool = false)
    nx, ny = grid_dims
    n = nx * ny
    n == partition.n_boxes ||
        error("partition has $(partition.n_boxes) boxes, grid is $nx×$ny = $n")
    domain = partition.domain
    x0, y0 = inf(domain[1]), inf(domain[2])
    dx = diam(domain[1]) / nx
    dy = diam(domain[2]) / ny
    _, grid_to_box = build_grid_index_maps(partition, grid_dims)

    nt = Threads.maxthreadid()
    tI = [Int[] for _ in 1:nt]
    tJ = [Int[] for _ in 1:nt]
    tV = [Float64[] for _ in 1:nt]

    Threads.@threads :static for j in 1:n
        tid = Threads.threadid()
        S = partition.boxes[j]

        # subdivide the source cell into 2^subdiv equal-measure leaves
        leaves = [S]
        for _ in 1:subdiv
            nl = eltype(leaves)[]
            for b in leaves
                l, r = bisect_box(b)
                push!(nl, l, r)
            end
            leaves = nl
        end
        wleaf = 1.0 / length(leaves)

        acc = Dict{Int, Float64}()       # target cell → mass from this column
        for s in leaves
            img = enclosure_method(F, s)
            ib = img isa Tuple ? IntervalBox(img...) : img
            xa, xb = inf(ib[1]), sup(ib[1])
            ya, yb = inf(ib[2]), sup(ib[2])
            area = (xb - xa) * (yb - ya)

            # cell index range the (unwrapped) image box spans; fold/clip when assigning
            ix_lo = Int(floor((xa - x0) / dx)) + 1
            ix_hi = Int(floor((xb - x0) / dx)) + 1
            iy_lo = Int(floor((ya - y0) / dy)) + 1
            iy_hi = Int(floor((yb - y0) / dy)) + 1

            if area <= 0 || (ix_hi - ix_lo) > nx || (iy_hi - iy_lo) > ny
                # degenerate (point/line image) or image wider than a full period:
                # drop the whole leaf into the cell holding the image midpoint.
                cx = (xa + xb) / 2
                cy = (ya + yb) / 2
                ix = Int(floor((cx - x0) / dx)) + 1
                iy = Int(floor((cy - y0) / dy)) + 1
                ixw, iyw = _fold(ix, iy, nx, ny, periodic)
                ixw == 0 && continue
                k = grid_to_box[iyw, ixw]
                k != 0 && (acc[k] = get(acc, k, 0.0) + wleaf)
                continue
            end

            for ix in ix_lo:ix_hi
                cell_xa = x0 + (ix - 1) * dx
                ox = _overlap1d(xa, xb, cell_xa, cell_xa + dx)
                ox <= 0 && continue
                for iy in iy_lo:iy_hi
                    cell_ya = y0 + (iy - 1) * dy
                    oy = _overlap1d(ya, yb, cell_ya, cell_ya + dy)
                    oy <= 0 && continue
                    ixw, iyw = _fold(ix, iy, nx, ny, periodic)
                    ixw == 0 && continue                 # absorbed (outside domain)
                    k = grid_to_box[iyw, ixw]
                    k == 0 && continue
                    acc[k] = get(acc, k, 0.0) + wleaf * (ox * oy) / area
                end
            end
        end

        for (k, w) in acc
            w <= 0 && continue
            push!(tI[tid], k)
            push!(tJ[tid], j)
            push!(tV[tid], w)
        end
    end

    P = sparse(reduce(vcat, tI), reduce(vcat, tJ), reduce(vcat, tV), n, n)
    return P
end

# fold a (possibly out-of-range) cell index onto the torus (periodic) or report
# absorption (ixw=0) for absorbing BC.
@inline function _fold(ix, iy, nx, ny, periodic)
    if periodic
        return (mod1(ix, nx), mod1(iy, ny))
    else
        (1 <= ix <= nx && 1 <= iy <= ny) ? (ix, iy) : (0, 0)
    end
end

"""
    transfer_operator_matvec(P_det, noise, partition, grid_dims; periodic)

Return the composed matvec closure `v ↦ N(P_det · v)`, with the noise convolution `N`
applied on the fly (`noise_matvec`). Neither `N` nor the composed `P = N·P_det` is formed,
so memory stays O(nnz(P_det)) + O(n) regardless of the noise width.
"""
function transfer_operator_matvec(P_det, noise::IntervalBox{2}, partition::UlamPartition{2},
        grid_dims::Tuple{Int, Int}; periodic::Bool = false)
    b2g, g2b = build_grid_index_maps(partition, grid_dims)
    return v -> noise_matvec(P_det * v, noise, partition;
        grid_dims = grid_dims, periodic = periodic,
        box_to_grid = b2g, grid_to_box = g2b)
end
