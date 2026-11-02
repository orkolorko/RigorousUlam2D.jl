"""
The transition graph of a map on the uniform grids of `build_uniform_partition`, refined level
by level, and the Ulam matrices read off the refinement.

The grid of depth `d` has `2^d × 2^d` cells; a `UlamTree` refines the source cells level by
level from a depth `d0` (`refine!`), with the target cells at depth `d1`, and `ulam_matrices`
reads the matrices at any depth up to the current level; `ulam_hierarchy` refines to
`d1 + leaf_depth` in one go:

- a cell at depth `d ≤ d1` carries the list of its candidate targets at depth `d`, the cells
  met by the enclosure of its image, and the children of a cell can only go to the children
  of its candidates, since the true image of a child lies in the true image of the parent;
  that list is the column of the transition graph `G_d`, and `G_{d+1}` is computed from `G_d`
  with one image enclosure per child;
- below `d1` the targets stay at depth `d1` and the candidate list of a child is a subset of
  its parent's;
- a cell at depth `d ≥ d1` whose list has a single entry is resolved: its mass goes to that
  target and to the ancestors of that target at every coarser depth, and it is not subdivided
  again; above `d1` every cell is subdivided, since its targets at depth `d1` are not known;
- a cell at the last level with several candidates is uncertain.

Every leaf of this tree, resolved or uncertain, has a depth `ℓ ≥ d1`, a position, and its
candidate targets at depth `d1`.  The Ulam matrix `L^{(d)}` at a depth `d0 ≤ d ≤ d1`,
`L^{(d)}[i, j] = |I_j ∩ T⁻¹(I_i)| / |I_j|`, is then enclosed by

    L_lo^{(d)} ≤ L^{(d)} ≤ L_lo^{(d)} + U^{(d)}   entrywise,

where a leaf in `I_j`, of mass `4^{-(ℓ - d)}` relative to `I_j`, contributes to
`L_lo^{(d)}[i, j]` when the ancestors at depth `d` of its candidates are the single cell `I_i`,
and to `U^{(d)}[i, j]` for each ancestor `I_i` otherwise.  All entries are exact dyadic
numbers, no rounding enters, and the column sums of `U^{(d)}` are at least the uncertain mass
of each source cell.

Boundary conditions are `:periodic` or `:reflecting`.  Nothing is wrapped: an image interval
is translated by an integer multiple of the period (`L`, or `2L` when reflecting, in which
case the part beyond the domain is reflected back) in interval arithmetic and compared with
the grid lines, which are exact floating-point numbers (checked once in `_Axis`).  The image
of a box is the intersection of the natural interval extension of `F` with the enclosure of
`enclosure_method`; both enclose the true image, so does their intersection.
"""

export transition_graph, ulam_hierarchy, UlamTree, refine!, ulam_matrices

# ---------------------------------------------------------------------------------------------
# One axis of a grid: n cells of width dx on [x0, x0 + L], with the grid lines x0 + i dx exact.
# ---------------------------------------------------------------------------------------------
struct _Axis
    x0::Float64
    L::Float64
    n::Int
    dx::Float64
    bc::Symbol
end

function _Axis(I::Interval, n::Int, bc::Symbol)
    bc in (:periodic, :reflecting) ||
        throw(ArgumentError("boundary condition must be :periodic or :reflecting, got :$bc"))
    x0, x1 = inf(I), sup(I)
    L = x1 - x0
    dx = L / n
    R(x) = Rational{BigInt}(x)
    R(L) == R(x1) - R(x0) || throw(ArgumentError("the width of $I is not exact in Float64"))
    R(dx) * n == R(L) || throw(ArgumentError("the cell width $L/$n is not exact in Float64"))
    for i in 0:n
        R(x0 + i * dx) == R(x0) + i * R(dx) ||
            throw(ArgumentError("the grid line x0 + $i dx is not exact in Float64"))
    end
    for k in -8:8
        R(k * L) == k * R(L) || throw(ArgumentError("the translate $k L is not exact in Float64"))
        R(x0 + k * L) == R(x0) + k * R(L) ||
            throw(ArgumentError("the point x0 + $k L is not exact in Float64"))
    end
    return _Axis(x0, L, n, dx, bc)
end

_gridline(ax::_Axis, i::Int) = ax.x0 + i * ax.dx
_cell(ax::_Axis, i::Int) = interval(_gridline(ax, i - 1), _gridline(ax, i))

# X translated by a multiple of `period` so that inf lands in [x0, x0 + period)
function _shift(X::Interval, x0, period)
    k = floor(Int, (inf(X) - x0) / period)
    Xs = X - interval(k * period)
    while inf(Xs) < x0
        k -= 1; Xs = X - interval(k * period)
    end
    while inf(Xs) >= x0 + period
        k += 1; Xs = X - interval(k * period)
    end
    return Xs
end

# cells met by [a, b] with x0 ≤ a ≤ b; b may lie past x1 (periodic), so ihi may exceed n
function _range_in(ax::_Axis, a, b)
    (; x0, n, dx) = ax
    ilo = clamp(floor(Int, (a - x0) / dx) + 1, 1, n)
    while ilo > 1 && _gridline(ax, ilo - 1) > a
        ilo -= 1
    end
    while ilo < n && _gridline(ax, ilo) <= a
        ilo += 1
    end
    ihi = max(ceil(Int, (b - x0) / dx), ilo)
    while ihi > ilo && _gridline(ax, ihi - 1) >= b
        ihi -= 1
    end
    while _gridline(ax, ihi) < b
        ihi += 1
    end
    return ilo, ihi
end

"""
    _cell_range(ax::_Axis, X::Interval) -> (ilo, ihi, all)

The cells of the axis met by the image interval `X` after the boundary condition: the cells
`ilo ≤ i ≤ ihi`, taken `mod1 n` (periodic: `ihi` may exceed `n`); `all` means every cell.
Reflecting: `X` is translated by a multiple of `2L` so that it starts in `[x0, x0 + 2L)`, its
part in `[x0 + L, x0 + 2L]` is reflected about `x0 + L` and the part beyond is translated
back; the pieces share an endpoint of the domain, so their cells form one range.
"""
function _cell_range(ax::_Axis, X::Interval)
    (; x0, L, n) = ax
    diam(X) >= L && return (1, n, true)
    if ax.bc === :periodic
        Xs = _shift(X, x0, L)
        diam(Xs) >= L && return (1, n, true)
        ilo, ihi = _range_in(ax, inf(Xs), sup(Xs))
    else
        Xs = _shift(X, x0, 2L)
        x1, x2 = x0 + L, x0 + 2L
        a, b = inf(Xs), sup(Xs)
        ilo, ihi = n + 1, 0
        if a < x1                                               # direct part
            r = _range_in(ax, a, min(b, x1))
            ilo, ihi = min(ilo, r[1]), max(ihi, min(r[2], n))
        end
        if b > x1                                               # reflected part
            P = interval(2x1) - interval(max(a, x1), min(b, x2))
            r = _range_in(ax, max(inf(P), x0), min(sup(P), x1))
            ilo, ihi = min(ilo, r[1]), max(ihi, min(r[2], n))
        end
        if b > x2                                               # part beyond, translated back
            P = interval(x2, b) - interval(2L)
            r = _range_in(ax, max(inf(P), x0), min(sup(P), x1))
            ilo, ihi = min(ilo, r[1]), max(ihi, min(r[2], n))
        end
    end
    ihi - ilo + 1 >= n && return (1, n, true)
    return (ilo, ihi, false)
end

_isfinite(I::Interval) = isfinite(inf(I)) && isfinite(sup(I))

function _meet(A::Interval, B::Interval)
    lo = max(inf(A), inf(B)); hi = min(sup(A), sup(B))
    lo <= hi || error("two enclosures of the same image are disjoint: $A and $B")
    return interval(lo, hi)
end

"""
    _image(F, box, enclosure_method) -> (X, Y)

Intersection of the natural extension `F(box...)` with `enclosure_method(F, box)`; the latter is
dropped when it throws or is not finite (a mean-value form where `F` is not differentiable).
"""
function _image(F, box::IntervalBox{2}, enclosure_method)
    nat = F(box[1], box[2])
    X, Y = nat[1], nat[2]
    enclosure_method === zeroth_order_enclosure && return (X, Y)
    e = try
        enclosure_method(F, box)
    catch
        nothing
    end
    e === nothing && return (X, Y)
    E1, E2 = e[1], e[2]
    (_isfinite(E1) && _isfinite(E2)) || return (X, Y)
    return (_meet(X, E1), _meet(Y, E2))
end

# ---------------------------------------------------------------------------------------------
# Grids: cells at depth d are addressed by (ix, iy) ∈ {1..2^d}², linear id (iy - 1) 2^d + ix.
# ---------------------------------------------------------------------------------------------
struct _Grid
    depth::Int
    ax::_Axis
    ay::_Axis
end

function _Grid(domain::IntervalBox{2}, depth::Int, bc::Symbol)
    n = 2^depth
    return _Grid(depth, _Axis(domain[1], n, bc), _Axis(domain[2], n, bc))
end

_id(g::_Grid, ix, iy) = (mod1(iy, g.ay.n) - 1) * g.ax.n + mod1(ix, g.ax.n)
_xy(g::_Grid, id) = (mod1(id, g.ax.n), (id - 1) ÷ g.ax.n + 1)
_box(g::_Grid, ix, iy) = IntervalBox(_cell(g.ax, ix), _cell(g.ay, iy))
# the ancestor at depth d of the cell `id` of grid g (d ≤ g.depth)
function _ancestor(g::_Grid, id, d)
    ix, iy = _xy(g, id)
    s = g.depth - d
    return ((iy - 1) >> s) * (g.ax.n >> s) + ((ix - 1) >> s) + 1
end

# Cells of g met by the image (X, Y).
function _targets!(ids::Vector{Int}, g::_Grid, X::Interval, Y::Interval)
    empty!(ids)
    ixlo, ixhi, _ = _cell_range(g.ax, X)
    iylo, iyhi, _ = _cell_range(g.ay, Y)
    for iy in iylo:iyhi, ix in ixlo:ixhi
        push!(ids, _id(g, ix, iy))
    end
    return ids
end

"""
    transition_graph(F, domain, depth; bc=:periodic, enclosure_method=first_order_enclosure)

The transition graph at one depth, with no refinement: `G[i, j] = true` when the enclosure of
`F(I_j)` meets `I_i`.  Cells are indexed as in `build_uniform_partition(domain, depth)`.
"""
function transition_graph(F, domain::IntervalBox{2}, depth::Int;
        bc::Symbol = :periodic, enclosure_method = first_order_enclosure)
    h = ulam_hierarchy(F, domain, depth, depth; leaf_depth = 0, bc, enclosure_method)
    return h[1].graph
end

# A node of the refinement: a source cell at depth `depth`, and its candidate targets at
# depth min(depth, d1).
struct _Node
    depth::Int
    id::Int
    cands::Vector{Int}
end

"""
    UlamTree(F, domain, d0, d1; bc=:periodic, enclosure_method=first_order_enclosure)

The refinement of the file docstring as a mutable object: `refine!(tree)` descends one level,
`tree.level` is the depth of the active cells, and `ulam_matrices(tree, d)` reads the graph and
the matrices at any depth `d0 ≤ d ≤ min(tree.level, d1)` off the current level, so a long
refinement can hand out the coarse matrices as it goes and each level halves their widths.
`ulam_hierarchy` is the one-shot form.
"""
mutable struct UlamTree
    F
    domain::IntervalBox{2}
    d0::Int
    d1::Int
    bc::Symbol
    enclosure_method
    grids::Vector{_Grid}        # grids[d + 1] has depth d, grown as the tree descends
    level::Int                  # depth of the active cells
    active::Vector{_Node}       # cells with several candidates, or above d1
    leaves::Vector{_Node}       # resolved cells: one candidate at depth d1
end

function UlamTree(F, domain::IntervalBox{2}, d0::Int, d1::Int;
        bc::Symbol = :periodic, enclosure_method = first_order_enclosure)
    0 <= d0 <= d1 || throw(ArgumentError("need 0 ≤ d0 ≤ d1, got $d0, $d1"))
    grids = [_Grid(domain, d, bc) for d in 0:d0]
    g0 = grids[d0 + 1]
    gt = grids[min(d0, d1) + 1]
    active = Vector{_Node}(undef, g0.ax.n * g0.ay.n)
    Threads.@threads :dynamic for id in eachindex(active)
        X, Y = _image(F, _box(g0, _xy(g0, id)...), enclosure_method)
        active[id] = _Node(d0, id, _targets!(Int[], gt, X, Y))
    end
    tree = UlamTree(F, domain, d0, d1, bc, enclosure_method, grids, d0, active, _Node[])
    _split!(tree)
    return tree
end

function _grid(tree::UlamTree, d::Int)
    while length(tree.grids) <= d
        push!(tree.grids, _Grid(tree.domain, length(tree.grids), tree.bc))
    end
    return tree.grids[d + 1]
end

"""
    refine!(tree::UlamTree) -> tree

Quadrisect every active cell: the candidates of a child are the cells met by the enclosure of
its image among the children of its parent's candidates (above `d1`), or among its parent's
candidates (at or below `d1`); children with a single candidate at depth `d1` are resolved.
"""
function refine!(tree::UlamTree)
    d, d1 = tree.level, tree.d1
    gp, gs = _grid(tree, d), _grid(tree, d + 1)         # parents' and children's grids
    gc = _grid(tree, min(d + 1, d1))                     # children's target grid
    refine_targets = d + 1 <= d1
    F, encl = tree.F, tree.enclosure_method
    children = Vector{Vector{_Node}}(undef, length(tree.active))
    Threads.@threads :dynamic for k in eachindex(children)
        node = tree.active[k]
        ix, iy = _xy(gp, node.id)
        out = _Node[]
        ids = Int[]
        for (cx, cy) in ((2ix - 1, 2iy - 1), (2ix, 2iy - 1), (2ix - 1, 2iy), (2ix, 2iy))
            X, Y = _image(F, _box(gs, cx, cy), encl)
            _targets!(ids, gc, X, Y)
            if refine_targets
                cands = [i for i in ids if _ancestor(gc, i, d) in node.cands]
            else
                cands = [i for i in ids if i in node.cands]
            end
            push!(out, _Node(d + 1, _id(gs, cx, cy), cands))
        end
        children[k] = out
    end
    tree.active = isempty(children) ? _Node[] : reduce(vcat, children)
    tree.level = d + 1
    _split!(tree)
    return tree
end

# Move the resolved active cells (one candidate at depth d1, which needs level ≥ d1) to the
# leaves.  A cell at a depth d < d1 with a single candidate at depth d keeps refining, since its
# targets at depth d1 are not known yet.
function _split!(tree::UlamTree)
    tree.level >= tree.d1 || return tree
    keep = _Node[]
    for node in tree.active
        isempty(node.cands) && error("a cell has no candidate target: the map leaves the domain")
        length(node.cands) == 1 ? push!(tree.leaves, node) : push!(keep, node)
    end
    tree.active = keep
    return tree
end

"""
    ulam_matrices(tree::UlamTree, d) -> (depth, partition, graph, L_lo, U)

The graph and the Ulam matrix enclosure at depth `d`, `d0 ≤ d ≤ min(tree.level, d1)`, read off
the current level: every cell of the tree, resolved or active, contributes its mass `4^{-(ℓ-d)}`
relative to its ancestor at depth `d` to `L_lo` when the ancestors at depth `d` of its
candidates are a single cell and to `U` for each of them otherwise.  Matrices are indexed as
`partition = build_uniform_partition(domain, d)`.
"""
function ulam_matrices(tree::UlamTree, d::Int)
    tree.d0 <= d <= min(tree.level, tree.d1) ||
        throw(ArgumentError("depth $d is not in $(tree.d0):$(min(tree.level, tree.d1))"))
    gd = _grid(tree, d)
    n = gd.ax.n * gd.ay.n
    lo = [Dict{Int,Float64}() for _ in 1:n]
    up = [Dict{Int,Float64}() for _ in 1:n]
    for node in Iterators.flatten((tree.leaves, tree.active))
        gl = _grid(tree, node.depth)
        gt = _grid(tree, min(node.depth, tree.d1))
        anc = unique(_ancestor(gt, c, d) for c in node.cands)
        j = _ancestor(gl, node.id, d)
        mass = 4.0^(-(node.depth - d))
        if length(anc) == 1
            lo[j][anc[1]] = get(lo[j], anc[1], 0.0) + mass
        else
            for i in anc
                up[j][i] = get(up[j], i, 0.0) + mass
            end
        end
    end
    I = Int[]; J = Int[]; V = Float64[]; UI = Int[]; UJ = Int[]; UV = Float64[]
    for j in 1:n
        for (i, v) in lo[j]
            push!(I, i); push!(J, j); push!(V, v)
        end
        for (i, v) in up[j]
            push!(UI, i); push!(UJ, j); push!(UV, v)
        end
    end
    partition = build_uniform_partition(tree.domain, d)
    perm = _box_permutation(partition, gd)         # perm[id] = box index in the partition
    L_lo = sparse(perm[I], perm[J], V, n, n)
    U = sparse(perm[UI], perm[UJ], UV, n, n)
    graph = sparse(perm[vcat(I, UI)], perm[vcat(J, UJ)], true, n, n)
    return (; depth = d, partition, graph, L_lo, U)
end

"""
    ulam_hierarchy(F, domain, d0, d1; leaf_depth=8, bc=:periodic,
                   enclosure_method=first_order_enclosure)

Transition graphs and Ulam matrix enclosures at the depths `d0:d1` (see the file docstring):
a `UlamTree` refined to depth `d1 + leaf_depth`, read at each depth with `ulam_matrices`.
Returns a vector with one named tuple `(depth, partition, graph, L_lo, U)` per depth.  The
uncertain mass of a cell at depth `d` is at most the mass of a strip of width
`diam(domain) 2^{-(d1 + leaf_depth)}` around the preimages of the grid lines of depth `d1`.
"""
function ulam_hierarchy(F, domain::IntervalBox{2}, d0::Int, d1::Int; leaf_depth::Int = 8,
        bc::Symbol = :periodic, enclosure_method = first_order_enclosure)
    leaf_depth >= 0 || throw(ArgumentError("leaf_depth must be nonnegative"))
    tree = UlamTree(F, domain, d0, d1; bc, enclosure_method)
    while tree.level < d1 + leaf_depth
        refine!(tree)
    end
    return [ulam_matrices(tree, d) for d in d0:d1]
end

# Map the grid ids of g to the box indices of `partition` (built by build_uniform_partition).
function _box_permutation(partition::UlamPartition{2}, g::_Grid)
    n = g.ax.n * g.ay.n
    perm = zeros(Int, n)
    for (k, box) in enumerate(partition.boxes)
        ix = round(Int, (inf(box[1]) - g.ax.x0) / g.ax.dx) + 1
        iy = round(Int, (inf(box[2]) - g.ay.x0) / g.ay.dx) + 1
        (inf(box[1]) == _gridline(g.ax, ix - 1) && sup(box[1]) == _gridline(g.ax, ix) &&
         inf(box[2]) == _gridline(g.ay, iy - 1) && sup(box[2]) == _gridline(g.ay, iy)) ||
            error("box $k of the partition is not a cell of the grid of depth $(g.depth)")
        perm[_id(g, ix, iy)] = k
    end
    all(>(0), perm) || error("the partition does not cover the grid of depth $(g.depth)")
    return perm
end
