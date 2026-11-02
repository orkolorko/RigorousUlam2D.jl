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

The tree is kept on disk, one subtree per cell of depth `d0` (see `UlamTree`), so a
refinement needs memory for the subtrees being refined only and resumes where it stopped.

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

# ---------------------------------------------------------------------------------------------
# The tree lives on disk, one subtree per cell of depth d0, so that nothing but the subtree
# being refined is in memory and a refinement interrupted at any point resumes from the
# subtrees already written.

# The state of one subtree: its uncertain cells (the frontier), each at its own depth
# `levels[k] ≥ d0` with the candidate targets `cands[candptr[k]:candptr[k+1]-1]` at depth
# `min(levels[k], d1)`, and the mass already resolved, one accumulator per depth `d` in
# `d0:d1` keyed by `(j - 1) 4^d + i` for the entry `(i, j)` of `L_lo^{(d)}`.
struct _Subtree
    root::Int
    levels::Vector{Int}
    ids::Vector{Int}
    candptr::Vector{Int}
    cands::Vector{Int}
    resolved::Vector{Dict{Int,Float64}}
    nresolved::Int
    depth::Int                  # the finest depth reached, resolved cells included
end

_nactive(sub::_Subtree) = length(sub.ids)

"""
    UlamTree(F, domain, d0, d1; bc=:periodic, enclosure_method=first_order_enclosure,
             store=mktempdir())

The refinement of the file docstring as an object kept on disk: `refine!(tree)` descends one
level, `tree.level` is the depth of the coarsest uncertain cell, and `ulam_matrices(tree, d)`
reads the graph and the matrices at any depth `d0 ≤ d ≤ min(tree.level, d1)`, so a long
refinement hands out the coarse matrices as it goes and each level halves their widths.

The tree is split into one subtree per cell of depth `d0`, each in its own file
`root<k>.jls` of the directory `store`, written whole after every refinement of that
subtree; only the subtrees being refined are in memory.  Constructing a tree on a `store`
that already holds one with the same domain, depths and boundary condition resumes it, so an
interrupted refinement is continued by running it again; `F` and `enclosure_method` are not
stored and must be the ones the store was built with.  `ulam_hierarchy` is the one-shot form.
"""
mutable struct UlamTree
    F
    domain::IntervalBox{2}
    d0::Int
    d1::Int
    bc::Symbol
    enclosure_method
    store::String
    grids::Vector{_Grid}        # grids[d + 1] has depth d, grown as the tree descends
    level::Int                  # depth of the coarsest uncertain cell (maxlevel if none)
    maxlevel::Int               # depth of the finest cell
    nactive::Int                # uncertain cells
    nresolved::Int              # resolved cells
end

_meta(domain::IntervalBox{2}, d0, d1, bc) =
    (; x = (inf(domain[1]), sup(domain[1])), y = (inf(domain[2]), sup(domain[2])), d0, d1, bc)
_rootfile(tree::UlamTree, k::Int) = joinpath(tree.store, "root$k.jls")
_nroots(tree::UlamTree) = 4^tree.d0

function UlamTree(F, domain::IntervalBox{2}, d0::Int, d1::Int;
        bc::Symbol = :periodic, enclosure_method = first_order_enclosure,
        store::AbstractString = mktempdir())
    0 <= d0 <= d1 || throw(ArgumentError("need 0 ≤ d0 ≤ d1, got $d0, $d1"))
    grids = [_Grid(domain, d, bc) for d in 0:d0]
    tree = UlamTree(F, domain, d0, d1, bc, enclosure_method, String(store), grids, d0, d0, 0, 0)
    meta = _meta(domain, d0, d1, bc)
    metafile = joinpath(tree.store, "tree.jls")
    if isfile(metafile)
        Serialization.deserialize(metafile) == meta ||
            throw(ArgumentError("$(tree.store) holds a tree with another domain, depths or boundary condition"))
    else
        mkpath(tree.store)
        g0 = grids[d0 + 1]
        Threads.@threads :dynamic for k in 1:_nroots(tree)
            X, Y = _image(F, _box(g0, _xy(g0, k)...), enclosure_method)
            cands = _targets!(Int[], g0, X, Y)
            isempty(cands) && error("cell $k of depth $d0 has no candidate target: the map leaves the domain")
            resolved = [Dict{Int,Float64}() for _ in d0:d1]
            if d0 == d1 && length(cands) == 1
                _resolve!(tree, resolved, d0, k, cands[1])
                _write(tree, _Subtree(k, Int[], Int[], [1], Int[], resolved, 1, d0))
            else
                _write(tree, _Subtree(k, [d0], [k], [1, length(cands) + 1], cands, resolved, 0, d0))
            end
        end
        Serialization.serialize(metafile, meta)    # written last: a half-built store is rebuilt
    end
    _index!(tree)
    return tree
end

function _grid(tree::UlamTree, d::Int)
    while length(tree.grids) <= d
        push!(tree.grids, _Grid(tree.domain, length(tree.grids), tree.bc))
    end
    return tree.grids[d + 1]
end

# A cell `id` at depth ℓ ≥ d1 resolved to the single target `c` at depth d1: its mass goes to
# every depth d in d0:d1, to the entry (ancestor of c, ancestor of id).
function _resolve!(tree::UlamTree, resolved, ℓ::Int, id::Int, c::Int)
    gl, gt = tree.grids[ℓ + 1], tree.grids[tree.d1 + 1]
    for (q, d) in enumerate(tree.d0:tree.d1)
        n = tree.grids[d + 1].ax.n
        key = (_ancestor(gl, id, d) - 1) * n * n + _ancestor(gt, c, d)
        acc = resolved[q]
        acc[key] = get(acc, key, 0.0) + exp2(-2 * (ℓ - d))
    end
    return resolved
end

# The subtree file holds a small header, read alone by `_header`, then the subtree.  Written to
# a temporary name and moved into place, so an interrupted write leaves the previous state.
function _write(tree::UlamTree, sub::_Subtree)
    hdr = (; root = sub.root, nactive = _nactive(sub), nresolved = sub.nresolved,
             levelmin = isempty(sub.levels) ? typemax(Int) : minimum(sub.levels),
             levelmax = sub.depth)
    tmp = _rootfile(tree, sub.root) * ".tmp"
    open(tmp, "w") do io
        Serialization.serialize(io, hdr)
        Serialization.serialize(io, sub)
    end
    mv(tmp, _rootfile(tree, sub.root); force = true)
    return hdr
end
_header(tree::UlamTree, k::Int) = open(Serialization.deserialize, _rootfile(tree, k))
function _read(tree::UlamTree, k::Int)
    open(_rootfile(tree, k)) do io
        Serialization.deserialize(io)
        Serialization.deserialize(io)::_Subtree
    end
end

# Set level, maxlevel and the counts from the headers; returns the active count per root.
function _index!(tree::UlamTree)
    hdrs = [_header(tree, k) for k in 1:_nroots(tree)]
    tree.nactive = sum(h.nactive for h in hdrs)
    tree.nresolved = sum(h.nresolved for h in hdrs)
    tree.maxlevel = maximum(h.levelmax for h in hdrs)
    lmin = minimum(h.levelmin for h in hdrs)
    tree.level = lmin == typemax(Int) ? tree.maxlevel : lmin
    return [h.nactive for h in hdrs]
end

"""
    refine!(tree::UlamTree; targets=nothing) -> tree

Quadrisect the uncertain cells, every subtree in turn: the candidates of a child are the
cells met by the enclosure of its image among the children of its parent's candidates (above
`d1`), or among its parent's candidates (at or below `d1`); a child with a single candidate
at depth `d1` or finer is resolved.  With `targets`, a vector of booleans indexed as
`build_uniform_partition(domain, d1)`, only the cells at depth `d1` or finer having a
candidate among the marked targets are quadrisected, which refines the rows of `U` that are
too heavy; cells coarser than `d1` are always quadrisected.
"""
function refine!(tree::UlamTree; targets::Union{Nothing,AbstractVector{Bool}} = nothing)
    gt = _grid(tree, tree.d1)
    marked = if targets === nothing
        nothing
    else
        length(targets) == gt.ax.n * gt.ay.n ||
            throw(ArgumentError("targets must have one entry per cell of depth $(tree.d1)"))
        perm = _box_permutation(build_uniform_partition(tree.domain, tree.d1), gt)
        [targets[perm[id]] for id in 1:length(perm)]   # by grid id
    end
    order = sortperm(_index!(tree); rev = true)         # largest subtrees first
    _grid(tree, tree.maxlevel + 1)                       # grown here, read in the threads
    Threads.@threads :dynamic for k in order
        _header(tree, k).nactive == 0 && continue
        sub = _refine(tree, _read(tree, k), marked)
        sub === nothing || _write(tree, sub)
    end
    _index!(tree)
    return tree
end

# One quadrisection of the selected cells of a subtree; `nothing` if none was selected.
function _refine(tree::UlamTree, sub::_Subtree, marked)
    d0, d1, F, encl = tree.d0, tree.d1, tree.F, tree.enclosure_method
    levels = Int[]; ids = Int[]; candptr = [1]; cands = Int[]
    resolved = sub.resolved
    nres = sub.nresolved
    buf = Int[]
    refined = false
    depth = sub.depth
    for k in 1:_nactive(sub)
        ℓ, pid = sub.levels[k], sub.ids[k]
        pc = view(sub.cands, sub.candptr[k]:sub.candptr[k + 1] - 1)
        if ℓ >= d1 && marked !== nothing && !any(c -> marked[c], pc)
            push!(levels, ℓ); push!(ids, pid); append!(cands, pc); push!(candptr, length(cands) + 1)
            continue
        end
        refined = true
        depth = max(depth, ℓ + 1)
        gp, gs = tree.grids[ℓ + 1], tree.grids[ℓ + 2]       # parent's and children's grids
        gc = tree.grids[min(ℓ + 1, d1) + 1]                 # children's target grid
        ix, iy = _xy(gp, pid)
        for (cx, cy) in ((2ix - 1, 2iy - 1), (2ix, 2iy - 1), (2ix - 1, 2iy), (2ix, 2iy))
            X, Y = _image(F, _box(gs, cx, cy), encl)
            _targets!(buf, gc, X, Y)
            n0 = length(cands)
            for i in buf
                (ℓ + 1 <= d1 ? _ancestor(gc, i, ℓ) in pc : i in pc) && push!(cands, i)
            end
            m = length(cands) - n0
            m == 0 && error("a cell at depth $(ℓ + 1) has no candidate target: the map leaves the domain")
            cid = _id(gs, cx, cy)
            if ℓ + 1 >= d1 && m == 1
                _resolve!(tree, resolved, ℓ + 1, cid, cands[end])
                resize!(cands, n0)
                nres += 1
            else
                push!(levels, ℓ + 1); push!(ids, cid); push!(candptr, length(cands) + 1)
            end
        end
    end
    refined || return nothing
    return _Subtree(sub.root, levels, ids, candptr, cands, resolved, nres, depth)
end

"""
    ulam_matrices(tree::UlamTree, d) -> (depth, partition, graph, L_lo, U)
    ulam_matrices(tree::UlamTree, depths::AbstractRange) -> Vector of the above

The graph and the Ulam matrix enclosure at depth `d`, `d0 ≤ d ≤ min(tree.level, d1)`, read
off the subtrees: every cell, resolved or uncertain, contributes its mass `4^{-(ℓ-d)}`
relative to its ancestor at depth `d` to `L_lo` when the ancestors at depth `d` of its
candidates are a single cell and to `U` for each of them otherwise.  Matrices are indexed as
`partition = build_uniform_partition(domain, d)`.  Several depths are read in one pass over
the subtrees.
"""
ulam_matrices(tree::UlamTree, d::Int) = ulam_matrices(tree, d:d)[1]

function ulam_matrices(tree::UlamTree, depths::AbstractRange{Int})
    dmax = min(tree.level, tree.d1)
    all(d -> tree.d0 <= d <= dmax, depths) ||
        throw(ArgumentError("depths $depths are not in $(tree.d0):$dmax"))
    _grid(tree, tree.maxlevel)
    nroots = _nroots(tree)
    parts = Vector{Tuple{Vector{Dict{Int,Float64}},Vector{Dict{Int,Float64}}}}(undef, nroots)
    Threads.@threads :dynamic for k in 1:nroots
        sub = _read(tree, k)
        lo = [copy(sub.resolved[d - tree.d0 + 1]) for d in depths]
        up = [Dict{Int,Float64}() for _ in depths]
        anc = Int[]
        for c in 1:_nactive(sub)
            ℓ, id = sub.levels[c], sub.ids[c]
            gl, gt = tree.grids[ℓ + 1], tree.grids[min(ℓ, tree.d1) + 1]
            cs = view(sub.cands, sub.candptr[c]:sub.candptr[c + 1] - 1)
            for (q, d) in enumerate(depths)
                n = tree.grids[d + 1].ax.n
                empty!(anc)
                for i in cs
                    a = _ancestor(gt, i, d)
                    a in anc || push!(anc, a)
                end
                base = (_ancestor(gl, id, d) - 1) * n * n
                mass = exp2(-2 * (ℓ - d))
                acc = length(anc) == 1 ? lo[q] : up[q]
                for a in anc
                    acc[base + a] = get(acc, base + a, 0.0) + mass
                end
            end
        end
        parts[k] = (lo, up)
    end
    return map(enumerate(depths)) do (q, d)
        gd = _grid(tree, d)
        n = gd.ax.n * gd.ay.n
        lo = Dict{Int,Float64}(); up = Dict{Int,Float64}()
        for (los, ups) in parts
            for (key, v) in los[q]
                lo[key] = get(lo, key, 0.0) + v
            end
            for (key, v) in ups[q]
                up[key] = get(up, key, 0.0) + v
            end
        end
        partition = build_uniform_partition(tree.domain, d)
        perm = _box_permutation(partition, gd)         # perm[id] = box index in the partition
        ij(key) = (perm[(key - 1) % n + 1], perm[(key - 1) ÷ n + 1])
        I = Int[]; J = Int[]; V = Float64[]
        for (key, v) in lo
            i, j = ij(key); push!(I, i); push!(J, j); push!(V, v)
        end
        UI = Int[]; UJ = Int[]; UV = Float64[]
        for (key, v) in up
            i, j = ij(key); push!(UI, i); push!(UJ, j); push!(UV, v)
        end
        L_lo = sparse(I, J, V, n, n)
        U = sparse(UI, UJ, UV, n, n)
        graph = sparse(vcat(I, UI), vcat(J, UJ), true, n, n)
        (; depth = d, partition, graph, L_lo, U)
    end
end

"""
    ulam_hierarchy(F, domain, d0, d1; leaf_depth=8, bc=:periodic,
                   enclosure_method=first_order_enclosure, store=mktempdir())

Transition graphs and Ulam matrix enclosures at the depths `d0:d1` (see the file docstring):
a `UlamTree` refined to depth `d1 + leaf_depth`, read at each depth with `ulam_matrices`.
Returns a vector with one named tuple `(depth, partition, graph, L_lo, U)` per depth.  The
uncertain mass of a cell at depth `d` is at most the mass of a strip of width
`diam(domain) 2^{-(d1 + leaf_depth)}` around the preimages of the grid lines of depth `d1`.
"""
function ulam_hierarchy(F, domain::IntervalBox{2}, d0::Int, d1::Int; leaf_depth::Int = 8,
        bc::Symbol = :periodic, enclosure_method = first_order_enclosure,
        store::AbstractString = mktempdir())
    leaf_depth >= 0 || throw(ArgumentError("leaf_depth must be nonnegative"))
    tree = UlamTree(F, domain, d0, d1; bc, enclosure_method, store)
    while tree.level < d1 + leaf_depth && tree.nactive > 0
        refine!(tree)
    end
    return ulam_matrices(tree, d0:d1)
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
