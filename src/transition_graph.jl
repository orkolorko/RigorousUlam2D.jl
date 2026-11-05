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
refinement needs memory for the subtrees being refined only and resumes where it stopped;
`ulam_matrices` can also continue the refinement in memory, cell by cell and depth first,
and read the matrices at a finer level than the one stored without writing it.

Every leaf of this tree, resolved or uncertain, has a depth `ℓ ≥ d1`, a position, and its
candidate targets at depth `d1`.  The Ulam matrix `L^{(d)}` at a depth `d0 ≤ d ≤ d1`,
`L^{(d)}[i, j] = |I_j ∩ T⁻¹(I_i)| / |I_j|`, is then enclosed by

    L_lo^{(d)} ≤ L^{(d)} ≤ L_lo^{(d)} + U^{(d)}   entrywise,

where a leaf in `I_j`, of mass `4^{-(ℓ - d)}` relative to `I_j`, contributes to
`L_lo^{(d)}[i, j]` when the ancestors at depth `d` of its candidates are the single cell `I_i`,
and to `U^{(d)}[i, j]` for each ancestor `I_i` otherwise.  All entries are then exact dyadic
numbers, no rounding enters, and the column sums of `U^{(d)}` are at least the uncertain mass
of each source cell.

With an `area` function (see `fibred_area`), an uncertain leaf is read more closely: the
fraction of the leaf mapped into each candidate is enclosed, the fractions going to one
ancestor are summed, and the leaf contributes the lower end of each sum to `L_lo` and the
difference of the two ends to `U`.  The accumulation is then rounded, downward for `L_lo`
and upward for the upper bound, so that `L_lo ≤ L ≤ L_lo + U` holds in exact arithmetic,
and the result depends on the order of the sums, that is on the number of threads.  The
graph holds the entries whose upper bound is positive.

Boundary conditions are `:periodic` or `:reflecting`.  Nothing is wrapped: an image interval
is translated by an integer multiple of the period (`L`, or `2L` when reflecting, in which
case the part beyond the domain is reflected back) in interval arithmetic and compared with
the grid lines, which are exact floating-point numbers (checked once in `_Axis`).  The image
of a box is the intersection of the natural interval extension of `F` with the enclosure of
`enclosure_method`; both enclose the true image, so does their intersection.
"""

export transition_graph, ulam_hierarchy, UlamTree, refine!, repair!, ulam_matrices, fibred_area

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

# X translated by k times `period` so that inf lands in [x0, x0 + period), and k
function _shiftk(X::Interval, x0, period)
    k = floor(Int, (inf(X) - x0) / period)
    Xs = X - interval(k * period)
    while inf(Xs) < x0
        k -= 1; Xs = X - interval(k * period)
    end
    while inf(Xs) >= x0 + period
        k += 1; Xs = X - interval(k * period)
    end
    return Xs, k
end
_shift(X::Interval, x0, period) = _shiftk(X, x0, period)[1]

# The windows of the cell `i` of the axis for an image interval `X` narrower than the domain,
# `X` not translated: the translates of the cell by the period (periodic), or by twice the
# domain together with their reflections about its far end (reflecting), that meet `X`; each
# as the pair of intervals enclosing its two endpoints.  The preimage of the cell under the
# boundary condition is the union of the preimages of the windows, and a window the true
# image misses only adds an area containing 0.
const _Window = Tuple{Interval{Float64},Interval{Float64}}
function _windows!(ws::Vector{_Window}, ax::_Axis, X::Interval, i::Int)
    empty!(ws)
    (; x0, L) = ax
    c = _cell(ax, i)
    clo, chi = interval(inf(c)), interval(sup(c))
    if ax.bc === :periodic
        _, k = _shiftk(X, x0, L)
        for m in k:k + 1
            t = interval(m) * interval(L)
            _meets(clo + t, chi + t, X) && push!(ws, (clo + t, chi + t))
        end
    else
        _, k = _shiftk(X, x0, 2L)
        far = interval(2) * interval(x0 + L)            # x0 + L is exact (checked in _Axis)
        for m in k:k + 1
            t = interval(2m) * interval(L)
            _meets(clo + t, chi + t, X) && push!(ws, (clo + t, chi + t))
            α, β = far + t - chi, far + t - clo         # the reflection about x0 + L
            _meets(α, β, X) && push!(ws, (α, β))
        end
    end
    return ws
end
_meets(α, β, X) = sup(β) >= inf(X) && inf(α) <= sup(X)

# a + b rounded down and up, from the exact error of the floating-point sum (Knuth's TwoSum)
function _down(a::Float64, b::Float64)
    s = a + b; bb = s - a
    e = (a - (s - bb)) + (b - bb)
    return e < 0 ? prevfloat(s) : s
end
function _up(a::Float64, b::Float64)
    s = a + b; bb = s - a
    e = (a - (s - bb)) + (b - bb)
    return e > 0 ? nextfloat(s) : s
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
             area=nothing, store=mktempdir())

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

`area(box, W)`, if given, encloses the fraction of the box `box` that `F` maps into the
window `W`, both `IntervalBox{2}`, with no boundary condition applied; `ulam_matrices` then
reads the uncertain leaves through it (see the file docstring and `fibred_area`).
"""
mutable struct UlamTree
    F
    domain::IntervalBox{2}
    d0::Int
    d1::Int
    bc::Symbol
    enclosure_method
    area
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
        bc::Symbol = :periodic, enclosure_method = first_order_enclosure, area = nothing,
        store::AbstractString = mktempdir())
    0 <= d0 <= d1 || throw(ArgumentError("need 0 ≤ d0 ≤ d1, got $d0, $d1"))
    grids = [_Grid(domain, d, bc) for d in 0:d0]
    tree = UlamTree(F, domain, d0, d1, bc, enclosure_method, area, String(store), grids, d0, d0, 0, 0)
    meta = _meta(domain, d0, d1, bc)
    metafile = joinpath(tree.store, "tree.jls")
    if isfile(metafile)
        Serialization.deserialize(metafile) == meta ||
            throw(ArgumentError("$(tree.store) holds a tree with another domain, depths or boundary condition"))
    else
        mkpath(tree.store)
        g0 = grids[d0 + 1]
        Threads.@threads :dynamic for k in 1:_nroots(tree)
            _write(tree, _root(tree, k, F, enclosure_method))
        end
        Serialization.serialize(metafile, meta)    # written last: a half-built store is rebuilt
    end
    _index!(tree)
    return tree
end

# The subtree of the cell `k` of depth d0 before any refinement.
function _root(tree::UlamTree, k::Int, F, encl)
    d0, d1 = tree.d0, tree.d1
    g0 = tree.grids[d0 + 1]
    X, Y = _image(F, _box(g0, _xy(g0, k)...), encl)
    cands = _targets!(Int[], g0, X, Y)
    isempty(cands) && error("cell $k of depth $d0 has no candidate target: the map leaves the domain")
    resolved = [Dict{Int,Float64}() for _ in d0:d1]
    if d0 == d1 && length(cands) == 1
        _resolve!(tree, resolved, d0, k, cands[1])
        return _Subtree(k, Int[], Int[], [1], Int[], resolved, 1, d0)
    end
    return _Subtree(k, [d0], [k], [1, length(cands) + 1], cands, resolved, 0, d0)
end

# Whether a subtree read from disk is well formed: the frontier arrays agree in length, the
# candidate pointers are a partition of the candidates, and every depth, cell and candidate
# is in range.
function _consistent(tree::UlamTree, sub::_Subtree)
    n = length(sub.ids)
    length(sub.levels) == n && length(sub.candptr) == n + 1 || return false
    sub.candptr[1] == 1 && sub.candptr[end] == length(sub.cands) + 1 && issorted(sub.candptr) || return false
    1 <= sub.root <= _nroots(tree) && sub.depth >= tree.d0 && sub.nresolved >= 0 || return false
    length(sub.resolved) == tree.d1 - tree.d0 + 1 || return false
    nt = 4^tree.d1
    for k in 1:n
        ℓ = sub.levels[k]
        tree.d0 <= ℓ <= sub.depth && 1 <= sub.ids[k] <= 4^ℓ || return false
        sub.candptr[k + 1] > sub.candptr[k] || return false
        nc = 4^min(ℓ, tree.d1)
        for c in sub.candptr[k]:sub.candptr[k + 1] - 1
            1 <= sub.cands[c] <= min(nc, nt) || return false
        end
    end
    return true
end

"""
    repair!(tree::UlamTree; level=tree.level) -> Vector{Int}

Read every subtree file and rebuild those that cannot be read or are not well formed, from
their cell of depth `d0` refined uniformly to the depth `level`; returns the roots rebuilt.
A subtree is written whole to a temporary file and renamed, but a file system may still
lose the data of a renamed file when the disk fills during its writeback, which is how a
truncated subtree arises.
"""
function repair!(tree::UlamTree; level::Int = tree.level)
    F, encl = tree.F, tree.enclosure_method
    _grid(tree, max(level, tree.maxlevel) + 1)
    bad = Int[]
    lk = ReentrantLock()
    Threads.@threads :dynamic for k in 1:_nroots(tree)
        ok = try
            _consistent(tree, _read(tree, k))
        catch
            false
        end
        ok && continue
        @lock lk push!(bad, k)
        sub = _root(tree, k, F, encl)
        while !isempty(sub.levels) && minimum(sub.levels) < level
            sub = _refine(tree, sub, nothing, F, encl)
        end
        _write(tree, sub)
    end
    _index!(tree)
    return sort!(bad)
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

# The accumulators of a read, one dictionary per depth keyed as in `_resolve!`: `lo`, the
# mass placed exactly (dyadic, summed exactly); `ulo` and `uhi`, the lower and upper bounds
# of the uncertain mass, summed rounding down and up.  An entry is enclosed by
# `lo + ulo ≤ L ≤ lo + uhi`.
_Acc = @NamedTuple{lo::Vector{Dict{Int,Float64}}, ulo::Vector{Dict{Int,Float64}}, uhi::Vector{Dict{Int,Float64}}}
_acc(depths, lo = [Dict{Int,Float64}() for _ in depths]) =
    (; lo, ulo = [Dict{Int,Float64}() for _ in depths], uhi = [Dict{Int,Float64}() for _ in depths])
function _add!(dst::_Acc, src::_Acc)
    for q in eachindex(dst.lo)
        _add!(dst.lo[q], src.lo[q], _down); _add!(dst.ulo[q], src.ulo[q], _down); _add!(dst.uhi[q], src.uhi[q], _up)
    end
    return dst
end

# A cell `id` at depth ℓ with candidates `cs` (at depth min(ℓ, d1)) read into the accumulators
# `acc` at the depths `depths[q] ≤ ℓ`.  Without `area`, or when the image is as wide as the
# domain, the whole mass goes to `lo` when the ancestors at that depth of its candidates are
# one cell and to `uhi` for each of them otherwise; with `area`, the fraction mapped into each
# ancestor is enclosed (`_areas!`), sharpened by the fractions summing to 1, and its ends go
# to `ulo` and `uhi`.
function _read_cell!(acc::_Acc, tree::UlamTree, F, encl, area, depths, ℓ::Int, id::Int, cs, s)
    gl, gt = tree.grids[ℓ + 1], tree.grids[min(ℓ, tree.d1) + 1]
    anc = s.anc
    exact = area === nothing || length(cs) == 1 || !_areas!(s, tree, F, encl, area, gl, gt, id, cs)
    for (q, d) in enumerate(depths)
        n = tree.grids[d + 1].ax.n
        empty!(anc); empty!(s.Aa)
        for (t, c) in enumerate(cs)
            a = _ancestor(gt, c, d)
            p = findfirst(==(a), anc)
            if p === nothing
                push!(anc, a); exact || push!(s.Aa, s.A[t])
            elseif !exact
                s.Aa[p] += s.A[t]
            end
        end
        base = (_ancestor(gl, id, d) - 1) * n * n
        mass = exp2(-2 * (ℓ - d))
        lo, ulo, uhi = acc.lo[q], acc.ulo[q], acc.uhi[q]
        if length(anc) == 1
            key = base + anc[1]
            lo[key] = get(lo, key, 0.0) + mass
        elseif exact
            for a in anc
                uhi[base + a] = get(uhi, base + a, 0.0) + mass
            end
        else
            for (p, a) in enumerate(anc)
                others = interval(0.0)
                for p′ in eachindex(anc)
                    p′ == p || (others += s.Aa[p′])
                end
                Ap = intersect_interval(s.Aa[p], intersect_interval(interval(1.0) - others, interval(0.0, 1.0)))
                isempty_interval(Ap) &&
                    error("the fractions of a cell at depth $ℓ mapped into its targets do not sum to 1: the area function is wrong")
                P = interval(mass) * Ap
                key = base + a
                inf(P) > 0 && (ulo[key] = _down(get(ulo, key, 0.0), inf(P)))
                uhi[key] = _up(get(uhi, key, 0.0), sup(P))
            end
        end
    end
    return nothing
end

# The fractions `s.A[t]` of the cell `id` of the grid `gl` mapped into its candidates `cs` of
# the grid `gt`, through `area` and the windows of the boundary condition; `false`, with
# nothing computed, when the image is as wide as the domain along an axis.
function _areas!(s, tree::UlamTree, F, encl, area, gl::_Grid, gt::_Grid, id::Int, cs)
    box = _box(gl, _xy(gl, id)...)
    X, Y = _image(F, box, encl)
    (diam(X) < gt.ax.L && diam(Y) < gt.ay.L) || return false
    resize!(s.A, length(cs))
    for (t, c) in enumerate(cs)
        ix, iy = _xy(gt, c)
        _windows!(s.wx, gt.ax, X, ix)
        _windows!(s.wy, gt.ay, Y, iy)
        s.A[t] = _area(area, box, s.wx, s.wy, s)
    end
    return true
end

# The fraction of `box` mapped into the union of the windows `wxs × wys`: the sum over the
# pairs, each read through the window inside both endpoint enclosures and the one around
# them, since the fraction grows with the window.
function _area(area, box::IntervalBox{2}, wxs::Vector{_Window}, wys::Vector{_Window}, s)
    A = interval(0.0)
    for (αx, βx) in wxs, (αy, βy) in wys
        outer = IntervalBox(interval(inf(αx), sup(βx)), interval(inf(αy), sup(βy)))
        if isthin(αx) && isthin(βx) && isthin(αy) && isthin(βy)
            A += _call(area, box, outer, s)
        else
            hi = sup(_call(area, box, outer, s))
            lo = sup(αx) <= inf(βx) && sup(αy) <= inf(βy) ?
                inf(_call(area, box, IntervalBox(interval(sup(αx), inf(βx)), interval(sup(αy), inf(βy))), s)) : 0.0
            A += interval(min(lo, hi), hi)
        end
    end
    return A
end

# The four children of the cell `pid` at depth ℓ whose candidates are `pc`: their ids go to
# `cids` and their candidates to `ccands`, those of the q-th child at `cptr[q]:cptr[q+1]-1`.
# `F` and `encl` are passed rather than read off the tree, whose fields are untyped, so that
# the image evaluations are compiled for their types.
function _children!(cids::Vector{Int}, cptr::Vector{Int}, ccands::Vector{Int}, tree::UlamTree,
        F, encl, ℓ::Int, pid::Int, pc, buf::Vector{Int})
    d1 = tree.d1
    gp, gs = tree.grids[ℓ + 1], tree.grids[ℓ + 2]       # parent's and children's grids
    gc = tree.grids[min(ℓ + 1, d1) + 1]                 # children's target grid
    ix, iy = _xy(gp, pid)
    resize!(cids, 4); resize!(cptr, 5)
    empty!(ccands)
    cptr[1] = 1
    for (q, (cx, cy)) in enumerate(((2ix - 1, 2iy - 1), (2ix, 2iy - 1), (2ix - 1, 2iy), (2ix, 2iy)))
        X, Y = _image(F, _box(gs, cx, cy), encl)
        _targets!(buf, gc, X, Y)
        for i in buf
            (ℓ + 1 <= d1 ? _ancestor(gc, i, ℓ) in pc : i in pc) && push!(ccands, i)
        end
        length(ccands) + 1 == cptr[q] &&
            error("a cell at depth $(ℓ + 1) has no candidate target: the map leaves the domain")
        cids[q] = _id(gs, cx, cy)
        cptr[q + 1] = length(ccands) + 1
    end
    return nothing
end

# Whether the cell at depth ℓ with candidates `pc` is quadrisected: always above d1, and at d1
# or finer only when `marked` is `nothing`, or marks one of its candidates (booleans), or asks
# one of them for a level finer than ℓ (integers).
_selected(ℓ::Int, pc, d1::Int, ::Nothing) = true
_selected(ℓ::Int, pc, d1::Int, marked::Vector{Bool}) = ℓ < d1 || any(c -> marked[c], pc)
_selected(ℓ::Int, pc, d1::Int, marked::Vector{Int}) = ℓ < d1 || any(c -> marked[c] > ℓ, pc)

# `targets`, a vector indexed as `build_uniform_partition(domain, d1)`, reindexed by the grid
# ids of depth d1; `nothing` stays `nothing`.
function _marked(tree::UlamTree, targets)
    targets === nothing && return nothing
    gt = _grid(tree, tree.d1)
    length(targets) == gt.ax.n * gt.ay.n ||
        throw(ArgumentError("targets must have one entry per cell of depth $(tree.d1)"))
    perm = _box_permutation(build_uniform_partition(tree.domain, tree.d1), gt)
    return [targets[perm[id]] for id in 1:length(perm)]   # by grid id
end

# dst += src entrywise, rounded by `round` (`_down` or `_up`)
function _add!(dst::Dict{Int,Float64}, src::Dict{Int,Float64}, round)
    for (key, v) in src
        dst[key] = round(get(dst, key, 0.0), v)
    end
    return dst
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
    marked = _marked(tree, targets)
    order = sortperm(_index!(tree); rev = true)         # largest subtrees first
    _grid(tree, tree.maxlevel + 1)                       # grown here, read in the threads
    F, encl = tree.F, tree.enclosure_method
    Threads.@threads :dynamic for k in order
        _header(tree, k).nactive == 0 && continue
        sub = _refine(tree, _read(tree, k), marked, F, encl)
        sub === nothing || _write(tree, sub)
    end
    _index!(tree)
    return tree
end

# One quadrisection of the selected cells of a subtree; `nothing` if none was selected.
function _refine(tree::UlamTree, sub::_Subtree, marked, F, encl)
    d1 = tree.d1
    levels = Int[]; ids = Int[]; candptr = [1]; cands = Int[]
    resolved = sub.resolved
    nres = sub.nresolved
    buf = Int[]; cids = Int[]; cptr = Int[]; ccands = Int[]
    refined = false
    depth = sub.depth
    for k in 1:_nactive(sub)
        ℓ, pid = sub.levels[k], sub.ids[k]
        pc = view(sub.cands, sub.candptr[k]:sub.candptr[k + 1] - 1)
        if !_selected(ℓ, pc, d1, marked)
            push!(levels, ℓ); push!(ids, pid); append!(cands, pc); push!(candptr, length(cands) + 1)
            continue
        end
        refined = true
        depth = max(depth, ℓ + 1)
        _children!(cids, cptr, ccands, tree, F, encl, ℓ, pid, pc, buf)
        for q in 1:4
            cc = view(ccands, cptr[q]:cptr[q + 1] - 1)
            if ℓ + 1 >= d1 && length(cc) == 1
                _resolve!(tree, resolved, ℓ + 1, cids[q], cc[1])
                nres += 1
            else
                push!(levels, ℓ + 1); push!(ids, cids[q]); append!(cands, cc); push!(candptr, length(cands) + 1)
            end
        end
    end
    refined || return nothing
    return _Subtree(sub.root, levels, ids, candptr, cands, resolved, nres, depth)
end

# The cells `cells` of the frontier of `sub` read into `acc`.  One call per chunk: `F`, `encl`
# and `area` come from the untyped fields of the tree, and a call through them is dispatched
# at run time and boxes its arguments, so it is made once here and not once per cell.
function _read_chunk!(acc::_Acc, tree::UlamTree, F, encl, area, depths, sub::_Subtree, cells, lmax::Int, marked)
    s = _scratch()
    for cell in cells
        pc = view(sub.cands, sub.candptr[cell]:sub.candptr[cell + 1] - 1)
        _descend!(acc, tree, F, encl, area, depths, sub.levels[cell], sub.ids[cell], pc, lmax, marked, s)
    end
    return acc
end

# Scratch vectors of one in-memory descent: the stack of pending cells (level, id, and the
# range `p:q` of their candidates in `scands`), the buffers of `_children!` and `_areas!`, and
# the stack of `_fibred_area`.
_scratch() = (; slevels = Int[], sids = Int[], sp = Int[], sq = Int[], scands = Int[],
                buf = Int[], cids = Int[], cptr = Int[], ccands = Int[], anc = Int[],
                A = Interval{Float64}[], Aa = Interval{Float64}[], wx = _Window[], wy = _Window[],
                stack = _Piece[])

# Descend the cell `id` at depth ℓ with candidates `pc`, depth first in memory and writing
# nothing, down to the depth `level`, and read the leaves into `acc` with `_read_cell!`:
# a cell at depth `level` or finer, or one at depth d1 or finer that has one candidate or is
# not selected by `marked` (the levels asked per target, or `nothing`), is a leaf.
# Candidates are stacked LIFO: whatever lies above the range of a popped cell belongs to
# cells already finished.
function _descend!(acc::_Acc, tree::UlamTree, F, encl, area, depths, ℓ::Int, id::Int, pc, level::Int,
        marked::Union{Nothing,Vector{Int}}, s)
    d1 = tree.d1
    empty!(s.slevels); empty!(s.sids); empty!(s.sp); empty!(s.sq); empty!(s.scands)
    append!(s.scands, pc)
    push!(s.slevels, ℓ); push!(s.sids, id); push!(s.sp, 1); push!(s.sq, length(s.scands))
    while !isempty(s.slevels)
        ℓ = pop!(s.slevels); id = pop!(s.sids); p = pop!(s.sp); q = pop!(s.sq)
        resize!(s.scands, q)
        cs = view(s.scands, p:q)
        if ℓ >= level || (ℓ >= d1 && (length(cs) == 1 || !_selected(ℓ, cs, d1, marked)))
            _read_cell!(acc, tree, F, encl, area, depths, ℓ, id, cs, s)
            continue
        end
        _children!(s.cids, s.cptr, s.ccands, tree, F, encl, ℓ, id, cs, s.buf)
        for c in 1:4
            cp = length(s.scands) + 1
            append!(s.scands, view(s.ccands, s.cptr[c]:s.cptr[c + 1] - 1))
            push!(s.slevels, ℓ + 1); push!(s.sids, s.cids[c]); push!(s.sp, cp); push!(s.sq, length(s.scands))
        end
    end
    return nothing
end

"""
    ulam_matrices(tree::UlamTree, d; level=0) -> (depth, partition, graph, L_lo, U)
    ulam_matrices(tree::UlamTree, depths::AbstractRange; level=0) -> Vector of the above

The graph and the Ulam matrix enclosure at depth `d`, `d0 ≤ d ≤ min(tree.level, d1)`, read
off the subtrees: every cell, resolved or uncertain, contributes its mass `4^{-(ℓ-d)}`
relative to its ancestor at depth `d` to `L_lo` when the ancestors at depth `d` of its
candidates are a single cell and to `U` for each of them otherwise.  Matrices are indexed as
`partition = build_uniform_partition(domain, d)`.  Several depths are read in one pass over
the subtrees.

With `level` finer than `tree.level`, the uncertain cells are quadrisected in memory, depth
first and cell by cell, down to the depth `level` before being read, and nothing is written
to the store: the matrices are those `refine!` would give after the same quadrisections, at
the memory cost of one descent, and `d` may then reach `min(level, d1)`.  `level` may also be
a vector of integers indexed as `build_uniform_partition(domain, d1)`, the depth asked for
the cells feeding each row: a cell at depth `d1` or finer is quadrisected while one of its
candidates asks for a finer depth, so a row that is too heavy is refined further than the
others, and repeated calls that raise the levels of the heavy rows only keep what the
previous calls gave the rest.
"""
ulam_matrices(tree::UlamTree, d::Int; kwargs...) = ulam_matrices(tree, d:d; kwargs...)[1]

function ulam_matrices(tree::UlamTree, depths::AbstractRange{Int};
        level::Union{Int,AbstractVector{Int}} = 0)
    marked = level isa Int ? nothing : _marked(tree, Vector{Int}(level))
    lmax = level isa Int ? level : maximum(level; init = 0)
    dmax = min(max(tree.level, lmax), tree.d1)
    all(d -> tree.d0 <= d <= dmax, depths) ||
        throw(ArgumentError("depths $depths are not in $(tree.d0):$dmax"))
    _grid(tree, max(tree.maxlevel, lmax))               # grown here, read in the threads
    F, encl, area = tree.F, tree.enclosure_method, tree.area
    order = sortperm(_index!(tree); rev = true)         # largest subtrees first
    parts = Vector{_Acc}(undef, _nroots(tree))
    Threads.@threads :dynamic for k in order
        sub = _read(tree, k)
        nact = _nactive(sub)
        # the frontier in chunks, so that a large subtree is descended by several threads
        nchunks = clamp(cld(nact, 1 << 16), 1, 4 * Threads.nthreads())
        bounds = round.(Int, range(0, nact; length = nchunks + 1))
        cparts = Vector{_Acc}(undef, nchunks)
        Threads.@threads :dynamic for c in 1:nchunks
            # named apart from the accumulator of the root below: a name assigned in both
            # scopes would be one variable, shared by the chunks
            cacc = _acc(depths)
            _read_chunk!(cacc, tree, F, encl, area, depths, sub, bounds[c] + 1:bounds[c + 1], lmax, marked)
            cparts[c] = cacc
        end
        racc = _acc(depths, [copy(sub.resolved[d - tree.d0 + 1]) for d in depths])
        for cacc in cparts
            _add!(racc, cacc)
        end
        parts[k] = racc
    end
    return map(enumerate(depths)) do (q, d)
        gd = _grid(tree, d)
        n = gd.ax.n * gd.ay.n
        lo = Dict{Int,Float64}(); ulo = Dict{Int,Float64}(); uhi = Dict{Int,Float64}()
        for racc in parts
            _add!(lo, racc.lo[q], _down); _add!(ulo, racc.ulo[q], _down); _add!(uhi, racc.uhi[q], _up)
        end
        partition = build_uniform_partition(tree.domain, d)
        perm = _box_permutation(partition, gd)         # perm[id] = box index in the partition
        ij(key) = (perm[(key - 1) % n + 1], perm[(key - 1) ÷ n + 1])
        I = Int[]; J = Int[]; V = Float64[]
        for key in union(keys(lo), keys(ulo))           # L_lo = lo + ulo, downward
            i, j = ij(key); push!(I, i); push!(J, j); push!(V, _down(get(lo, key, 0.0), get(ulo, key, 0.0)))
        end
        UI = Int[]; UJ = Int[]; UV = Float64[]
        for (key, v) in uhi                             # U = (lo + uhi) - L_lo, upward
            v > 0 || continue                           # an upper bound of 0: the entry is 0
            l = _down(get(lo, key, 0.0), get(ulo, key, 0.0))
            i, j = ij(key); push!(UI, i); push!(UJ, j); push!(UV, _up(_up(get(lo, key, 0.0), v), -l))
        end
        L_lo = sparse(I, J, V, n, n)
        U = sparse(UI, UJ, UV, n, n)
        graph = sparse(vcat(I, UI), vcat(J, UJ), true, n, n)
        (; depth = d, partition, graph, L_lo, U)
    end
end

"""
    ulam_hierarchy(F, domain, d0, d1; leaf_depth=8, bc=:periodic,
                   enclosure_method=first_order_enclosure, area=nothing, store=mktempdir())

Transition graphs and Ulam matrix enclosures at the depths `d0:d1` (see the file docstring):
a `UlamTree` refined to depth `d1 + leaf_depth`, read at each depth with `ulam_matrices`.
Returns a vector with one named tuple `(depth, partition, graph, L_lo, U)` per depth.  The
uncertain mass of a cell at depth `d` is at most the mass of a strip of width
`diam(domain) 2^{-(d1 + leaf_depth)}` around the preimages of the grid lines of depth `d1`.
"""
function ulam_hierarchy(F, domain::IntervalBox{2}, d0::Int, d1::Int; leaf_depth::Int = 8,
        bc::Symbol = :periodic, enclosure_method = first_order_enclosure, area = nothing,
        store::AbstractString = mktempdir())
    leaf_depth >= 0 || throw(ArgumentError("leaf_depth must be nonnegative"))
    tree = UlamTree(F, domain, d0, d1; bc, enclosure_method, area, store)
    while tree.level < d1 + leaf_depth && tree.nactive > 0
        refine!(tree)
    end
    return ulam_matrices(tree, d0:d1)
end

"""
    fibred_area(xonly, xcomp, yof; yint=nothing, bisect=12) -> area

An `area` function for `UlamTree`, for a map `F(x, y)` whose component `xonly` (1 or 2)
depends on `x` alone, `xcomp(x)`, and whose other component is strictly monotone in `y` for
every fixed `x`, with `yof(x, w)` the `y` at which it equals `w` and `yint(x, δ, w)` the
integral of `yof(·, w)` from `x` to `x + δ`.  The three are interval extensions: `xcomp` and
`yof` take an interval `x`, `yint` thin `x` and `δ`, and `w` is always thin.  `yint` is to
be written as an increment, not as the difference of an antiderivative at the two ends:
the pieces are as small as `2^-bisect` of a leaf, and the difference of two values of the
antiderivative loses the integral of such a piece in their rounding (for a polynomial
antiderivative, expand in `δ`; for `1/x²`, write the difference over a common denominator).

The fraction of a box `[x0, x1] × [y0, y1]` that `F` maps into a window `W` is the integral
over `x` of the length of `[y0, y1] ∩ {y : F(x, y) ∈ W}`.  The part of `[x0, x1]` that
`xcomp` sends into the window's `xonly` side is found by bisection, and on a piece inside it
the length is the difference of the two inverses `yof(x, w)`, `w` the ends of the other
side, each clamped to `[y0, y1]`: the integral of a clamped inverse over a piece is the
piece's width times the clamp value where the inverse lies beyond it, and `yint` over the
piece where it lies within `[y0, y1]`, intersected with the piece's width times the range of
the inverse.  A piece on which `xcomp` crosses the window's end, the inverse crosses `y0` or
`y1`, or the monotonicity of `F` in `y` cannot be told, is bisected, up to `bisect` times
from the whole box, and is then bounded by its width times the range of the integrand over
it.  Without `yint` every piece is bounded that way, which makes the enclosure a Riemann sum
over `2^bisect` pieces.
"""
function fibred_area(xonly::Int, xcomp, yof; yint = nothing, bisect::Int = 12)
    xonly in (1, 2) || throw(ArgumentError("xonly must be 1 or 2, got $xonly"))
    bisect >= 0 || throw(ArgumentError("bisect must be nonnegative"))
    return _FibredArea(xonly, xcomp, yof, yint, bisect)
end

# The `area` function of `fibred_area`, a callable; called by the tree through `_call` with
# its scratch, so that the stack of pieces is not allocated at every call.
struct _FibredArea{XC,YO,YI}
    xonly::Int
    xcomp::XC
    yof::YO
    yint::YI
    bisect::Int
end
const _Piece = Tuple{Interval{Float64},Int,Int,Interval{Float64},Int}
(a::_FibredArea)(box::IntervalBox{2}, W::IntervalBox{2}) =
    _fibred_area(box, W, a.xonly, a.xcomp, a.yof, a.yint, a.bisect, sizehint!(_Piece[], 4 * a.bisect + 4))
_call(a::_FibredArea, box::IntervalBox{2}, W::IntervalBox{2}, s) =
    _fibred_area(box, W, a.xonly, a.xcomp, a.yof, a.yint, a.bisect, s.stack)
_call(area, box::IntervalBox{2}, W::IntervalBox{2}, s) = area(box, W)

_width(X::Interval) = interval(sup(X)) - interval(inf(X))
_clamp(V, y0, y1) = max(min(V, y1), y0)                # min and max of intervals end by end

# The integral is a sum of signed pieces, taken off an explicit stack rather than by
# recursion (four routines calling one another recursively were compiled with boxed
# returns).  A piece is an interval `X` of x at a remaining bisection depth, at one of three
# stages: 1, the test of `xcomp(X)` against the window; 2, the orientation of the fibre, which
# turns the piece into two clamped inverses of opposite signs; 3, the integral of one clamped
# inverse `yof(x, w)` with its sign.
function _fibred_area(box::IntervalBox{2}, W::IntervalBox{2}, xonly, xcomp, yof, yint, bisect, stack::Vector{_Piece})
    X0, Y0 = box[1], box[2]
    Wx, Wy = W[xonly], W[3 - xonly]
    y0, y1 = interval(inf(Y0)), interval(sup(Y0))
    a, b = interval(inf(Wy)), interval(sup(Wy))
    total = interval(0.0)
    empty!(stack)
    push!(stack, (X0, bisect, 1, a, 1))
    while !isempty(stack)
        X, depth, stage, w, sign = pop!(stack)
        m = mid(X)
        if stage == 1
            P = xcomp(X)
            isdisjoint_interval(P, Wx) && continue
            if issubset_interval(P, Wx)
                push!(stack, (X, depth, 2, a, 1))
            elseif depth == 0
                total += interval(0.0, 1.0) * _ybound(X, a, b, y0, y1, yof)
            else
                push!(stack, (interval(inf(X), m), depth - 1, 1, a, 1))
                push!(stack, (interval(m, sup(X)), depth - 1, 1, a, 1))
            end
        elseif stage == 2
            A, B = yof(X, a), yof(X, b)
            if sup(A) <= inf(B)                         # increasing in y on X
                push!(stack, (X, depth, 3, b, 1)); push!(stack, (X, depth, 3, a, -1))
            elseif sup(B) <= inf(A)
                push!(stack, (X, depth, 3, a, 1)); push!(stack, (X, depth, 3, b, -1))
            elseif depth == 0
                total += _ybound(X, a, b, y0, y1, yof)
            else
                push!(stack, (interval(inf(X), m), depth - 1, 2, a, 1))
                push!(stack, (interval(m, sup(X)), depth - 1, 2, a, 1))
            end
        else                                            # ∫_X clamp(yof(x, w), y0, y1) dx
            V = yof(X, w)
            term = if sup(V) <= inf(y0)
                _width(X) * y0
            elseif inf(V) >= sup(y1)
                _width(X) * y1
            elseif yint !== nothing && inf(V) >= inf(y0) && sup(V) <= sup(y1)
                intersect_interval(yint(interval(inf(X)), _width(X), w), _width(X) * V)
            elseif depth == 0
                _width(X) * _clamp(V, y0, y1)
            else
                push!(stack, (interval(inf(X), m), depth - 1, 3, w, sign))
                push!(stack, (interval(m, sup(X)), depth - 1, 3, w, sign))
                continue
            end
            total += sign == 1 ? term : -term
        end
    end
    return total / (_width(X0) * _width(Y0))
end

function _ybound(X, a, b, y0, y1, yof)
    A, B = yof(X, a), yof(X, b)
    return _width(X) * (_clamp(max(A, B), y0, y1) - _clamp(min(A, B), y0, y1))
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
