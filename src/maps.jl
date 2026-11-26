"""
The two maps of the paper's examples (Section 6 of `RigorousUlam2D.tex`), with what `UlamTree`
needs to read their Ulam matrices: the map as an interval extension, its domain, and the
`fibred_area` of each.  Every parameter is an exact rational entered as an interval quotient
(`7/5` as `interval(7)/interval(5)`), so the maps are those of the paper and not of the nearest
floating-point parameters; the multipliers that are dyadic (`3/2`, `1/2`, `9/4`) are exact as
`Float64`.

- `henon_system`: `H(X, Y) = (1 − a X² + Y, b X)`, `a = 7/5`, `b = 3/10`, in the coordinates
  `X = 3x/2`, `Y = y/2` that send `[-1, 1]²` to the box of the attractor, periodic:
  `F(x, y) = ((1 − a (3x/2)² + y/2) / (3/2), 3b x)`.  The second component depends on `x`
  alone (`x' = 9x/10` after the rescaling), the first is increasing in `y`, with inverse
  `y = 3w − 2 + 2a (3x/2)²` and integral `δ(3w − 2) + (3a/2) δ (3x² + 3xδ + δ²)` over
  `[x, x + δ]`; the crossing of `9x/10 = w` is `x = 10w/9`.
- `lorenz_system`: the contracting Lorenz map on `[-1/2, 1/2]²`, periodic, with `s = 1/2`,
  `r = 3`, `ρ = 13/10`, `c0 = −1/4`, `c1 = 1/4`: for `x > 0`,
  `F = (1/2 − ρ x^s, y x^r + c0)`, for `x < 0`, `F = (ρ |x|^s − 1/2, −y |x|^r + c1)`.  The
  first component depends on `x` alone; the second is monotone in `y`, with inverse
  `(w − c0)/x³` (`x > 0`) and `(c1 − w)/|x|³` (`x < 0`), whose integrals over `[x, x + δ]` are
  `±(w ∓ c)δ(2x + δ)/(2x²(x + δ)²)`, written over a common denominator as `fibred_area` asks;
  the crossings of the first component with `w` are `x = ((1/2 − w)/ρ)²` and
  `x = −((w + 1/2)/ρ)²`.  An interval of `x` meeting both signs is given the hull of the two
  branches.  Every grid line of the domain at depth `≥ 1` passes through `x = 0`, so no cell
  of a tree straddles it.

`xinv` encloses every crossing in `X` of the branch that `X` lies on: one point for Hénon, one
for each sign of `x` for Lorenz, so the crossing enclosures of the two ends of a window never
nest (see `_cut!`).
"""

export henon_system, lorenz_system, contracting_lorenz_system

const _A_HENON = interval(7) / interval(5)
const _B_HENON = interval(3) / interval(10)
const _RHO_LORENZ = interval(13) / interval(10)

"""
    henon_system(; bisect = 30, xinv = true) -> (; name, F, domain, area)

The rescaled Hénon map of the file docstring on `[-1, 1]²`, its `fibred_area` with `bisect`
bisections, and the cut at the inverse of the `x`-only component when `xinv`.
"""
function henon_system(; bisect::Int = 30, xinv::Bool = true)
    a, b = _A_HENON, _B_HENON
    c = interval(3) * b                                     # 9/10
    F(x, y) = ((interval(1) - a * (interval(1.5) * x)^2 + interval(0.5) * y) / interval(1.5), c * x)
    xcomp(x) = c * x
    yof(x, w) = interval(3) * w - interval(2) + interval(2) * a * (interval(1.5) * x)^2
    yint(x, δ, w) = δ * (interval(3) * w - interval(2)) +
                    interval(1.5) * a * δ * (interval(3) * x^2 + interval(3) * x * δ + δ^2)
    inv(X, w) = w / c
    area = fibred_area(2, xcomp, yof; yint, xinv = xinv ? inv : nothing, bisect)
    domain = IntervalBox(interval(-1.0, 1.0), interval(-1.0, 1.0))
    return (; name = "henon", F, domain, area)
end

"""
    lorenz_system(; bisect = 30, xinv = true) -> (; name, F, domain, area)

The contracting Lorenz map of the file docstring on `[-1/2, 1/2]²`, its `fibred_area` with
`bisect` bisections, and the cut at the inverse of the `x`-only component when `xinv`.
"""
function lorenz_system(; bisect::Int = 30, xinv::Bool = true)
    ρ, c0, c1, h = _RHO_LORENZ, interval(-0.25), interval(0.25), interval(0.5)
    pos(x) = inf(x) >= 0
    neg(x) = sup(x) <= 0
    function F(x, y)
        ax = abs(x)
        Pp = (h - ρ * sqrt(ax), y * ax^3 + c0)
        Pn = (ρ * sqrt(ax) - h, -y * ax^3 + c1)
        pos(x) && return Pp
        neg(x) && return Pn
        return (hull(Pp[1], Pn[1]), hull(Pp[2], Pn[2]))
    end
    function xcomp(x)
        ax = abs(x)
        pos(x) && return h - ρ * sqrt(ax)
        neg(x) && return ρ * sqrt(ax) - h
        return hull(h - ρ * sqrt(ax), ρ * sqrt(ax) - h)
    end
    function yof(x, w)
        ax = abs(x)
        pos(x) && return (w - c0) / ax^3
        neg(x) && return (c1 - w) / ax^3
        return hull((w - c0) / ax^3, (c1 - w) / ax^3)
    end
    # ∫_x^{x+δ} of the inverse; called on pieces of one sign only
    yint(x, δ, w) = (pos(x) ? (w - c0) : -(c1 - w)) * δ * (interval(2) * x + δ) /
                    (interval(2) * x^2 * (x + δ)^2)
    function inv(X, w)
        pos(X) && return ((h - w) / ρ)^2
        neg(X) && return -((w + h) / ρ)^2
        return hull(((h - w) / ρ)^2, -((w + h) / ρ)^2)
    end
    area = fibred_area(1, xcomp, yof; yint, xinv = xinv ? inv : nothing, bisect)
    domain = IntervalBox(interval(-0.5, 0.5), interval(-0.5, 0.5))
    return (; name = "lorenz", F, domain, area)
end

"""
    contracting_lorenz_system(; ρ = 4//1, bisect = 30, xinv = true) -> (; name, F, domain, area)

A contracting Lorenz map on `[-1/2, 1/2]²`, periodic: for `x > 0`,
`F = (1/2 − ρ x², y x³ − 1/4)`, for `x < 0`, `F = (ρ x² − 1/2, −y |x|³ + 1/4)`.  The first component
is the one-dimensional map `f` with `f′(x) = 2ρ|x|`, i.e. `K₂|x|^{s−1} ≤ f′ ≤ K₁|x|^{s−1}` with
`s = 2` (condition (A0) of Metzger, Ann. IHP 17 (2000) 247, p. 249), negative Schwarzian derivative
`−(3/2)(f″/f′)²`, and `f(0±) = ±1/2`; `ρ ≤ 4` keeps `f` in `[-1/2, 1/2]`.  The second component is
that of `lorenz_system`.  `ρ` is a rational, entered as an interval quotient; the name of the
system carries it (`clorenz_rho<p>-<q>`), so that its blocks and records are named apart.

The x-only component is monotone on each sign of `x`, with crossings of `w` at
`x = ((1/2 − w)/ρ)^{1/2}` and `x = −((w + 1/2)/ρ)^{1/2}` (empty where the radicand is negative);
the fibre inverse and its integral are those of `lorenz_system`.
"""
function contracting_lorenz_system(; ρ::Rational = 4 // 1, bisect::Int = 30, xinv::Bool = true)
    0 < ρ <= 4 || throw(ArgumentError("ρ = $ρ must be in (0, 4] for f to map [-1/2, 1/2] into itself"))
    r = interval(numerator(ρ)) / interval(denominator(ρ))
    c0, c1, h = interval(-0.25), interval(0.25), interval(0.5)
    pos(x) = inf(x) >= 0
    neg(x) = sup(x) <= 0
    function F(x, y)
        ax = abs(x)
        Pp = (h - r * ax^2, y * ax^3 + c0)
        Pn = (r * ax^2 - h, -y * ax^3 + c1)
        pos(x) && return Pp
        neg(x) && return Pn
        return (hull(Pp[1], Pn[1]), hull(Pp[2], Pn[2]))
    end
    function xcomp(x)
        ax = abs(x)
        pos(x) && return h - r * ax^2
        neg(x) && return r * ax^2 - h
        return hull(h - r * ax^2, r * ax^2 - h)
    end
    function yof(x, w)
        ax = abs(x)
        pos(x) && return (w - c0) / ax^3
        neg(x) && return (c1 - w) / ax^3
        return hull((w - c0) / ax^3, (c1 - w) / ax^3)
    end
    yint(x, δ, w) = (pos(x) ? (w - c0) : -(c1 - w)) * δ * (interval(2) * x + δ) /
                    (interval(2) * x^2 * (x + δ)^2)
    # the radicand intersected with [0, ∞): empty where f takes no value w on that side
    root(t) = (q = intersect_interval(t, interval(0.0, Inf)); isempty_interval(q) ? q : sqrt(q))
    function inv(X, w)
        pos(X) && return root((h - w) / r)
        neg(X) && return -root((w + h) / r)
        return hull(root((h - w) / r), -root((w + h) / r))
    end
    area = fibred_area(1, xcomp, yof; yint, xinv = xinv ? inv : nothing, bisect)
    domain = IntervalBox(interval(-0.5, 0.5), interval(-0.5, 0.5))
    return (; name = "clorenz_rho$(numerator(ρ))-$(denominator(ρ))", F, domain, area)
end
