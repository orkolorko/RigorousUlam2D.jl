import Pkg
Pkg.activate(@__DIR__)

using RigorousUlam2D
using IntervalArithmetic, IntervalBoxes
using Printf

# Iterate the ξ/2-grid of points under T (rigorous intervals) and record the
# first time each orbit hits B_ξ(p) for a chosen target point p on the attractor.

function henon_rescaled(x, y; a=1.4, b=0.3)
    x_phys = 1.5 * x
    y_phys = 0.5 * y
    x_new = (1 - a * x_phys^2 + y_phys) / 1.5
    y_new = (b * x_phys) / 0.5
    return (x_new, y_new)
end

function contracting_lorenz(x, y; s=0.5, r=3.0, ρ=1.3, c0=-0.25, c1=0.25)
    ax = abs(x)
    if inf(x) > 0
        return (-ρ * ax^s + 0.5, y * ax^r + c0)
    elseif sup(x) < 0
        return (ρ * ax^s - 0.5, -y * ax^r + c1)
    else
        Tx_pos = -ρ * ax^s + 0.5; Gxy_pos = y * ax^r + c0
        Tx_neg = ρ * ax^s - 0.5; Gxy_neg = -y * ax^r + c1
        return (RigorousUlam2D.hull(Tx_pos, Tx_neg), RigorousUlam2D.hull(Gxy_pos, Gxy_neg))
    end
end

# Wrap into domain interval for periodic BC
function wrap_periodic(z::Interval, a::Float64, b::Float64)
    L = b - a
    k_lo = floor(inf(z - a) / L)
    k_hi = floor(sup(z - a) / L)
    if k_lo == k_hi
        return z - k_lo * L
    else
        return interval(a, b)  # spans wrap → conservative full-domain interval
    end
end

# ξ/2 grid of POINTS (cell centers, so we never land on a map discontinuity at x=0)
function grid_points(domain::IntervalBox{2}, spacing::Float64)
    ax = inf(domain[1]); bx = sup(domain[1])
    ay = inf(domain[2]); by = sup(domain[2])
    Nx = Int(round((bx - ax) / spacing))
    Ny = Int(round((by - ay) / spacing))
    pts = IntervalBox{2,Float64,Interval{Float64}}[]
    for ix in 0:Nx-1, iy in 0:Ny-1
        x = ax + (ix + 0.5) * spacing
        y = ay + (iy + 0.5) * spacing
        push!(pts, IntervalBox(interval(x), interval(y)))
    end
    return pts
end

function analyze(name, F, domain, p, ξ, Nmax; periodic::Bool)
    println("="^70)
    @printf("%s: ξ = %.4f, target p = (%.4f, %.4f), Nmax = %d\n",
            name, ξ, p[1], p[2], Nmax)
    println("="^70)

    spacing = ξ / 2
    grid = grid_points(domain, spacing)
    K = length(grid)
    @printf("  Grid: %d points at spacing %.4f\n", K, spacing)

    px = interval(p[1]); py = interval(p[2])
    ax = inf(domain[1]); bx = sup(domain[1])
    ay = inf(domain[2]); by = sup(domain[2])

    pts = copy(grid)
    visited = falses(K)     # has each orbit visited B_ξ(p)?
    first_visit = fill(-1, K)

    @printf("%4s  %9s  %12s  %14s\n", "N", "visited", "min_dist", "max_box_width")
    for N in 1:Nmax
        new_pts = similar(pts)
        min_dist = Inf
        max_width = 0.0
        @inbounds for k in 1:K
            b = pts[k]
            (x2, y2) = F(b[1], b[2])
            if periodic
                x2 = wrap_periodic(x2, ax, bx)
                y2 = wrap_periodic(y2, ay, by)
            end
            new_pts[k] = IntervalBox(x2, y2)
            # Is this orbit currently within ξ of p?
            dx = max(sup(x2 - px), -inf(x2 - px))
            dy = max(sup(y2 - py), -inf(y2 - py))
            dist = max(dx, dy)
            if !visited[k] && dist ≤ ξ
                visited[k] = true
                first_visit[k] = N
            end
            min_dist = min(min_dist, dist)
            width = max(diam(x2), diam(y2))
            max_width = max(max_width, width)
        end
        pts = new_pts
        nvis = count(visited)
        @printf("%4d  %4d/%4d  %12.6e  %14.6e\n", N, nvis, K, min_dist, max_width)
        if nvis == K
            @printf("  --> EVERY orbit visited B_ξ(p) by N = %d\n", N)
            @printf("      max first-visit time: %d\n", maximum(first_visit))
            return
        end
    end
    nvis = count(visited)
    @printf("  --> %d/%d orbits visited B_ξ(p) within %d steps\n", nvis, K, Nmax)
end

# Scalar (Float64) versions of the maps, for finding p by forward iteration.
function henon_scalar(xy)
    x, y = xy
    x_phys = 1.5 * x; y_phys = 0.5 * y
    x_new = (1 - 1.4 * x_phys^2 + y_phys) / 1.5
    y_new = (0.3 * x_phys) / 0.5
    # wrap to [-1, 1]
    x_new = x_new - 2.0 * floor((x_new + 1.0) / 2.0)
    y_new = y_new - 2.0 * floor((y_new + 1.0) / 2.0)
    return (x_new, y_new)
end

function lorenz_scalar(xy; s=0.5, r=3.0, ρ=1.3, c0=-0.25, c1=0.25)
    x, y = xy
    ax = abs(x)
    if x > 0
        x_new, y_new = -ρ * ax^s + 0.5, y * ax^r + c0
    else
        x_new, y_new = ρ * ax^s - 0.5, -y * ax^r + c1
    end
    # wrap to [-0.5, 0.5]
    x_new = x_new - 1.0 * floor(x_new + 0.5)
    y_new = y_new - 1.0 * floor(y_new + 0.5)
    return (x_new, y_new)
end

# Iterate n times in Float64 to produce a point on the attractor
function find_p_on_attractor(Fscalar, start, n)
    p = start
    for _ in 1:n
        p = Fscalar(p)
    end
    return p
end

# --- Hénon periodic ---
henon_domain = IntervalBox(interval(-1.0, 1.0), interval(-1.0, 1.0))
p_henon = find_p_on_attractor(henon_scalar, (0.1, 0.1), 10_000)
@printf("\np_henon (after 10000 Hénon iterates) = (%.6f, %.6f)\n\n", p_henon...)
for ξ in [0.5, 0.25, 0.125]
    analyze("Hénon periodic", henon_rescaled, henon_domain, p_henon, ξ, 60;
            periodic=true)
end

# --- Contracting Lorenz on torus ---
lorenz_domain = IntervalBox(interval(-0.5, 0.5), interval(-0.5, 0.5))
p_lorenz = find_p_on_attractor(lorenz_scalar, (0.1, 0.1), 10_000)
@printf("\np_lorenz (after 10000 Lorenz iterates) = (%.6f, %.6f)\n\n", p_lorenz...)
for ξ in [0.5, 0.25, 0.125]
    analyze("Contracting Lorenz", contracting_lorenz, lorenz_domain, p_lorenz, ξ, 60;
            periodic=true)
end
