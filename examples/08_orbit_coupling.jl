import Pkg
Pkg.activate(@__DIR__)

using RigorousUlam2D
using IntervalArithmetic, IntervalBoxes
using Printf

# Pairwise coupling algorithm:
#   - start with ξ/2 grid of point orbits
#   - at each step k, iterate all orbits under T
#   - for any two orbits (in different classes) with |T^k(g_i) - T^k(g_j)| < ξ,
#     the noise kernels ρ_ξ(·-T^k(g_i)) and ρ_ξ(·-T^k(g_j)) overlap with mass
#     w = ∏_a (1 - |Δa|/ξ)  (box noise, coordinate-wise)
#     Couple them (merge classes), forget one representative, accumulate w.
#   - stop when all orbits are in a single class.
#
# The accumulated product ε = ∏ w_merges is a lower bound on the coupling
# probability of the synchronously-coupled chain, hence on the Doeblin constant.

function henon_rescaled(x, y; a=1.4, b=0.3)
    x_phys = 1.5 * x; y_phys = 0.5 * y
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

function wrap_periodic(z::Interval, a::Float64, b::Float64)
    L = b - a
    k_lo = floor(inf(z - a) / L)
    k_hi = floor(sup(z - a) / L)
    if k_lo == k_hi
        return z - k_lo * L
    else
        return interval(a, b)
    end
end

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

# --- minimal union-find ---
mutable struct UF
    parent::Vector{Int}
    rank::Vector{Int}
    nclasses::Int
end
UF(n) = UF(collect(1:n), zeros(Int, n), n)

function find!(uf::UF, i)
    r = i
    while uf.parent[r] != r
        r = uf.parent[r]
    end
    while uf.parent[i] != r
        uf.parent[i], i = r, uf.parent[i]
    end
    return r
end

function union!(uf::UF, i, j)
    ri, rj = find!(uf, i), find!(uf, j)
    ri == rj && return false
    if uf.rank[ri] < uf.rank[rj]
        uf.parent[ri] = rj
    elseif uf.rank[ri] > uf.rank[rj]
        uf.parent[rj] = ri
    else
        uf.parent[rj] = ri
        uf.rank[ri] += 1
    end
    uf.nclasses -= 1
    return true
end

# periodic L∞ distance on torus of width L per axis
function periodic_dist_linf(bx1::IntervalBox{2}, bx2::IntervalBox{2}, Lx::Float64, Ly::Float64)
    # use midpoints (intervals are near-zero width)
    mx1 = (sup(bx1[1]) + inf(bx1[1])) / 2
    my1 = (sup(bx1[2]) + inf(bx1[2])) / 2
    mx2 = (sup(bx2[1]) + inf(bx2[1])) / 2
    my2 = (sup(bx2[2]) + inf(bx2[2])) / 2
    dx = abs(mx1 - mx2); dx = min(dx, Lx - dx)
    dy = abs(my1 - my2); dy = min(dy, Ly - dy)
    return (dx, dy)
end

function analyze(name, F, domain, ξ, Nmax; periodic::Bool)
    println("="^70)
    @printf("%s: ξ = %.4f, Nmax = %d\n", name, ξ, Nmax)
    println("="^70)

    spacing = ξ / 2
    pts = grid_points(domain, spacing)
    K = length(pts)
    Lx = sup(domain[1]) - inf(domain[1])
    Ly = sup(domain[2]) - inf(domain[2])
    ax = inf(domain[1]); bx = sup(domain[1])
    ay = inf(domain[2]); by = sup(domain[2])

    uf = UF(K)
    log_eps = 0.0
    @printf("  Grid: %d points at spacing %.4f\n", K, spacing)
    @printf("%4s  %9s  %14s  %14s\n", "N", "classes", "log₁₀(ε)", "last_w")

    # step 0: before any iteration, points at spacing ξ/2, so pairwise distance
    # is ξ/2 at nearest, already eligible for coupling.
    last_w = 0.0
    for k in 1:Nmax
        # Iterate every surviving representative.
        new_pts = similar(pts)
        for i in 1:K
            b = pts[i]
            (x2, y2) = F(b[1], b[2])
            if periodic
                x2 = wrap_periodic(x2, ax, bx)
                y2 = wrap_periodic(y2, ay, by)
            end
            new_pts[i] = IntervalBox(x2, y2)
        end
        pts = new_pts

        # Merge class representatives pairwise where |Δ| < ξ (torus L∞).
        # Use one representative per class; union any close pair.
        reps = Int[]
        seen = Dict{Int,Int}()
        for i in 1:K
            r = find!(uf, i)
            if !haskey(seen, r)
                seen[r] = i; push!(reps, i)
            end
        end

        @inbounds for a in 1:length(reps), bidx in (a+1):length(reps)
            i, j = reps[a], reps[bidx]
            (dx, dy) = periodic_dist_linf(pts[i], pts[j], Lx, Ly)
            if dx < ξ && dy < ξ
                w = (1 - dx/ξ) * (1 - dy/ξ)
                if union!(uf, i, j)
                    log_eps += log(w)
                    last_w = w
                end
            end
        end

        @printf("%4d  %4d/%4d  %14.6e  %14.6e\n",
                k, uf.nclasses, K, log_eps / log(10), last_w)
        if uf.nclasses == 1
            eps = exp(log_eps)
            @printf("  --> FULL coupling at N = %d, ε ≥ %.6e (log₁₀ ε ≈ %.3f)\n",
                    k, eps, log_eps / log(10))
            return
        end
    end
    @printf("  --> %d classes remain after %d steps (not fully coupled)\n",
            uf.nclasses, Nmax)
end

# --- Hénon periodic on [-1,1]² ---
henon_domain = IntervalBox(interval(-1.0, 1.0), interval(-1.0, 1.0))
for ξ in [0.5, 0.25, 0.125]
    analyze("Hénon periodic", henon_rescaled, henon_domain, ξ, 60; periodic=true)
end

# --- Contracting Lorenz on [-0.5, 0.5]² torus ---
lorenz_domain = IntervalBox(interval(-0.5, 0.5), interval(-0.5, 0.5))
for ξ in [0.5, 0.25, 0.125]
    analyze("Contracting Lorenz", contracting_lorenz, lorenz_domain, ξ, 60; periodic=true)
end
