import Pkg
Pkg.activate(@__DIR__)

using RigorousUlam2D
using IntervalArithmetic, IntervalBoxes
using LinearAlgebra, SparseArrays
using Printf

# Coarse-scale mixing rates for the two target systems.
# Partition at δ = ξ/2 so grid-aligned noise has rx=1 (noise spans 3×3 cells).
# Matrices are tiny; we compute everything with dense arithmetic.

function henon_rescaled(x, y; a=1.4, b=0.3)
    x_phys = 1.5 * x
    y_phys = 0.5 * y
    x_new = (1 - a * x_phys^2 + y_phys) / 1.5
    y_new = (b * x_phys) / 0.5
    return (x_new, y_new)
end

function contracting_lorenz(x, y; s=0.5, r=3.0, ρ=1.3, c0=-0.25, c1=0.25)
    ax = abs(x)
    if x isa Interval
        if inf(x) > 0
            return (-ρ * ax^s + 0.5, y * ax^r + c0)
        elseif sup(x) < 0
            return (ρ * ax^s - 0.5, -y * ax^r + c1)
        else
            Tx_pos = -ρ * ax^s + 0.5; Gxy_pos = y * ax^r + c0
            Tx_neg = ρ * ax^s - 0.5; Gxy_neg = -y * ax^r + c1
            return (RigorousUlam2D.hull(Tx_pos, Tx_neg), RigorousUlam2D.hull(Gxy_pos, Gxy_neg))
        end
    else
        x_val = x isa Real ? x : ForwardDiff.value(x)
        if x_val > 0
            return (-ρ * ax^s + 0.5, y * ax^r + c0)
        else
            return (ρ * ax^s - 0.5, -y * ax^r + c1)
        end
    end
end

# ||P^n|_V||_{L^1} = Dobrushin coefficient of P^n
# = (1/2) max_{j,k} ||P^n[:,j] - P^n[:,k]||_1
function dobrushin(Pn::AbstractMatrix)
    n = size(Pn, 2)
    maxnorm = 0.0
    for j in 1:n, k in (j+1):n
        d = 0.0
        for i in 1:size(Pn, 1)
            d += abs(Pn[i, j] - Pn[i, k])
        end
        maxnorm = max(maxnorm, d)
    end
    return maxnorm / 2
end

# Doeblin constant: ε = sum_i min_j P^n[i,j]
function doeblin(Pn::AbstractMatrix)
    m, n = size(Pn)
    ε = 0.0
    for i in 1:m
        mn = Pn[i, 1]
        for j in 2:n
            mn = min(mn, Pn[i, j])
        end
        ε += mn
    end
    return ε
end

function analyze(name, F, domain, ξ, depth; periodic, enclosure=first_order_enclosure)
    println("="^70)
    @printf("%s: ξ = %.4f, depth = %d (%d boxes)\n", name, ξ, depth, 4^depth)
    println("="^70)

    noise = uniform_noise_box(ξ/2, 2)

    # build_adaptive_partition with initial_depth=depth and a large max_diameter
    # returns a uniform partition of 4^depth boxes plus matching dyn/topo graphs.
    partition, dyn_graph, topo_graph = build_adaptive_partition(
        F, domain, noise;
        initial_depth=depth,
        max_diameter=1e9,
        enclosure_method=enclosure,
        periodic=periodic,
        verbose=false,
    )

    P_det, _ = build_transition_matrix(
        F, partition;
        transition_graph=dyn_graph,
        periodic=periodic,
        enclosure_method=enclosure,
        max_iterations=8,
        ε=1e-4,
    )
    N = build_grid_aligned_noise_operator(
        noise, partition;
        topological_graph=topo_graph,
        periodic=periodic,
    )
    P = N * P_det
    normalize_columns!(P)

    Pdense = Matrix(P)
    λ = sort(abs.(eigvals(Pdense)), rev=true)
    @printf("  |λ_1| = %.6f, |λ_2| = %.6f, gap = %.6f\n",
            λ[1], length(λ) >= 2 ? λ[2] : NaN,
            length(λ) >= 2 ? 1 - λ[2] : NaN)

    Pn = copy(Pdense)
    @printf("%4s  %10s  %10s\n", "n", "α=Dob(Pⁿ)", "ε=Doeb(Pⁿ)")
    for n in 1:10
        α = dobrushin(Pn)
        ε = doeblin(Pn)
        @printf("%4d  %10.6f  %10.6f\n", n, α, ε)
        Pn = Pn * Pdense
    end
    println()
end

# Hénon periodic, domain [-1,1]^2
henon_domain = IntervalBox(interval(-1.0, 1.0), interval(-1.0, 1.0))
for (ξ, depth) in [(0.5, 3), (0.25, 4), (0.125, 5)]
    analyze("Hénon periodic", henon_rescaled, henon_domain, ξ, depth; periodic=true)
end

# Contracting Lorenz, domain [-0.5, 0.5]^2
lorenz_domain = IntervalBox(interval(-0.5, 0.5), interval(-0.5, 0.5))
for (ξ, depth) in [(0.5, 2), (0.25, 3), (0.125, 4)]
    analyze("Contracting Lorenz", contracting_lorenz, lorenz_domain, ξ, depth;
            periodic=true, enclosure=zeroth_order_enclosure)
end
