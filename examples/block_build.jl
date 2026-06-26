# Stage 1 (RigorousUlam2D env): build the noisy transfer operator L for each map and form
# the OVERSIZED Arnoldi compression H_{N} (Rayleigh/Hessenberg block, N = n_safe + buffer).
# Serialize H_N + the Arnoldi residual β_N for certification by miyajima_rump in the
# RigorousPseudospectra env (the env split is unavoidable: IntervalArithmetic 0.22 vs 1.x).
#
#   julia --project=examples examples/block_build.jl [N] [depth]

import Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()
using RigorousUlam2D, IntervalArithmetic, IntervalBoxes, LinearAlgebra, SparseArrays, Serialization, Printf

function henon_rescaled(x, y; a = 1.4, b = 0.3)
    xp = 1.5x; yp = 0.5y
    return ((1 - a * xp^2 + yp) / 1.5, (b * xp) / 0.5)
end
function contracting_lorenz(x, y; s = 0.5, r = 3.0, ρ = 1.3, c0 = -0.25, c1 = 0.25)
    ax = abs(x)
    if x isa Interval
        if inf(x) > 0
            return (-ρ * ax^s + 0.5, y * ax^r + c0)
        elseif sup(x) < 0
            return (ρ * ax^s - 0.5, -y * ax^r + c1)
        else
            return (RigorousUlam2D.hull(-ρ * ax^s + 0.5, ρ * ax^s - 0.5),
                RigorousUlam2D.hull(y * ax^r + c0, -y * ax^r + c1))
        end
    else
        xv = x isa Real ? x : ForwardDiff.value(x)
        return xv > 0 ? (-ρ * ax^s + 0.5, y * ax^r + c0) : (ρ * ax^s - 0.5, -y * ax^r + c1)
    end
end

function build_L(F, domain, noise, depth, periodic, enc)
    part = build_uniform_partition(domain, depth); n = part.n_boxes; nx = ny = 2^depth
    b2g, g2b = build_grid_index_maps(part, (nx, ny))
    Pd = build_transfer_forward(F, part, (nx, ny); subdiv = 4, enclosure_method = enc, periodic = periodic)
    normalize_columns!(Pd)
    dx = diam(domain[1]) / nx; rx = max(1, Int(round((diam(noise[1]) / 2) / dx)))
    w = 1.0 / ((2rx + 1)^2); I = Int[]; J = Int[]; V = Float64[]
    for j in 1:n
        jx, jy = b2g[j]
        for ox in (-rx):rx, oy in (-rx):rx
            ix = jx + ox; iy = jy + oy
            if periodic
                ix = mod1(ix, nx); iy = mod1(iy, ny)
            else
                (1 <= ix <= nx && 1 <= iy <= ny) || continue
            end
            t = g2b[iy, ix]; t == 0 && continue
            push!(I, t); push!(J, j); push!(V, w)
        end
    end
    return sparse(I, J, V, n, n) * Pd, n
end

# Arnoldi compression: returns the N×N Hessenberg H_N (= Vₙ*L Vₙ) and the residual β_N =
# h_{N+1,N} (Arnoldi: L Vₙ = Vₙ H_N + β_N q_{N+1} eₙᵀ; β_N·|last Ritz comp| bounds the
# distance of each Ritz value to σ(L)).
function arnoldi_H(L, N; v0 = randn(ComplexF64, size(L, 1)))
    n = size(L, 1); V = zeros(ComplexF64, n, N + 1); H = zeros(ComplexF64, N + 1, N)
    V[:, 1] = v0 ./ norm(v0)
    for k in 1:N
        w = L * V[:, k]
        for _ in 1:2                                   # modified GS + reorthogonalisation
            for i in 1:k
                c = dot(V[:, i], w); H[i, k] += c; w .-= c .* V[:, i]
            end
        end
        H[k + 1, k] = norm(w)
        abs(H[k + 1, k]) < 1e-13 && return H[1:k, 1:k], abs(H[k + 1, k]), k
        V[:, k + 1] = w ./ H[k + 1, k]
    end
    return H[1:N, 1:N], abs(H[N + 1, N]), N
end

N = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 12
depth = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 6

for (F, dom, per, enc, tag) in (
    (henon_rescaled, IntervalBox(interval(-1.0, 1.0), interval(-1.0, 1.0)), true,
        first_order_enclosure, "henon"),
    (contracting_lorenz, IntervalBox(interval(-0.5, 0.5), interval(-0.5, 0.5)), false,
        zeroth_order_enclosure, "lorenz"))
    L, n = build_L(F, dom, uniform_noise_box(0.1, 2), depth, per, enc)
    H, βN, k = arnoldi_H(L, N)
    out = joinpath(@__DIR__, "block_$(tag)_d$(depth)_N$(k).jls")
    serialize(out, (; tag, depth, n, N = k, H, βN))
    @printf("%-8s depth %d, n=%d: Arnoldi H_%d, β_N=%.3e → %s\n", tag, depth, n, k, βN, basename(out))
    @printf("    Ritz |λ|: %s\n", join(round.(sort(abs.(eigvals(H)); rev = true)[1:min(k, 8)]; digits = 4), ", "))
end
