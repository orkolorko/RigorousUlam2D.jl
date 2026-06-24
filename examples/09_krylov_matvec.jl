# Krylov behaviour of the matrix-free transfer / Koopman operators.
#
# The noisy Ulam operator is applied as a PURE MAT-VEC — P_det is NEVER formed as
# a matrix. On every application we recompute the forward image of each source
# cell from the map F and scatter/gather its mass:
#
#   Transfer (Perron–Frobenius):  v ↦ N (P_det v)     P_det v  by forward SCATTER
#   Koopman  (adjoint):           u ↦ P_detᵀ (N u)    P_detᵀ u by GATHER (transpose)
#
# Only two O(n) vectors are precomputed: the grid index maps and the column-mass
# vector `colsum` (so the on-the-fly action is column-stochastic). The noise N is a
# symmetric convolution stencil applied by `noise_matvec` (Nᵀ = N). Nothing O(n²)
# or the sparse P_det is ever stored — this is genuinely matrix-free.
#
# These operators are STRONGLY NON-NORMAL (contraction + transport), so Arnoldi /
# Ritz convergence is governed by pseudospectra, not eigenvalues. We instrument:
#   • Ritz convergence of the leading eigenvalues (λ₁ and the gap λ₂),
#   • the Arnoldi residual estimate  βₖ·|eₖᵀyᵢ| per Ritz pair,
#   • Henrici's departure-from-normality of the Hessenberg Hₖ,
#   • transfer vs Koopman (same spectrum, different Krylov behaviour),
# measured against a reference from a long matrix-free Arnoldi run.
#
#   julia --project=examples --threads=auto examples/09_krylov_matvec.jl
#   DEPTH=6 MSTEPS=40 MREF=70 julia --project=examples ... (env overrides)

import Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using RigorousUlam2D
using IntervalArithmetic, IntervalBoxes
using LinearAlgebra, SparseArrays
using Printf
using Random
using CairoMakie

Random.seed!(1234)
println("Julia threads: ", Threads.nthreads())
flush(stdout)

# ============================================================
# Maps (same as matvec_small_noise.jl)
# ============================================================
function henon_rescaled(x, y; a = 1.4, b = 0.3)
    x_phys = 1.5 * x
    y_phys = 0.5 * y
    x_new = (1 - a * x_phys^2 + y_phys) / 1.5
    y_new = (b * x_phys) / 0.5
    return (x_new, y_new)
end

function contracting_lorenz(x, y; s = 0.5, r = 3.0, ρ = 1.3, c0 = -0.25, c1 = 0.25)
    ax = abs(x)
    if x isa Interval
        if inf(x) > 0
            return (-ρ * ax^s + 0.5, y * ax^r + c0)
        elseif sup(x) < 0
            return (ρ * ax^s - 0.5, -y * ax^r + c1)
        else
            Tx_pos = -ρ * ax^s + 0.5
            Gxy_pos = y * ax^r + c0
            Tx_neg = ρ * ax^s - 0.5
            Gxy_neg = -y * ax^r + c1
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

# ============================================================
# Matrix-free deterministic Ulam action (forward-image scatter / gather).
# No matrix is stored — the image of each source cell is recomputed every call.
# ============================================================

# O(n) noise convolution by separable prefix-sum box filter.
# The uniform-noise kernel is a normalised top-hat over (2rx+1)×(2ry+1) cells, which
# is separable, so it applies in O(n) per matvec INDEPENDENT of the stencil radius r
# (vs O(n·r²) for the dense stencil). Matches `noise_matvec`: periodic = circular sum,
# absorbing = clamp window + drop boundary mass (uniform weight w, no renormalisation).
function box_noise_matvec(v::AbstractVector{T}, rx, ry, nx, ny, b2g; periodic) where {T}
    n = nx * ny
    w = one(real(T)) / ((2rx + 1) * (2ry + 1))
    V = zeros(T, ny, nx)
    for k in 1:n
        ix, iy = b2g[k]
        V[iy, ix] = v[k]
    end
    # x-direction windowed sum (radius rx)
    Sx = zeros(T, ny, nx)
    if periodic
        for iy in 1:ny
            s = zero(T)
            for d in -rx:rx
                s += V[iy, mod1(1 + d, nx)]
            end
            Sx[iy, 1] = s
            for ix in 2:nx
                s += V[iy, mod1(ix + rx, nx)] - V[iy, mod1(ix - 1 - rx, nx)]
                Sx[iy, ix] = s
            end
        end
    else
        P = zeros(T, nx + 1)
        for iy in 1:ny
            for ix in 1:nx
                P[ix + 1] = P[ix] + V[iy, ix]
            end
            for ix in 1:nx
                lo = max(1, ix - rx); hi = min(nx, ix + rx)
                Sx[iy, ix] = P[hi + 1] - P[lo]
            end
        end
    end
    # y-direction windowed sum (radius ry) on Sx
    W = zeros(T, ny, nx)
    if periodic
        for ix in 1:nx
            s = zero(T)
            for d in -ry:ry
                s += Sx[mod1(1 + d, ny), ix]
            end
            W[1, ix] = s
            for iy in 2:ny
                s += Sx[mod1(iy + ry, ny), ix] - Sx[mod1(iy - 1 - ry, ny), ix]
                W[iy, ix] = s
            end
        end
    else
        P = zeros(T, ny + 1)
        for ix in 1:nx
            for iy in 1:ny
                P[iy + 1] = P[iy] + Sx[iy, ix]
            end
            for iy in 1:ny
                lo = max(1, iy - ry); hi = min(ny, iy + ry)
                W[iy, ix] = P[hi + 1] - P[lo]
            end
        end
    end
    result = zeros(T, n)
    for k in 1:n
        ix, iy = b2g[k]
        result[k] = w * W[iy, ix]
    end
    return result
end

# Fold a (possibly out-of-range) grid index onto the torus / report absorption.
@inline function _fold(ix, iy, nx, ny, periodic)
    if periodic
        return (mod1(ix, nx), mod1(iy, ny))
    else
        (1 <= ix <= nx && 1 <= iy <= ny) ? (ix, iy) : (0, 0)
    end
end

# Raw (unnormalised) transition weights from source cell `j`: target box index → mass.
# This is one column of P_det, computed on the fly (mirrors build_transfer_forward).
function column_targets(ctx, j)
    (; F, partition, g2b, x0, y0, dx, dy, nx, ny, subdiv, enclosure_method, periodic) = ctx
    S = partition.boxes[j]
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
    acc = Dict{Int, Float64}()
    for s in leaves
        img = enclosure_method(F, s)
        ib = img isa Tuple ? IntervalBox(img...) : img
        xa, xb = inf(ib[1]), sup(ib[1])
        ya, yb = inf(ib[2]), sup(ib[2])
        area = (xb - xa) * (yb - ya)
        ix_lo = Int(floor((xa - x0) / dx)) + 1
        ix_hi = Int(floor((xb - x0) / dx)) + 1
        iy_lo = Int(floor((ya - y0) / dy)) + 1
        iy_hi = Int(floor((yb - y0) / dy)) + 1
        if area <= 0 || (ix_hi - ix_lo) > nx || (iy_hi - iy_lo) > ny
            cx = (xa + xb) / 2; cy = (ya + yb) / 2
            ix = Int(floor((cx - x0) / dx)) + 1
            iy = Int(floor((cy - y0) / dy)) + 1
            ixw, iyw = _fold(ix, iy, nx, ny, periodic)
            ixw == 0 && continue
            k = g2b[iyw, ixw]
            k != 0 && (acc[k] = get(acc, k, 0.0) + wleaf)
            continue
        end
        for ix in ix_lo:ix_hi
            cxa = x0 + (ix - 1) * dx
            ox = max(0.0, min(xb, cxa + dx) - max(xa, cxa))
            ox <= 0 && continue
            for iy in iy_lo:iy_hi
                cya = y0 + (iy - 1) * dy
                oy = max(0.0, min(yb, cya + dy) - max(ya, cya))
                oy <= 0 && continue
                ixw, iyw = _fold(ix, iy, nx, ny, periodic)
                ixw == 0 && continue
                k = g2b[iyw, ixw]
                k == 0 && continue
                acc[k] = get(acc, k, 0.0) + wleaf * (ox * oy) / area
            end
        end
    end
    return acc
end

# Precompute the column masses colsum[j] = Σ_target weight(j→target).  O(n) storage.
# (Absorbing BC loses mass at the boundary ⇒ colsum[j] < 1 there.)
function compute_colsum(ctx)
    n = ctx.partition.n_boxes
    cs = zeros(n)
    Threads.@threads :static for j in 1:n
        cs[j] = sum(values(column_targets(ctx, j)); init = 0.0)
    end
    return cs
end

# P_det · v  by forward SCATTER (matrix-free, column-normalised by colsum).
function det_transfer(ctx, v::AbstractVector{T}) where {T}
    n = ctx.partition.n_boxes
    nt = Threads.maxthreadid()
    buf = [zeros(T, n) for _ in 1:nt]
    Threads.@threads :static for j in 1:n
        cj = ctx.colsum[j]
        cj <= 0 && continue
        cw = v[j] / cj
        cw == 0 && continue
        b = buf[Threads.threadid()]
        for (k, w) in column_targets(ctx, j)
            b[k] += w * cw
        end
    end
    return reduce(+, buf)
end

# P_detᵀ · u  by GATHER (the transpose: same column geometry, accumulated into source).
function det_koopman(ctx, u::AbstractVector{T}) where {T}
    n = ctx.partition.n_boxes
    out = zeros(T, n)
    Threads.@threads :static for j in 1:n
        cj = ctx.colsum[j]
        cj <= 0 && continue
        s = zero(T)
        for (k, w) in column_targets(ctx, j)
            s += w * u[k]
        end
        out[j] = s / cj
    end
    return out
end

# ============================================================
# Build the matrix-free transfer + Koopman closures for one map.
# ============================================================
function build_operators(F, domain, noise, label;
        depth::Int = 6,
        enclosure_method = zeroth_order_enclosure,
        periodic::Bool = false,
        subdiv::Int = 4)
    partition = build_uniform_partition(domain, depth)
    n = partition.n_boxes
    nx = 2^depth; ny = 2^depth
    n == nx * ny || error("box count $n ≠ $(nx*ny)")
    b2g, g2b = build_grid_index_maps(partition, (nx, ny))

    ctx = (; F, partition, g2b, b2g,
        x0 = inf(domain[1]), y0 = inf(domain[2]),
        dx = diam(domain[1]) / nx, dy = diam(domain[2]) / ny,
        nx, ny, subdiv, enclosure_method, periodic,
        colsum = zeros(n))

    println("\n" * "="^70)
    @printf("%s  —  depth %d, %d×%d = %d boxes, periodic=%s\n",
        label, depth, nx, ny, n, periodic)
    println("="^70)
    flush(stdout)

    # Noise radius in cells (σ/δ): the ratio the study is about.
    σx = diam(noise[1]) / 2; σy = diam(noise[2]) / 2
    rx = max(1, Int(round(σx / ctx.dx))); ry = max(1, Int(round(σy / ctx.dy)))
    @printf("  noise σ = %.4f,  cell δ = %.6f,  σ/δ = %d cells  (stencil %d×%d)\n",
        σx, ctx.dx, rx, 2rx + 1, 2ry + 1)
    flush(stdout)

    print("  Precomputing column masses (one scatter pass)... ")
    tcs = @elapsed cs = compute_colsum(ctx)
    ctx = merge(ctx, (; colsum = cs))
    @printf("done (%.2fs).  colsum ∈ [%.4f, %.4f]\n", tcs, minimum(cs), maximum(cs))

    # O(n) separable box-filter noise (stencil radius is free).
    nm(v) = box_noise_matvec(v, rx, ry, nx, ny, b2g; periodic = periodic)

    # Transfer  v ↦ N (P_det v)   ;   Koopman  u ↦ P_detᵀ (N u)   (N symmetric)
    transfer = v -> nm(det_transfer(ctx, v))
    koopman  = u -> det_koopman(ctx, nm(u))

    return (; transfer, koopman, ctx, n, nx, ny, partition, label)
end

# ============================================================
# Instrumented Arnoldi (modified Gram–Schmidt + one reorthogonalisation).
# ============================================================
function arnoldi(Aop, n, m; v0 = randn(n))
    V = zeros(ComplexF64, n, m + 1)
    H = zeros(ComplexF64, m + 1, m)
    V[:, 1] = v0 / norm(v0)
    ksteps = m
    for k in 1:m
        w = ComplexF64.(Aop(V[:, k]))
        for i in 1:k
            H[i, k] = dot(V[:, i], w)
            w .-= H[i, k] .* V[:, i]
        end
        for i in 1:k                       # reorthogonalise once
            c = dot(V[:, i], w)
            H[i, k] += c
            w .-= c .* V[:, i]
        end
        H[k + 1, k] = norm(w)
        if abs(H[k + 1, k]) < 1e-12 * (abs(H[1, 1]) + 1)
            ksteps = k
            break
        end
        V[:, k + 1] = w / H[k + 1, k]
    end
    return V, H, ksteps
end

function ritz(H, k)
    F = eigen(H[1:k, 1:k])
    β = abs(H[k + 1, k])
    res = [β * abs(F.vectors[k, i]) for i in 1:k]
    p = sortperm(abs.(F.values), rev = true)
    return F.values[p], res[p]
end

function departure_from_normality(A)
    λ = eigvals(A)
    return sqrt(max(norm(A)^2 - sum(abs2, λ), 0.0))
end

# ============================================================
# Krylov study of one operator against a reference spectrum.
# ============================================================
function krylov_study(Aop, n, opname, λ_ref; m = 40, report_at = 5:5:40)
    t = @elapsed (V, H, ks) = arnoldi(Aop, n, m; v0 = randn(n))
    @printf("\n  [%s]  Arnoldi: %d steps (%.1fs)", opname, ks, t)
    ks < m && @printf("  (happy breakdown at k=%d)", ks)
    println()

    ref = sort(λ_ref, by = abs, rev = true)
    λ1_ref = ref[1]
    λ2_ref = length(ref) >= 2 ? ref[2] : ComplexF64(NaN)
    # subdominant references = everything except the λ₁ branch (handles conjugate
    # pairs: θ₂ may converge to λ₂ OR its conjugate — both are "the gap").
    subref = ref[2:end]
    nearest_sub(z) = isempty(subref) ? NaN : minimum(abs(z - λ) for λ in subref)

    @printf("    %-4s %-14s %-22s %-10s %-10s %-9s\n",
        "k", "|θ₁|→|λ₁|", "θ₂→λ₂ (the gap)", "res(θ₁)", "res(θ₂)", "dep(Hₖ)")
    conv = Float64[]
    ks_report = filter(k -> k <= ks, collect(report_at))
    for k in ks_report
        θ, res = ritz(H, k)
        dep = departure_from_normality(H[1:k, 1:k])
        e2 = length(θ) >= 2 ? nearest_sub(θ[2]) : NaN
        push!(conv, e2)
        @printf("    %-4d %-14s %-22s %-10.2e %-10.2e %-9.2e\n",
            k, @sprintf("%.6f", abs(θ[1])),
            length(θ) >= 2 ? @sprintf("%.4f%+.4fi", real(θ[2]), imag(θ[2])) : "—",
            res[1], length(res) >= 2 ? res[2] : NaN, dep)
    end
    @printf("    reference:  |λ₁| = %.6f,  λ₂ = %.4f%+.4fi  (gap 1−|λ₂| = %.4f)\n",
        abs(λ1_ref), real(λ2_ref), imag(λ2_ref), 1 - abs(λ2_ref))

    θfinal, _ = ritz(H, ks)
    return (; H, ks, θfinal, ks_report, conv, opname)
end

# ============================================================
# Per-map driver
# ============================================================
function run_map(F, domain, noise, label; target_ratio, periodic, enclosure_method,
        m = 30, mref = 40, subdiv = 2, max_depth = 12)
    # choose depth so σ/δ ≈ target_ratio:  σ/δ = σ·2^depth / W  ⇒  depth = log2(ratio·W/σ)
    W = diam(domain[1]); σ = diam(noise[1]) / 2
    depth = clamp(Int(round(log2(target_ratio * W / σ))), 5, max_depth)
    @printf("\n[%s] target σ/δ ≈ %d → depth %d (%d boxes)\n",
        label, target_ratio, depth, (2^depth)^2)
    op = build_operators(F, domain, noise, label;
        depth = depth, enclosure_method = enclosure_method,
        periodic = periodic, subdiv = subdiv)

    # Reference spectrum: a long matrix-free Arnoldi on the transfer operator.
    print("  Reference spectrum: matrix-free Arnoldi, m=$mref ... ")
    tr = @elapsed (_, Href, ksref) = arnoldi(op.transfer, op.n, mref; v0 = randn(op.n))
    θref, _ = ritz(Href, ksref)
    λ_ref = θref[1:min(10, length(θref))]
    @printf("ok (%d steps, %.1fs)\n", ksref, tr)

    st = krylov_study(op.transfer, op.n, "transfer", λ_ref; m = m, report_at = 5:5:m)
    sk = krylov_study(op.koopman,  op.n, "koopman",  λ_ref; m = m, report_at = 5:5:m)
    return (; op, λ_ref, transfer = st, koopman = sk, label)
end

# ============================================================
# Run both maps
# ============================================================
RATIO  = get(ENV, "RATIO", "")  == "" ? 128 : parse(Int, ENV["RATIO"])
MSTEPS = get(ENV, "MSTEPS", "") == "" ? 30  : parse(Int, ENV["MSTEPS"])
MREF   = get(ENV, "MREF", "")   == "" ? 40  : parse(Int, ENV["MREF"])
MAXD   = get(ENV, "MAXD", "")   == "" ? 12  : parse(Int, ENV["MAXD"])
SUBDIV = get(ENV, "SUBDIV", "") == "" ? 2   : parse(Int, ENV["SUBDIV"])

# Noise half-width σ = 0.1 (uniform_noise_box(0.1,2) ⇒ interval [-0.1,0.1]); each
# map's depth is auto-chosen so the cell is ≈ noise/σ_ratio (σ/δ ≈ RATIO).
NOISE = uniform_noise_box(0.1, 2)
domain_henon  = IntervalBox(interval(-1.0, 1.0), interval(-1.0, 1.0))
domain_lorenz = IntervalBox(interval(-0.5, 0.5), interval(-0.5, 0.5))

results = Any[]
push!(results, run_map(henon_rescaled, domain_henon, NOISE, "Hénon (periodic)";
    target_ratio = RATIO, periodic = true, enclosure_method = zeroth_order_enclosure,
    m = MSTEPS, mref = MREF, subdiv = SUBDIV, max_depth = MAXD))
push!(results, run_map(contracting_lorenz, domain_lorenz, NOISE, "Contracting Lorenz";
    target_ratio = RATIO, periodic = false, enclosure_method = zeroth_order_enclosure,
    m = MSTEPS, mref = MREF, subdiv = SUBDIV, max_depth = MAXD))

# ============================================================
# Visualisation: Ritz vs reference (complex plane) + λ₂ convergence
# ============================================================
println("\nCreating visualisation...")
fig = Figure(size = (1300, 900))
for (row, R) in enumerate(results)
    ax = Axis(fig[row, 1], title = "$(R.label): spectrum",
        xlabel = "Re", ylabel = "Im", aspect = DataAspect())
    θu = Float64.(range(0, 2π, length = 200))
    lines!(ax, cos.(θu), sin.(θu), color = (:gray, 0.4))
    scatter!(ax, real.(R.λ_ref), imag.(R.λ_ref), color = :black,
        marker = :circle, markersize = 14, label = "ref (long Arnoldi)")
    scatter!(ax, real.(R.transfer.θfinal), imag.(R.transfer.θfinal),
        color = :dodgerblue, marker = :xcross, markersize = 11, label = "transfer Ritz")
    scatter!(ax, real.(R.koopman.θfinal), imag.(R.koopman.θfinal),
        color = :crimson, marker = :cross, markersize = 11, label = "koopman Ritz")
    axislegend(ax, position = :lb, framevisible = false)

    ax2 = Axis(fig[row, 2], title = "$(R.label): |θ₂ − λ₂| vs Arnoldi step",
        xlabel = "Arnoldi step k", ylabel = "error", yscale = log10)
    if !isempty(R.transfer.conv)
        scatterlines!(ax2, R.transfer.ks_report, max.(R.transfer.conv, 1e-16),
            color = :dodgerblue, label = "transfer")
    end
    if !isempty(R.koopman.conv)
        scatterlines!(ax2, R.koopman.ks_report, max.(R.koopman.conv, 1e-16),
            color = :crimson, label = "koopman")
    end
    axislegend(ax2, position = :rt, framevisible = false)
end
out = joinpath(@__DIR__, "09_krylov_matvec_output.png")
save(out, fig)
println("Saved ", out)

# ============================================================
# Summary
# ============================================================
println("\n" * "="^70)
println("SUMMARY (matrix-free Krylov, σ/δ ≈ $RATIO, m = $MSTEPS)")
println("="^70)
@printf("%-24s %-7s %-9s %-12s %-12s\n",
    "map / operator", "steps", "dep(Hₘ)", "|θ₁−λ₁|", "gap err θ₂")
println("-"^70)
for R in results
    λ1 = abs(sort(R.λ_ref, by = abs, rev = true)[1])
    for s in (R.transfer, R.koopman)
        dep = departure_from_normality(s.H[1:s.ks, 1:s.ks])
        e1 = abs(abs(s.θfinal[1]) - λ1)
        e2 = isempty(s.conv) ? NaN : s.conv[end]
        @printf("%-24s %-7d %-9.2e %-12.2e %-12.2e\n",
            "$(R.label) / $(s.opname)", s.ks, dep, e1, e2)
    end
end
println("\nNote: transfer and Koopman share the spectrum but the operator is non-normal")
println("(dep(Hₘ) ≫ 0) ⇒ Ritz convergence is governed by pseudospectra, not eigenvalues;")
println("the two adjoint Krylov runs need not converge at the same rate.")
