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
using Serialization
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
# Per-thread accumulators are preallocated once in `ctx.scratch` and reused (zeroed)
# every call — avoids allocating/GC-ing nt·n complex buffers per matvec (multi-GB
# churn at n ~ 1e7). Zero entries of v skip the (expensive) image recomputation.
function det_transfer(ctx, v::AbstractVector{T}) where {T}
    n = ctx.partition.n_boxes
    buf = ctx.scratch
    for b in buf
        fill!(b, zero(T))
    end
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
    out = copy(buf[1])                  # single allocation for the result
    @inbounds for t in 2:length(buf)
        @. out += buf[t]                # in-place reduction (no per-step alloc)
    end
    return out
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

    nt = Threads.maxthreadid()
    ctx = (; F, partition, g2b, b2g,
        x0 = inf(domain[1]), y0 = inf(domain[2]),
        dx = diam(domain[1]) / nx, dy = diam(domain[2]) / ny,
        nx, ny, subdiv, enclosure_method, periodic,
        colsum = zeros(n),
        scratch = [Vector{ComplexF64}(undef, n) for _ in 1:nt])   # reused scatter buffers

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
# Matrix-free resolvent certificate on a circle around λ₁ (draft §; ported from
# RigorousPseudospectra bench/ulam_circle_near1.jl). Numerical (Float64) — the
# harness, not the rigorous arithmetic. Needs BOTH the transfer P and its adjoint
# P' (Koopman) as matvecs:
#   capture λ₁ by rank-1 deflation  P22 = I − w₁w₁*  (w₁ = stationary measure),
#   far block G22 = P22 P P22 applied matvec-only, σ_min(G22−zI) by Lanczos on
#   M_z = P22 (G22−zI)*(G22−zI) P22, then assemble
#       σ_min(P−zI) ≥ σ₁₁σ₂₂/(σ₁₁+σ₂₂+γ),  σ₁₁=|G11−z|, γ = capture coupling,
#   and cover the whole circle by Lipschitz arc-covering (σ_min is 1-Lipschitz).
# A positive whole-circle floor ⇒ exactly one eigenvalue (λ₁) inside ⇒ gap ≥ ρ.

# STORAGE-FREE Lanczos on the Hermitian M_z (matvec-only): keeps only q/qprev (O(n)),
# NO basis stored. Streams the tridiagonal (α,β), the realized local rounding defect
# norms ‖f_k‖, and the adjacent Gram seeds |⟨q_k,q_{k+1}⟩| — all the certifier needs.
function lanczos_streamed(Mz, N, kmax; v0 = nothing)
    α = zeros(Float64, kmax); β = zeros(Float64, kmax)
    fn = zeros(Float64, kmax); sdot = zeros(Float64, kmax)
    q = v0 === nothing ? randn(ComplexF64, N) : copy(v0); q ./= norm(q)
    qprev = zeros(ComplexF64, N); βprev = 0.0
    kact = kmax
    for k in 1:kmax
        w = Mz(q)
        a = real(dot(q, w)); α[k] = a
        r = w .- a .* q .- βprev .* qprev
        b = norm(r); β[k] = b
        (!isfinite(b) || b <= eps(Float64) * abs(a)) && (kact = k; break)
        qnext = r ./ b
        combo = a .* q .+ βprev .* qprev .+ b .* qnext
        fn[k] = norm(combo .- w)            # realized local defect ‖f_k‖ (storage-free)
        sdot[k] = abs(dot(q, qnext))        # adjacent Gram seed |⟨q_k,q_{k+1}⟩|
        qprev, βprev, q = q, b, qnext
    end
    return α[1:kact], β[1:kact], fn[1:kact], sdot[1:kact], kact
end

# Streamed Paige bound on ‖R₂(k)‖ = ‖Q*Q − I‖₂ (draft eq. paige): propagate W ≥ |ω_ij|
# one anti-diagonal at a time from the scalars only — no stored basis, no inner products.
function paige_R2(α, β, fn, sdot, k)
    W = zeros(Float64, k, k)
    for i in 1:(k - 1)
        W[i, i + 1] = W[i + 1, i] = sdot[i]           # seed: |⟨q_i,q_{i+1}⟩|  (W[i,i]=|‖q‖²−1|≈0)
    end
    ξ(i, j) = fn[i] + fn[j]                            # ≥ |ξ_ij| (‖q‖=1, Cauchy–Schwarz)
    for j in 2:(k - 1)
        for i in (j - 1):-1:1
            num = β[i] * W[i + 1, j] + abs(α[i] - α[j]) * W[i, j] +
                  (i > 1 ? β[i - 1] * W[i - 1, j] : 0.0) + β[j - 1] * W[i, j - 1] + ξ(i, j)
            W[i, j + 1] = W[j + 1, i] = num / β[j]
        end
    end
    return maximum(sum(@view W[i, 1:k]) for i in 1:k)  # ‖W‖₁ = ‖W‖∞ ≥ ‖R₂‖₂
end

# far floor σ_min(G22−zI) = √(max(0, θ_min − β₂₂)) from the tridiagonal + streamed defects
# (draft Prop. matvec-floor). θ_min = λ_min(T_k); β₂₂ = ‖R₁‖/(1−‖R₂‖); ‖R₂‖ from Paige.
function floor_streamed(α, β, fn, sdot, k)
    Tk = SymTridiagonal(α[1:k], β[1:(k - 1)])
    E = eigen(Tk); θmin = minimum(E.values); S = E.vectors
    rho = paige_R2(α, β, fn, sdot, k)
    lastcomp = β[k] * abs.(S[k, :])                          # β_k (s_i)_k  (Ritz residual tail)
    fcontrib = [sum(abs.(S[1:k, i]) .* fn[1:k]) for i in 1:k] # accumulated local defects F_k s_i
    ρ1 = sqrt(sum((lastcomp .+ fcontrib) .^ 2))              # ≥ ‖R₁‖₂ (column Frobenius)
    β22 = rho < 1 ? ρ1 / (1 - rho) : Inf
    return sqrt(max(0.0, θmin - β22)), θmin, ρ1, rho
end

# Certify σ_min(P−zI) over the circle Γ = { c + ρ·e^{iθ} } via matvecs P, Pt=P'.
function resolvent_circle(P, Pt, n, w1, c, ρ; nθ = 16, kmax = 80, Pmat = nothing,
        nullshift = 10.0)
    w1 = w1 / norm(w1)
    P22(x) = x .- w1 .* (w1' * x)
    G11 = w1' * P(w1)                                   # ≈ λ₁
    γ = max(norm(P22(P(w1))), norm(P22(Pt(w1))))        # capture coupling (rank-1 ⇒ ‖·‖)
    @printf("    G11 = w₁*Pw₁ = %.6f%+.6fi   coupling γ = %.3e\n", real(G11), imag(G11), γ)
    floors = Float64[]
    for t in 0:(nθ - 1)
        z = c + ρ * cis(2π * t / nθ)
        # M_z = P22 (G22−zI)*(G22−zI) P22, PLUS a shift `nullshift` on the deflated
        # direction w₁ (M_z's structural kernel). The shift moves w₁'s eigenvalue from 0
        # to `nullshift` (above ‖M_z‖) so leaked-w₁ rounding can't masquerade as σ_min=0.
        Mz(x) = (v1 = P22(x); v2 = P(v1) .- z .* v1; v3 = P22(v2);
        v4 = Pt(v3) .- conj(z) .* v3;
        P22(v4) .+ nullshift .* (w1 .* (w1' * x)))
        # STORAGE-FREE: stream (α,β,‖f‖,sdot); bound ‖R₂‖ and the floor from scalars only.
        α, β, fn, sdot, k = lanczos_streamed(Mz, n, kmax; v0 = P22(randn(ComplexF64, n)))
        fl, θraw, ρ1, r2 = floor_streamed(α, β, fn, sdot, k)
        σ11 = abs(G11 - z)
        σA = (σ11 <= 0 || fl <= 0 || r2 >= 1) ? 0.0 : σ11 * fl / (σ11 + fl + γ)
        push!(floors, σA)
        tru = Pmat === nothing ? NaN : minimum(svdvals(Pmat - z * I))
        # clean true far σ_min (small-n validation): P22(P−zI)P22 has one structural zero
        # (the w₁ direction); the 2nd-smallest singular value is σ_min(M22−zI) on ran P22.
        trufar = NaN
        if Pmat !== nothing
            P22m = I - w1 * w1'
            trufar = sort(svdvals(P22m * (Pmat - z * I) * P22m))[2]
        end
        ok = isnan(tru) ? "" : (σA <= tru + 1e-9 ? "sound" : "VIOLATION")
        @printf("    θ=%4.2fπ σ11=%.3e farσ22=%.3e [√θmin=%.3e ‖R₂‖=%.1e | TRUEfar=%.3e] → σ_min≥%.3e true=%.3e %s\n",
            2t / nθ, σ11, fl, sqrt(max(θraw, 0.0)), r2, trufar, σA, tru, ok)
    end
    cover = 2ρ * sin(π / (2nθ))
    circ = minimum(floors) - cover
    @printf("    min over %d samples = %.4e ; Lipschitz cover = %.4e ; whole-circle floor = %.4e\n",
        nθ, minimum(floors), cover, circ)
    return (; floors, circ, ρ, c, γ, G11)
end

# ============================================================
# Krylov study of one operator against a reference spectrum.
# ============================================================
function krylov_study(Aop, n, opname, λ_ref; m = 40, report_at = 5:5:40,
        tag = "op", ksmall = 8)
    t = @elapsed (V, H, ks) = arnoldi(Aop, n, m; v0 = randn(n))
    @printf("\n  [%s]  Arnoldi: %d steps (%.1fs)", opname, ks, t)
    ks < m && @printf("  (happy breakdown at k=%d)", ks)
    println()

    ref = sort(λ_ref, by = abs, rev = true)
    have_ref = !isempty(ref)
    λ1_ref = have_ref ? ref[1] : ComplexF64(NaN)
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
        have_ref && push!(conv, e2)
        @printf("    %-4d %-14s %-22s %-10.2e %-10.2e %-9.2e\n",
            k, @sprintf("%.6f", abs(θ[1])),
            length(θ) >= 2 ? @sprintf("%.4f%+.4fi", real(θ[2]), imag(θ[2])) : "—",
            res[1], length(res) >= 2 ? res[2] : NaN, dep)
    end
    if have_ref
        @printf("    reference:  |λ₁| = %.6f,  λ₂ = %.4f%+.4fi  (gap 1−|λ₂| = %.4f)\n",
            abs(λ1_ref), real(λ2_ref), imag(λ2_ref), 1 - abs(λ2_ref))
    end

    θfinal, _ = ritz(H, ks)

    # --- save the small-part Ritz data: the candidate UPPER-LEFT BLOCK for the
    # certified Krylov–Rump–Miyajima step (draft §4). After ks Arnoldi steps,
    #     A·V_ks = V_ks·H_ks + v_{ks+1}·(h_{ks+1,ks} eₖᵀ),                  (Arnoldi)
    # so the Rayleigh quotient H_ks (ks×ks) is the upper-left block whose
    # eigenvalues are the Ritz values. For the ksmall dominant Ritz pairs
    # (Y the eigenvectors of H_ks) the Ritz vectors are X = V_ks·Y, and the
    # exact (numerical) residual of the invariant-subspace relation is
    #     A·X − X·diag(Θ) = v_{ks+1} · rrow,   rrow = h_{ks+1,ks}·Y[ks,·],
    # i.e. rank-1 in the single Krylov direction v_{ks+1} (‖v_{ks+1}‖=1).
    # The certifier (BallArithmetic) re-derives R₂/β/κ on H_ks and inflates the
    # Gershgorin discs by ‖rrow‖ to enclose σ(A) near this subspace.
    Fh = eigen(H[1:ks, 1:ks])
    ord = sortperm(abs.(Fh.values), rev = true)
    ksm = min(ksmall, ks)
    sel = ord[1:ksm]
    Hks = H[1:ks, 1:ks]
    Ysel = Fh.vectors[:, sel]                 # ks × ksm  (selected Ritz eigvectors of H_ks)
    Θsel = Fh.values[sel]                     # ksm Ritz values (the small/dominant part)
    Xsmall = V[:, 1:ks] * Ysel                # n × ksm   Ritz vectors (approx. invariant subspace)
    rrow = H[ks + 1, ks] .* Ysel[ks, :]       # ksm residual coupling scalars
    vlast = V[:, ks + 1]                      # residual direction (‖·‖ = 1)
    XtX = Xsmall' * Xsmall                    # ksm × ksm Gram (subspace conditioning, κ input)
    savepath = joinpath(@__DIR__, "krylov_block_$(tag).jls")
    serialize(savepath,
        (; tag, opname, ks, ksm, Hks, Ysel, Θsel, sel, Xsmall, rrow, vlast, XtX,
            ritz_res = abs.(rrow)))
    @printf("    saved upper-left block (%d×%d) + %d Ritz vectors → %s\n",
        ks, ks, ksm, basename(savepath))

    # dominant Ritz vector (largest modulus): for the transfer operator this is the
    # stationary measure (Perron eigenvector, λ₁≈1).
    dom = Xsmall[:, 1]
    domλ = Θsel[1]
    return (; H, ks, θfinal, ks_report, conv, opname, savepath, ksm, dom, domλ)
end

# ============================================================
# Per-map driver
# ============================================================
function run_map(F, domain, noise, label; target_ratio, periodic, enclosure_method,
        m = 30, subdiv = 2, max_depth = 12, frac = 0.5, nθ = 16, kmax = 80)
    # choose depth so σ/δ ≈ target_ratio:  σ/δ = σ·2^depth / W  ⇒  depth = log2(ratio·W/σ)
    W = diam(domain[1]); σ = diam(noise[1]) / 2
    depth = clamp(Int(round(log2(target_ratio * W / σ))), 5, max_depth)
    @printf("\n[%s] target σ/δ ≈ %d → depth %d (%d boxes)\n",
        label, target_ratio, depth, (2^depth)^2)
    op = build_operators(F, domain, noise, label;
        depth = depth, enclosure_method = enclosure_method,
        periodic = periodic, subdiv = subdiv)

    # Arnoldi on the TRANSFER operator only → stationary measure w₁ (the rank-1
    # capture for the resolvent certificate) + the dominant cluster / upper-left
    # block. The Koopman operator is NOT Arnoldi'd — it enters only as the adjoint
    # matvec P' inside the resolvent certificate below.
    stem = replace(lowercase(label), r"[^a-z0-9]+" => "_")
    st = krylov_study(op.transfer, op.n, "transfer", ComplexF64[];
        m = m, report_at = 5:5:m, tag = "$(stem)_transfer")
    w1 = st.dom                       # stationary measure (λ₁ Ritz vector)
    c = st.domλ                       # leading eigenvalue λ₁ (≈1)
    # admissible radius from the NEAREST other Ritz value (so Γ encloses only λ₁)
    others = st.θfinal[2:end]
    gap = isempty(others) ? 0.0 : minimum(abs(c - θ) for θ in others)
    ρ = frac * gap
    @printf("  resolvent circle: centre λ₁ = %.5f%+.5fi, dist to λ₂ = %.5f, ρ = %.2f·dist = %.5f\n",
        real(c), imag(c), gap, frac, ρ)

    # optional true-σ_min check by materialising P via matvecs (small grids only)
    Pmat = nothing
    if op.n <= 1024
        print("  materialising P (n≤1024) for true-σ_min check via $(op.n) matvecs... ")
        Pmat = reduce(hcat, (op.transfer(ComplexF64.(1:op.n .== j)) for j in 1:op.n))
        println("done")
    end

    println("  Certifying resolvent on circle around λ₁ (matvec P and adjoint P')...")
    cert = resolvent_circle(op.transfer, op.koopman, op.n, w1, c, ρ;
        nθ = nθ, kmax = kmax, Pmat = Pmat)
    if cert.circ > 0
        @printf("  ✓ σ_min(P−zI) ≥ %.4e on ALL of Γ ⇒ resolvent ≤ %.3e ⇒ gap ≥ ρ = %.5f\n",
            cert.circ, 1 / cert.circ, ρ)
    else
        @printf("  ✗ whole-circle floor ≤ 0 (need larger nθ / smaller ρ / more Lanczos steps)\n")
    end
    return (; op, transfer = st, w1, c, ρ, cert, label)
end

# ============================================================
# Run both maps
# ============================================================
RATIO  = get(ENV, "RATIO", "")  == "" ? 128 : parse(Int, ENV["RATIO"])
MSTEPS = get(ENV, "MSTEPS", "") == "" ? 30  : parse(Int, ENV["MSTEPS"])
MAXD   = get(ENV, "MAXD", "")   == "" ? 12  : parse(Int, ENV["MAXD"])
SUBDIV = get(ENV, "SUBDIV", "") == "" ? 2   : parse(Int, ENV["SUBDIV"])
FRAC   = get(ENV, "FRAC", "")   == "" ? 0.25 : parse(Float64, ENV["FRAC"]) # circle radius / gap
NTHETA = get(ENV, "NTHETA", "") == "" ? 48   : parse(Int, ENV["NTHETA"])   # circle samples
KMAX   = get(ENV, "KMAX", "")   == "" ? 60   : parse(Int, ENV["KMAX"])     # Lanczos steps

# Noise half-width σ = 0.1 (uniform_noise_box(0.1,2) ⇒ interval [-0.1,0.1]); each
# map's depth is auto-chosen so the cell is ≈ noise/σ_ratio (σ/δ ≈ RATIO).
NOISE = uniform_noise_box(0.1, 2)
domain_henon  = IntervalBox(interval(-1.0, 1.0), interval(-1.0, 1.0))
domain_lorenz = IntervalBox(interval(-0.5, 0.5), interval(-0.5, 0.5))

results = Any[]
push!(results, run_map(henon_rescaled, domain_henon, NOISE, "Hénon (periodic)";
    target_ratio = RATIO, periodic = true, enclosure_method = zeroth_order_enclosure,
    m = MSTEPS, subdiv = SUBDIV, max_depth = MAXD, frac = FRAC, nθ = NTHETA, kmax = KMAX))
push!(results, run_map(contracting_lorenz, domain_lorenz, NOISE, "Contracting Lorenz";
    target_ratio = RATIO, periodic = false, enclosure_method = zeroth_order_enclosure,
    m = MSTEPS, subdiv = SUBDIV, max_depth = MAXD, frac = FRAC, nθ = NTHETA, kmax = KMAX))

# ============================================================
# Visualisation: stationary measure (transfer λ₁ Ritz vector) + spectrum
# ============================================================
println("\nCreating visualisation...")

# Reshape a box-ordered vector onto the (ny × nx) grid for plotting.
function to_grid(v, op)
    D = fill(NaN, op.ny, op.nx)
    for k in 1:op.n
        ix, iy = op.ctx.b2g[k]
        D[iy, ix] = v[k]
    end
    return D
end

fig = Figure(size = (1250, 520 * length(results)))
for (row, R) in enumerate(results)
    op = R.op
    # invariant density = |dominant transfer Ritz vector| (Perron eigenvector), normalised
    μ = abs.(R.transfer.dom)
    μ ./= sum(μ)
    D = to_grid(μ, op)
    dom = op.partition.domain
    xs = range(inf(dom[1]), sup(dom[1]); length = op.nx)
    ys = range(inf(dom[2]), sup(dom[2]); length = op.ny)

    ax = Axis(fig[row, 1], aspect = DataAspect(),
        title = "$(R.label): stationary measure  (λ₁ = $(round(real(R.transfer.domλ), digits=4)), $(op.n) cells)",
        xlabel = "x", ylabel = "y")
    hm = heatmap!(ax, xs, ys, permutedims(D); colormap = :viridis)
    Colorbar(fig[row, 2], hm, label = "μ (mass / cell)")

    # spectrum panel: transfer Ritz values + the certified circle Γ around λ₁
    ax2 = Axis(fig[row, 3], title = "$(R.label): spectrum + certified circle Γ",
        xlabel = "Re", ylabel = "Im", aspect = DataAspect())
    θu = Float64.(range(0, 2π, length = 200))
    lines!(ax2, cos.(θu), sin.(θu), color = (:gray, 0.4))                  # unit circle
    scatter!(ax2, real.(R.transfer.θfinal), imag.(R.transfer.θfinal),
        color = :dodgerblue, marker = :xcross, markersize = 11, label = "transfer Ritz")
    Γc = R.c
    lines!(ax2, real(Γc) .+ R.ρ .* cos.(θu), imag(Γc) .+ R.ρ .* sin.(θu),
        color = R.cert.circ > 0 ? :seagreen : :red, linewidth = 2,
        label = R.cert.circ > 0 ? "Γ (gap ≥ ρ ✓)" : "Γ (not certified)")
    axislegend(ax2, position = :lb, framevisible = false)
end
out = joinpath(@__DIR__, "09_krylov_matvec_output.png")
save(out, fig)
println("Saved ", out)

# ============================================================
# Summary
# ============================================================
println("\n" * "="^70)
println("SUMMARY (matrix-free resolvent certificate around λ₁, σ/δ ≈ $RATIO)")
println("="^70)
@printf("%-22s %-10s %-9s %-10s %-12s %-10s\n",
    "map", "λ₁", "ρ (radius)", "Γ floor", "resolvent≤", "gap ≥ ρ?")
println("-"^74)
for R in results
    c = R.cert
    res = c.circ > 0 ? @sprintf("%.3e", 1 / c.circ) : "∞"
    ok = c.circ > 0 ? "✓ $(round(R.ρ, sigdigits=4))" : "✗"
    @printf("%-22s %-10.5f %-9.5f %-10.3e %-12s %-10s\n",
        R.label, real(R.c), R.ρ, c.circ, res, ok)
end
println("\nThe resolvent certificate uses BOTH matvecs: P (transfer) and P' (Koopman,")
println("the adjoint) inside M_z = P22(G22−zI)*(G22−zI)P22. A positive whole-circle")
println("floor ⇒ exactly one eigenvalue (λ₁) inside Γ ⇒ certified spectral gap ≥ ρ.")
println("The saved upper-left blocks (krylov_block_*.jls) feed the rigorous step.")
