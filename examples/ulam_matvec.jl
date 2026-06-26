# Matrix-free Ulam transfer operator (forward-image scatter + box-noise) and the certified
# matvec Lanczos far-block floor — shared by the contraction certificate (example 11).
# Everything is MATVEC ONLY: L is never assembled; it is applied by recomputing each cell's
# image. The matvec is generic in the vector precision (Float64 or BigFloat), so the far-block
# Lanczos can run in BigFloat where the Float64 orthogonality floor √u is too coarse.

using RigorousUlam2D, IntervalArithmetic, IntervalBoxes, LinearAlgebra, SparseArrays, Printf

# ---------------------------------------------------------------- maps
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

@inline function _fold(ix, iy, nx, ny, periodic)
    periodic ? (mod1(ix, nx), mod1(iy, ny)) :
    ((1 <= ix <= nx && 1 <= iy <= ny) ? (ix, iy) : (0, 0))
end

# raw transition weights from source cell j (one column of P_det), recomputed on the fly.
function _column_targets(ctx, j)
    (; F, partition, g2b, x0, y0, dx, dy, nx, ny, subdiv, enclosure_method, periodic) = ctx
    leaves = [partition.boxes[j]]
    for _ in 1:subdiv
        nl = eltype(leaves)[]
        for b in leaves
            l, r = bisect_box(b); push!(nl, l, r)
        end
        leaves = nl
    end
    wleaf = 1.0 / length(leaves); acc = Dict{Int, Float64}()
    for s in leaves
        img = enclosure_method(F, s)
        ib = img isa Tuple ? IntervalBox(img...) : img
        xa, xb = inf(ib[1]), sup(ib[1]); ya, yb = inf(ib[2]), sup(ib[2])
        area = (xb - xa) * (yb - ya)
        ixlo = Int(floor((xa - x0) / dx)) + 1; ixhi = Int(floor((xb - x0) / dx)) + 1
        iylo = Int(floor((ya - y0) / dy)) + 1; iyhi = Int(floor((yb - y0) / dy)) + 1
        if area <= 0 || (ixhi - ixlo) > nx || (iyhi - iylo) > ny
            cx = (xa + xb) / 2; cy = (ya + yb) / 2
            ix = Int(floor((cx - x0) / dx)) + 1; iy = Int(floor((cy - y0) / dy)) + 1
            ixw, iyw = _fold(ix, iy, nx, ny, periodic); ixw == 0 && continue
            k = g2b[iyw, ixw]; k != 0 && (acc[k] = get(acc, k, 0.0) + wleaf); continue
        end
        for ix in ixlo:ixhi
            cxa = x0 + (ix - 1) * dx; ox = max(0.0, min(xb, cxa + dx) - max(xa, cxa)); ox <= 0 && continue
            for iy in iylo:iyhi
                cya = y0 + (iy - 1) * dy; oy = max(0.0, min(yb, cya + dy) - max(ya, cya)); oy <= 0 && continue
                ixw, iyw = _fold(ix, iy, nx, ny, periodic); ixw == 0 && continue
                k = g2b[iyw, ixw]; k == 0 && continue
                acc[k] = get(acc, k, 0.0) + wleaf * (ox * oy) / area
            end
        end
    end
    return acc
end

function _colsum(ctx)
    n = ctx.partition.n_boxes; cs = zeros(n)
    Threads.@threads :static for j in 1:n
        cs[j] = sum(values(_column_targets(ctx, j)); init = 0.0)
    end
    return cs
end

# P_det · v  by forward SCATTER (generic precision; zeros skip the image recompute).
# THREADED over cells. Callers must NOT wrap this in their own @threads (no nesting) — the
# sample/cluster loops that use it are kept serial.
function det_transfer(ctx, v::AbstractVector{T}) where {T}
    n = ctx.partition.n_boxes; nt = Threads.maxthreadid()
    buf = [zeros(T, n) for _ in 1:nt]
    Threads.@threads :static for j in 1:n
        cj = ctx.colsum[j]; cj <= 0 && continue
        cw = v[j] / cj; iszero(cw) && continue
        b = buf[Threads.threadid()]
        for (k, w) in _column_targets(ctx, j)
            b[k] += T(w) * cw
        end
    end
    out = copy(buf[1])
    for t in 2:nt
        out .+= buf[t]
    end
    return out
end

# P_detᵀ · u  by GATHER (the transpose).
function det_koopman(ctx, u::AbstractVector{T}) where {T}
    n = ctx.partition.n_boxes; out = zeros(T, n)
    Threads.@threads :static for j in 1:n
        cj = ctx.colsum[j]; cj <= 0 && continue
        s = zero(T)
        for (k, w) in _column_targets(ctx, j)
            s += T(w) * u[k]
        end
        out[j] = s / cj
    end
    return out
end

# O(n) separable box-filter noise (generic precision), periodic or absorbing.
function box_noise(v::AbstractVector{T}, rx, ry, nx, ny, b2g; periodic) where {T}
    n = nx * ny; w = one(real(T)) / ((2rx + 1) * (2ry + 1))
    V = zeros(T, ny, nx)
    for k in 1:n
        ix, iy = b2g[k]; V[iy, ix] = v[k]
    end
    Sx = zeros(T, ny, nx)
    if periodic
        for iy in 1:ny
            s = zero(T)
            for d in -rx:rx
                s += V[iy, mod1(1 + d, nx)]
            end
            Sx[iy, 1] = s
            for ix in 2:nx
                s += V[iy, mod1(ix + rx, nx)] - V[iy, mod1(ix - 1 - rx, nx)]; Sx[iy, ix] = s
            end
        end
    else
        P = zeros(T, nx + 1)
        for iy in 1:ny
            for ix in 1:nx
                P[ix + 1] = P[ix] + V[iy, ix]
            end
            for ix in 1:nx
                Sx[iy, ix] = P[min(nx, ix + rx) + 1] - P[max(1, ix - rx)]
            end
        end
    end
    W = zeros(T, ny, nx)
    if periodic
        for ix in 1:nx
            s = zero(T)
            for d in -ry:ry
                s += Sx[mod1(1 + d, ny), ix]
            end
            W[1, ix] = s
            for iy in 2:ny
                s += Sx[mod1(iy + ry, ny), ix] - Sx[mod1(iy - 1 - ry, ny), ix]; W[iy, ix] = s
            end
        end
    else
        P = zeros(T, ny + 1)
        for ix in 1:nx
            for iy in 1:ny
                P[iy + 1] = P[iy] + Sx[iy, ix]
            end
            for iy in 1:ny
                W[iy, ix] = P[min(ny, iy + ry) + 1] - P[max(1, iy - ry)]
            end
        end
    end
    res = zeros(T, n)
    for k in 1:n
        ix, iy = b2g[k]; res[k] = w * W[iy, ix]
    end
    return res
end

# Build the matrix-free transfer L=N·P_det and adjoint Lᵀ=P_detᵀ·N as matvec closures.
function build_matvec(F, domain, noise, depth; periodic, enclosure_method, subdiv = 4)
    partition = build_uniform_partition(domain, depth)
    n = partition.n_boxes; nx = ny = 2^depth
    b2g, g2b = build_grid_index_maps(partition, (nx, ny))
    σx = diam(noise[1]) / 2; σy = diam(noise[2]) / 2
    dx = diam(domain[1]) / nx; dy = diam(domain[2]) / ny
    rx = max(1, Int(round(σx / dx))); ry = max(1, Int(round(σy / dy)))
    ctx = (; F, partition, g2b, b2g, x0 = inf(domain[1]), y0 = inf(domain[2]),
        dx, dy, nx, ny, subdiv, enclosure_method, periodic, colsum = zeros(n))
    ctx = merge(ctx, (; colsum = _colsum(ctx)))
    Lmul(v) = box_noise(det_transfer(ctx, v), rx, ry, nx, ny, b2g; periodic = periodic)
    Ladj(u) = det_koopman(ctx, box_noise(u, rx, ry, nx, ny, b2g; periodic = periodic))
    return Lmul, Ladj, n, rx
end

# Cached matvec: precompute the sparse P_det ONCE (O(nnz) storage, NOT the composed L=N·P_det,
# which is never formed) and apply it as a fast spmv. Use for the certificate runs (moderate n,
# many Lanczos steps); the recompute `build_matvec` stays the scaling path. P_det promotes to the
# vector's precision (BigFloat ok).
function build_matvec_cached(F, domain, noise, depth; periodic, enclosure_method, subdiv = 4)
    partition = build_uniform_partition(domain, depth)
    n = partition.n_boxes; nx = ny = 2^depth
    b2g, g2b = build_grid_index_maps(partition, (nx, ny))
    Pd = build_transfer_forward(F, partition, (nx, ny);
        subdiv = subdiv, enclosure_method = enclosure_method, periodic = periodic)
    normalize_columns!(Pd)
    σx = diam(noise[1]) / 2; σy = diam(noise[2]) / 2
    rx = max(1, Int(round(σx / (diam(domain[1]) / nx)))); ry = max(1, Int(round(σy / (diam(domain[2]) / ny))))
    Pmul(v) = eltype(v) <: Union{Float64, ComplexF64} ? Pd * v : SparseMatrixCSC{eltype(v)}(Pd) * v
    Padj(u) = eltype(u) <: Union{Float64, ComplexF64} ? Pd' * u : SparseMatrixCSC{eltype(u)}(Pd') * u
    Lmul(v) = box_noise(Pmul(v), rx, ry, nx, ny, b2g; periodic = periodic)
    Ladj(u) = Padj(box_noise(u, rx, ry, nx, ny, b2g; periodic = periodic))
    return Lmul, Ladj, n, rx
end

# ---------------------------------------------------------------- Arnoldi (matvec) → H_N
# returns the N×N Hessenberg H, the Krylov basis V (n×N), and the Arnoldi residual β_N.
function arnoldi_block(Lmul, n, N; v0 = randn(ComplexF64, n))
    V = zeros(ComplexF64, n, N + 1); H = zeros(ComplexF64, N + 1, N)
    V[:, 1] = v0 ./ norm(v0)
    for k in 1:N
        w = ComplexF64.(Lmul(V[:, k]))
        for _ in 1:2, i in 1:k
            c = dot(V[:, i], w); H[i, k] += c; w .-= c .* V[:, i]
        end
        H[k + 1, k] = norm(w)
        abs(H[k + 1, k]) < 1e-13 && return H[1:k, 1:k], V[:, 1:k], abs(H[k + 1, k]), k
        V[:, k + 1] = w ./ H[k + 1, k]
    end
    return H[1:N, 1:N], V[:, 1:N], abs(H[N + 1, N]), N
end

# ---------------------------------------------------------------- certified far-block Lanczos
# NOT duplicated here: the selective-reorth far-block Lanczos, the geometric-mean R₁ floor, and
# the Krylov–Rump–Miyajima decoupling now live (generic-precision) in RigorousPseudospectra —
#   far_block_sigma_min(Amul, Aadj, n, W, z; prec)   # σ_min(G₂₂−zI) by one matvec Lanczos
#   certify_gap_on_circle(Amul, Aadj, n, W, c, ρ; prec, auto_radius)
#   certify_leading_eigs(Amul, Aadj, n; nblock, neig, refine, prec)
# The examples pass this file's matvec operators (Lmul/Ladj) straight to those.

# thin wrapper (NOT a copy of the Lanczos): min over |z|=r of σ_min(G₂₂−zI) via the reference
# far-block certificate, + 1-Lipschitz arc-cover. Used by the contraction estimate (example 11).
function far_min_on_circle(Lmul, Ladj, W, r, n, ::Type{T}; nθ = 16) where {T}
    s = [RigorousPseudospectra.far_block_sigma_min(Lmul, Ladj, n, W, r * cis(2π * t / nθ);
             prec = T)[1] for t in 0:(nθ - 1)]
    return minimum(s) - 2r * sin(π / (2nθ)), minimum(s)
end
