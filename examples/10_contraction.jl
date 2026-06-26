# Contraction estimate:  find N with  ‖Lⁿ|_{U₀}‖ ≤ α < 1   (U₀ = mean-zero subspace).
#
# Method (slow-block + fast far-block; see RigorousPseudospectra src/krylov_resolvent.jl):
#   • L is the noisy Ulam transfer operator (Perron–Frobenius), λ₁ = 1 with stationary
#     density μ; U₀ = {v : ⟨𝟙,v⟩ = 0} is the L-invariant mean-zero subspace.
#   • Capture the m SLOW modes (λ₁=1, λ₂, …, λ_m) in an orthonormal frame W. Deflating μ
#     gives the small slow block S = (W-compression)|_{⊥μ}; ‖Sⁿ‖ is an exact (m−1)×(m−1)
#     matrix power — the slow, possibly non-normal transient. (This small SVD/‖·‖ is where
#     BallArithmetic would certify rigorously; here it is Float64.)
#   • The fast FAR block G₂₂ = P_m L P_m (P_m = I − WW*) has spectral radius ≈ |λ_{m+1}|.
#     A matvec selective-reorth Lanczos certifies σ_min(G₂₂ − zI) ≥ s_far on a circle
#     |z| = r_far (|λ_{m+1}| < r_far < |λ_m|), and σ_min is z-independent up to translation
#     (1-Lipschitz), so one run per sample + arc-cover gives min over the circle. The Cauchy
#     integral then bounds ‖G₂₂ⁿ‖₂ ≤ r_far^{n+1} / s_far.
#   • Block-triangular assembly (W ≈ invariant ⇒ G₂₁ ≈ 0):
#       ‖Lⁿ|_{U₀}‖ ≤ ‖Sⁿ‖ + ‖G₂₂ⁿ‖ + ‖B‖·Σ_{k} ‖Sᵏ‖·‖G₂₂^{n−1−k}‖,  B = slow→far coupling.
#     Smallest N with the RHS ≤ α is reported. (ℓ² norms; L¹ ≤ √dim · ℓ² — the √dim shifts N
#     by ~log√dim/|log ρ|, the rate ρ = max(|λ₂|, r_far) is unchanged.)
#
#   julia --project=examples --threads=auto examples/10_contraction.jl

import Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using RigorousUlam2D
using IntervalArithmetic, IntervalBoxes
using LinearAlgebra, SparseArrays
using Arpack
using Printf
using Random

Random.seed!(1234)
println("Julia threads: ", Threads.nthreads()); flush(stdout)

# ============================================================ maps
function henon_rescaled(x, y; a = 1.4, b = 0.3)
    x_phys = 1.5 * x; y_phys = 0.5 * y
    return ((1 - a * x_phys^2 + y_phys) / 1.5, (b * x_phys) / 0.5)
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

# sparse noise stencil (uniform box) consistent with noise_matvec
function noise_sparse(noise, partition, nx, ny, b2g, g2b; periodic)
    n = nx * ny; dom = partition.domain
    dx = diam(dom[1]) / nx; dy = diam(dom[2]) / ny
    rx = max(1, Int(round((diam(noise[1]) / 2) / dx)))
    ry = max(1, Int(round((diam(noise[2]) / 2) / dy)))
    w = 1.0 / ((2rx + 1) * (2ry + 1))
    I = Int[]; J = Int[]; V = Float64[]
    for j in 1:n
        jx, jy = b2g[j]
        for ox in -rx:rx, oy in -ry:ry
            if periodic
                ix = mod1(jx + ox, nx); iy = mod1(jy + oy, ny)
            else
                ix = jx + ox; iy = jy + oy
                (1 <= ix <= nx && 1 <= iy <= ny) || continue
            end
            t = g2b[iy, ix]; t == 0 && continue
            push!(I, t); push!(J, j); push!(V, w)
        end
    end
    return sparse(I, J, V, n, n), rx
end

# Build the sparse transfer operator L = N·P_det (column-normalised) and its adjoint.
function build_L(F, domain, noise, label; depth, periodic, enclosure_method)
    partition = build_uniform_partition(domain, depth)
    n = partition.n_boxes; nx = ny = 2^depth
    b2g, g2b = build_grid_index_maps(partition, (nx, ny))
    println("\n", "="^70); @printf("%s — depth %d, %d boxes\n", label, depth, n); println("="^70)
    P_det = build_transfer_forward(F, partition, (nx, ny);
        subdiv = 4, enclosure_method = enclosure_method, periodic = periodic)
    normalize_columns!(P_det)
    N, rx = noise_sparse(noise, partition, nx, ny, b2g, g2b; periodic = periodic)
    L = N * P_det
    @printf("  σ/δ = %d cells,  nnz(L) = %d (%.1f/col)\n", rx, nnz(L), nnz(L) / n)
    return L, n
end

# ============================================================ matvec Lanczos certifier
# (numerical port of RigorousPseudospectra.far_block_sigma_min: selective-reorth Lanczos
#  + online Paige ‖R₂‖; certifies σ_min(G₂₂ − zI), G₂₂ = P_m L P_m.)
function lanczos_pro(Mz, n, kmax; v0)
    T = eltype(v0); R = real(T)                          # generic precision (Float64 or BigFloat)
    u = eps(R); trig = sqrt(u); keepθ = u^(R(3) / 4); reset = 2u
    α = zeros(R, kmax); β = zeros(R, kmax); fn = zeros(R, kmax)
    W = zeros(R, kmax + 1, kmax + 1); cols = Vector{Vector{T}}(undef, kmax)
    q = copy(v0); q ./= norm(q); qprev = zeros(T, n); βprev = zero(R); kact = kmax
    for k in 1:kmax
        cols[k] = q; W[k, k] = u
        w = Mz(q); a = real(dot(q, w)); α[k] = a
        r = w .- a .* q .- βprev .* qprev
        if k == 1
            W[1, 2] = W[2, 1] = u
        else
            W[k, k + 1] = W[k + 1, k] = u
            ξ = u * (abs(α[k]) + norm(r))
            for i in (k - 1):-1:1
                num = β[i] * W[i + 1, k] + abs(α[i] - α[k]) * W[i, k] +
                      (i > 1 ? β[i - 1] * W[i - 1, k] : zero(R)) +
                      β[k - 1] * W[i, k - 1] + ξ + u * (abs(α[i]) + β[i])
                W[i, k + 1] = W[k + 1, i] = num / max(norm(r), u)
            end
            if maximum(@view W[1:k, k + 1]) > trig
                idx = findall(i -> W[i, k + 1] > keepθ, 1:k)
                for _ in 1:2, i in idx
                    r .-= cols[i] .* (cols[i]' * r)
                end
                for i in idx
                    W[i, k + 1] = W[k + 1, i] = reset
                end
            end
        end
        b = norm(r); β[k] = b
        (!isfinite(b) || b <= u * abs(a)) && (kact = k; break)
        qnext = r ./ b
        fn[k] = norm(a .* q .+ βprev .* qprev .+ b .* qnext .- w)
        qprev, βprev, q = q, b, qnext
    end
    R2 = maximum(sum(@view W[i, 1:kact]) for i in 1:kact)
    wlast = W[1:kact, kact + 1]              # Paige estimate of |⟨q_i, q_{kact+1}⟩| (orthogonality)
    return α[1:kact], β[1:kact], fn[1:kact], wlast, R2, kact
end

function far_floor(α, β, fn, wlast, R2, k)
    kk = k
    for j in 1:k
        if !isfinite(α[j]) || (j < k && !isfinite(β[j]))
            kk = j - 1; break
        end
    end
    kk < 2 && return 0.0
    # θmin = smallest eigenvalue of the tridiagonal — in Float64 (eigenvalues are well
    # conditioned and ~1e-7 is representable; avoids slow generic BigFloat eigen).
    θmin = minimum(eigvals(SymTridiagonal(Float64.(α[1:kk]), Float64.(β[1:(kk - 1)]))))
    # Deviation lemma: every eigenvalue of M_z is within β22 = ‖R₁‖₂/(1−‖R₂‖) of spec(T_k),
    # R₁ = Q*(M_zQ − QT_k) = β_k (Q*q_{k+1}) eₖᵀ + Q*F_k.  Entrywise
    #   |R₁[i,j]| ≤ β_k·wlast[i]·[j=k] + |⟨q_i,f_j⟩| ≤ β_k·wlast[i]·[j=k] + fn[j].
    # The β_k term is KILLED by the orthogonality wlast (≈√u with reorth) — the reason it must
    # NOT appear bare. Bound ‖R₁‖₂ ≤ √(‖R₁‖₁‖R₁‖∞) (geometric mean of 1- and ∞-norms).
    fns = @view fn[1:kk]
    sw = sum(@view wlast[1:kk])
    R1_1 = max(β[kk] * sw + kk * fns[kk], kk * maximum(fns))    # max column sum
    R1_inf = β[kk] * maximum(@view wlast[1:kk]) + sum(fns)      # max row sum
    R1 = sqrt(R1_1 * R1_inf)
    β22 = R2 < 1 ? Float64(R1 / (1 - R2)) : Inf          # tiny in BigFloat ⇒ clears θmin
    return sqrt(max(0.0, θmin - β22))
end

# σ_min(G₂₂ − zI) with G₂₂ = P_m L P_m, by ONE matvec Lanczos. `kmax` caps the steps: in a
# spectral gap (firing regime) the bottom eigenvalue converges fast, so the full basis isn't
# needed; selective reorth keeps ‖R₂‖<1.
# σ_min(G₂₂ − zI), G₂₂ = P_m L P_m, by ONE matvec Lanczos at full basis. `L` and `W` are in
# precision `T` (Float64 or BigFloat); BigFloat lowers the semi-orthogonality floor √u so the
# tiny far σ_min² of a strongly non-normal block clears the certificate floor.
function far_sigma_min(L, W, z, n; nullshift = 50.0)
    T = eltype(L)
    kmax = n - size(W, 2) - 1
    Pm(x) = x .- W * (W' * x)
    zz = Complex{real(T)}(z); ns = real(T)(nullshift)
    Mz(x) = (v1 = Pm(x); v2 = L * v1 .- zz .* v1; v3 = Pm(v2);
    v4 = (L' * v3) .- conj(zz) .* v3; Pm(v4) .+ ns .* (W * (W' * x)))
    v0 = Pm(randn(Complex{real(T)}, n))
    α, β, fn, wlast, R2, k = lanczos_pro(Mz, n, kmax; v0 = v0)
    return far_floor(α, β, fn, wlast, R2, k), Float64(R2), k
end

# min over the circle |z| = r of σ_min(G₂₂ − zI): nθ samples (PARALLEL — independent Lanczos
# runs) + 1-Lipschitz arc-cover.
function far_min_on_circle(L, W, r, n; nθ = 16, verbose = false)
    s = zeros(Float64, nθ); R2s = zeros(Float64, nθ); ks = zeros(Int, nθ)
    Threads.@threads for t in 0:(nθ - 1)
        fl, R2, k = far_sigma_min(L, W, r * cis(2π * t / nθ), n)
        s[t + 1] = fl; R2s[t + 1] = R2; ks[t + 1] = k
    end
    verbose && for t in 1:nθ
        @printf("      θ=%.2fπ  σ22=%.4e  ‖R₂‖=%.2e  k=%d\n", 2(t - 1) / nθ, s[t], R2s[t], ks[t])
    end
    return minimum(s) - 2r * sin(π / (2nθ)), minimum(s)
end

# ============================================================ contraction estimate
function estimate_N(L, n, label; nev = 14, α = 0.5, nθ = 24, prec::Type = Float64)
    println("  eigs: leading $nev modes...")
    vals, vecs = eigs(L; nev = nev, which = :LM)
    o = sortperm(abs.(vals); rev = true); λ = vals[o]; V = vecs[:, o]
    av = abs.(λ)
    @printf("  |λ|: %s\n", join(round.(av; digits = 4), ", "))

    # auto-select the capture cutoff m at the LARGEST relative gap below λ₂ (so the far
    # circle |z|=r_far sits in a genuine spectral gap, never splitting a cluster).
    ratios = av[1:(end - 1)] ./ av[2:end]
    m = argmax(@view ratios[2:(end - 1)]) + 1            # best split k≥2 (skip the 1↔λ₂ gap)
    r_far = sqrt(av[m] * av[m + 1])                      # geometric mean → middle of the gap
    @printf("  capture m=%d (|λ_m|=%.4f), far starts at |λ_{m+1}|=%.4f ⇒ r_far=%.4f\n",
        m, av[m], av[m + 1], r_far)

    # capture m slow modes (incl. μ at λ₁≈1); orthonormalise
    W = Matrix(qr(V[:, 1:m]).Q)[:, 1:m]
    H = W' * (L * W)                                   # m×m captured block
    # deflate μ (λ=1 eigenvector of H) → slow block S on U₀∩slow ((m−1)-dim)
    EH = eigen(H); iμ = argmin(abs.(EH.values .- 1)); μc = EH.vectors[:, iμ]; μc ./= norm(μc)
    Qc = Matrix(qr(hcat(μc, randn(ComplexF64, m, m - 1))).Q)[:, 2:m]   # orthonormal ⊥ μc
    S = Qc' * H * Qc                                   # (m−1)×(m−1) slow block on U₀
    @printf("  slow block S: %d×%d, ‖S‖=%.4f, spectral radius=%.5f\n",
        m - 1, m - 1, opnorm(S, 2), maximum(abs.(eigvals(S))))

    # far block in precision `prec` (BigFloat clears the √u orthogonality floor for very
    # non-normal blocks, where the true far σ_min² sits below the Float64 certificate floor).
    Lp = prec == Float64 ? L : SparseMatrixCSC{prec}(L)
    Wp = Matrix{Complex{prec}}(W)
    @printf("  certifying far block on |z| = r_far = %.5f  (%s, full-basis Lanczos/sample)...\n",
        r_far, prec)
    t = @elapsed (s_far, raw) = far_min_on_circle(Lp, Wp, r_far, n; nθ = nθ, verbose = true)
    @printf("  far σ_min ≥ %.4e on |z|=r_far  (min sample %.4e, %.1fs)\n", s_far, raw, t)
    if s_far <= 0
        println("  ✗ far block not certified on this circle (raise m or move r_far)"); return
    end
    bcauchy(j) = r_far^(j + 1) / s_far                 # ‖G₂₂ʲ‖₂ ≤ r_far^{j+1}/s_far
    Bcoup = opnorm((L * W) .- W * H, 2)                # ‖(I−WW*)LW‖ — slow→far coupling ‖B‖

    # assemble  ‖Lⁿ|_{U₀}‖ ≤ ‖Sⁿ‖ + ‖G₂₂ⁿ‖ + ‖B‖ Σ_k ‖Sᵏ‖‖G₂₂^{n−1−k}‖
    Spow = [opnorm(S^k, 2) for k in 0:200]
    function total(nn)
        conv = sum(Spow[k + 1] * bcauchy(nn - 1 - k) for k in 0:(nn - 1))
        return Spow[nn + 1] + bcauchy(nn) + Bcoup * conv
    end
    @printf("  ‖B‖ (coupling) = %.4e\n", Bcoup)
    println("    n     ‖Sⁿ‖       ‖G₂₂ⁿ‖     bound ‖Lⁿ|_U₀‖₂")
    Nstar = 0
    for nn in 1:200
        tb = total(nn)
        if nn <= 12 || tb < α || nn % 10 == 0
            @printf("    %-4d  %.3e  %.3e  %.4e %s\n", nn, Spow[nn + 1], bcauchy(nn), tb,
                tb < α ? "← ≤ α" : "")
        end
        if tb < α
            Nstar = nn; break
        end
    end
    rate = max(maximum(abs.(eigvals(S))), r_far)
    if Nstar > 0
        @printf("  ✓ ‖Lⁿ|_{U₀}‖₂ ≤ α=%.2f at N = %d  (rate ρ = max(|λ₂|,r_far) = %.5f)\n",
            α, Nstar, rate)
        @printf("    L¹: ‖·‖₁ ≤ √%d·‖·‖₂ ⇒ N_{L¹} ≈ %d  (same rate, +log√dim/|log ρ|)\n",
            n, Nstar + ceil(Int, log(sqrt(n)) / (-log(rate))))
    else
        println("  ✗ bound did not drop below α within 200 steps")
    end
    return Nstar
end

# ============================================================ run
DEPTH = get(ENV, "DEPTH", "") == "" ? 6 : parse(Int, ENV["DEPTH"])
NEV = get(ENV, "NEV", "") == "" ? 14 : parse(Int, ENV["NEV"])
ALPHA = get(ENV, "ALPHA", "") == "" ? 0.5 : parse(Float64, ENV["ALPHA"])
BITS = get(ENV, "BITS", "") == "" ? 256 : parse(Int, ENV["BITS"])
setprecision(BigFloat, BITS)

# Hénon: far block is mildly non-normal ⇒ Float64 suffices.
Lh, nh = build_L(henon_rescaled, IntervalBox(interval(-1.0, 1.0), interval(-1.0, 1.0)),
    uniform_noise_box(0.1, 2), "Hénon (periodic)";
    depth = DEPTH, periodic = true, enclosure_method = first_order_enclosure)
estimate_N(Lh, nh, "Hénon"; nev = NEV, α = ALPHA, prec = Float64)

# Contracting Lorenz: strongly non-normal far block (σ_min² ~ 1e-6 at the Float64 floor)
# ⇒ run the far-block Lanczos in BigFloat ($BITS bits) so it clears the orthogonality floor.
Ll, nl = build_L(contracting_lorenz, IntervalBox(interval(-0.5, 0.5), interval(-0.5, 0.5)),
    uniform_noise_box(0.1, 2), "Contracting Lorenz";
    depth = DEPTH, periodic = false, enclosure_method = zeroth_order_enclosure)
estimate_N(Ll, nl, "Lorenz"; nev = NEV, α = ALPHA, prec = BigFloat)
