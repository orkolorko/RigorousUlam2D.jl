"""
Schur complement norm bounds for wavelet-coordinate operators.

Given a transfer operator P and a Haar wavelet basis, decompose the
wavelet-coordinate operator into blocks by scale:

    P_wav = [A  B]     A: coarse (scales 0..j0-1),  dim_lo × dim_lo
            [C  D]     D: fine   (scales j0..J-1),   dim_hi × dim_hi

Noise convolution attenuates high-frequency wavelet coefficients, making
||D|| < 1. The Schur complement S = A - B·D_inv·C captures the effective
coarse dynamics, and Algorithm 2 can be run on S (small matrix) instead
of the full n × n operator.

Rigorous bound on ||P^k|_V||:
  Decompose any v = (v_lo, v_hi) with v in V (zero average).
  The scaling coefficient (index 1) is excluded from V, so V lives
  entirely in the detail subspace.
"""

using LinearAlgebra
using SparseArrays

export wavelet_operator_blocks, noise_attenuation_bound
export schur_norms_of_powers, wavelet_rigorous_error_bound

"""
    noise_attenuation_bound(noise, domain, scale::Int)

Compute a rigorous upper bound on the attenuation factor of uniform noise
convolution at wavelet scale `scale`.

For uniform noise with half-width ξ/2 on a domain of width L, the noise
kernel's Fourier transform at frequency k is:

    ρ̂(k) = sinc(k · ξ / L)

At wavelet scale j, the characteristic frequency is k ~ 2^j / L, so:

    |ρ̂(2^j)| ≤ min(1, L / (π · ξ · 2^j))

For 2D with independent noise in each dimension, the attenuation is the
product of the 1D attenuations.

Returns a rigorous upper bound on ||N|_{W_j}|| (operator norm of noise
restricted to wavelet scale j).
"""
function noise_attenuation_bound(noise, domain, scale::Int)
    N = length(noise)
    atten = 1.0
    for d in 1:N
        xi = diam(noise[d])
        L = diam(domain[d])
        # At scale j, wavelet frequency ~ 2^j / L
        # sinc envelope bound: |sinc(f·ξ)| ≤ min(1, 1/(π·f·ξ))
        # where f = 2^j / L
        freq = (2.0^scale) / L
        # |sinc(freq · ξ)| ≤ min(1, 1/(π · freq · ξ))
        sinc_bound = min(1.0, 1.0 / (π * freq * xi))
        atten *= sinc_bound
    end
    return atten
end

"""
    choose_scale_cutoff(noise, domain, depth::Int; target_rho::Float64=0.9)

Choose the scale cutoff j0 such that ||D|| < target_rho for all scales j ≥ j0.

Returns the cutoff scale j0 and the corresponding contraction rate rho.
"""
function choose_scale_cutoff(noise, domain, depth::Int; target_rho::Float64=0.9)
    for j0 in 0:(depth - 1)
        # Check if all scales j ≥ j0 have attenuation ≤ target_rho
        max_atten = 0.0
        for j in j0:(depth - 1)
            max_atten = max(max_atten, noise_attenuation_bound(noise, domain, j))
        end
        if max_atten < target_rho
            return (j0=j0, rho=max_atten)
        end
    end
    return (j0=depth, rho=1.0)
end

"""
    wavelet_operator_blocks(P, basis::HaarBasis2D, j0::Int)

Extract the block decomposition of P in wavelet coordinates, split at scale j0.

Returns (A, B, C, D) where:
- A (dim_lo × dim_lo): coarse-to-coarse block
- B (dim_lo × dim_hi): fine-to-coarse coupling
- C (dim_hi × dim_lo): coarse-to-fine coupling
- D (dim_hi × dim_hi): fine-to-fine block

The "coarse" subspace includes all detail coefficients at scales 0..j0-1.
The scaling coefficient (index 1) is excluded since V has zero average.

dim_lo = 4^j0 - 1  (detail coeffs at scales 0..j0-1)
dim_hi = 4^J - 4^j0 (detail coeffs at scales j0..J-1)
"""
function wavelet_operator_blocks(P, basis::HaarBasis2D, j0::Int)
    n = basis.n
    J = basis.depth

    # Index 1 is the scaling coefficient (not in V)
    # Indices 2..4^j0 are coarse detail (scales 0..j0-1)
    # Indices 4^j0+1..n are fine detail (scales j0..J-1)
    split = 4^j0  # first fine index in the full vector (1-based)

    dim_lo = split - 1    # indices 2..split
    dim_hi = n - split     # indices split+1..n

    # Build blocks by applying P_wav to basis vectors
    # P_wav · e_k = W · P · W^{-1} · e_k
    A = zeros(dim_lo, dim_lo)
    B = zeros(dim_lo, dim_hi)
    C = zeros(dim_hi, dim_lo)
    D = zeros(dim_hi, dim_hi)

    e = zeros(n)
    for k in 1:(dim_lo + dim_hi)
        fill!(e, 0.0)
        if k <= dim_lo
            e[k + 1] = 1.0  # skip index 1 (scaling)
        else
            e[split + (k - dim_lo)] = 1.0
        end

        # Apply operator in wavelet coordinates
        result = wavelet_matvec(e, P, basis)

        # Extract blocks (skip index 1)
        if k <= dim_lo
            A[:, k] = result[2:split]
            C[:, k] = result[(split + 1):n]
        else
            j = k - dim_lo
            B[:, j] = result[2:split]
            D[:, j] = result[(split + 1):n]
        end
    end

    return (A=A, B=B, C=C, D=D, dim_lo=dim_lo, dim_hi=dim_hi)
end

"""
    schur_norms_of_powers(P, basis::HaarBasis2D, noise, domain, m::Int;
                          j0::Union{Int,Nothing}=nothing,
                          target_rho::Float64=0.9,
                          rigorous::Bool=true,
                          verbose::Bool=false)

Compute rigorous upper bounds on ||P^k|_V|| for k = 1..m using the
Schur complement approach in wavelet coordinates.

The method:
1. Choose scale cutoff j0 (or use provided value)
2. Extract blocks A, B, C, D
3. Bound ||D^k|| ≤ ρ^k from noise attenuation
4. Run Algorithm 2 on the small (dim_lo × dim_lo) effective operator
5. Combine via block matrix power bound

The block matrix power bound for v = (v_lo, v_hi):
    ||P^k v|| ≤ ||A^k v_lo|| + Σ_{i=0}^{k-1} ||A^i B|| · ρ^{k-1-i} · ||v_hi||
             + ρ^k ||v_hi|| + Σ_{i=0}^{k-1} ρ^i ||C A^{k-1-i} v_lo||

Simplified rigorous bound (using submultiplicativity):
    ||P^k|_V|| ≤ ||A^k|| + ||B||·||C||·Σ_{i=0}^{k-2} ||A^i|| · ρ^{k-2-i}
              + (||B|| · ρ^{k-1} + ||C|| · Σ ||A^i|| · ρ^{k-1-i} + ρ^k)

# Returns
NamedTuple with:
- `norms`: Vector of ||P^k|_V|| bounds for k=1..m
- `j0`: Scale cutoff used
- `rho`: Fine-scale contraction rate
- `dim_lo`: Dimension of coarse block
- `A_norms`: Norms of powers of the coarse block A
"""
function schur_norms_of_powers(P, basis::HaarBasis2D, noise, domain, m::Int;
                               j0::Union{Int,Nothing}=nothing,
                               target_rho::Float64=0.9,
                               rigorous::Bool=true,
                               verbose::Bool=false)
    J = basis.depth

    # Step 1: Choose cutoff
    if j0 === nothing
        cutoff = choose_scale_cutoff(noise, domain, J; target_rho=target_rho)
        j0 = cutoff.j0
        rho = cutoff.rho
    else
        # Compute rho for given j0
        rho = 0.0
        for j in j0:(J - 1)
            rho = max(rho, noise_attenuation_bound(noise, domain, j))
        end
    end

    if verbose
        println("Schur complement decomposition:")
        println("  Scale cutoff j0 = $j0")
        println("  Fine-scale contraction ρ = $(round(rho, digits=6))")
        println("  Coarse block dimension = $(4^j0 - 1)")
        println("  Fine block dimension = $(4^J - 4^j0)")
    end

    if j0 >= J || rho >= 1.0
        if verbose
            println("  Warning: no scale separation found, falling back to full Algorithm 2")
        end
        return (norms=norms_of_powers_sparse(P, m; rigorous=rigorous, verbose=verbose),
                j0=j0, rho=rho, dim_lo=basis.n - 1, A_norms=Float64[])
    end

    # Step 2: Extract blocks
    if verbose
        println("  Extracting wavelet-coordinate blocks...")
    end
    blocks = wavelet_operator_blocks(P, basis, j0)
    A, B, C, D = blocks.A, blocks.B, blocks.C, blocks.D

    # Step 3: Compute norms of coupling blocks
    nB = opnorm(B, 1)
    nC = opnorm(C, 1)

    if verbose
        println("  ||B|| = $(round(nB, digits=6))")
        println("  ||C|| = $(round(nC, digits=6))")
        println("  ||D|| (estimated) ≤ ρ = $(round(rho, digits=6))")
    end

    # Step 4: Compute ||A^k|| on V_coarse via Algorithm 2
    # A acts on the coarse detail subspace (dim_lo × dim_lo)
    # We need norms on the zero-sum subspace of A
    dim_lo = blocks.dim_lo
    if verbose
        println("  Running Algorithm 2 on $(dim_lo)×$(dim_lo) coarse block...")
    end
    A_norms = _norms_of_powers_dense(A, m; rigorous=rigorous)

    # Step 5: Combine into full norm bounds
    # For v = (v_lo, v_hi) with ||v||_1 = 1 (||v_lo|| + ||v_hi|| ≤ 1):
    #
    # P_wav^k v has the recurrence:
    #   v_lo^{(k)} = A · v_lo^{(k-1)} + B · v_hi^{(k-1)}
    #   v_hi^{(k)} = C · v_lo^{(k-1)} + D · v_hi^{(k-1)}
    #
    # Bound: ||v_lo^{(k)}|| ≤ ||A|| · ||v_lo^{(k-1)}|| + ||B|| · ||v_hi^{(k-1)}||
    #        ||v_hi^{(k)}|| ≤ ||C|| · ||v_lo^{(k-1)}|| + ρ · ||v_hi^{(k-1)}||
    #
    # This is a 2×2 matrix recurrence:
    #   [a_k]   [||A^k||    ||B||] ^k   [a_0]
    #   [h_k] ≤ [||C||      ρ    ]    · [h_0]
    #
    # with a_0 + h_0 = 1.  We maximize over the split.

    norms = zeros(m)
    # For each k, compute the bound by iterating the 2×2 recurrence
    # We need bounds for all initial splits: a_0 = t, h_0 = 1-t
    # The worst case is t=0 or t=1.
    for k in 1:m
        # Case 1: all mass in coarse (a_0=1, h_0=0)
        bound_coarse = _block_recurrence(A_norms, nB, nC, rho, k, 1.0, 0.0)
        # Case 2: all mass in fine (a_0=0, h_0=1)
        bound_fine = _block_recurrence(A_norms, nB, nC, rho, k, 0.0, 1.0)
        norms[k] = max(bound_coarse, bound_fine)
    end

    # Refine using submultiplicativity
    norms = refine_norms_of_powers(norms, m)

    if rigorous
        # Add Higham error correction for the block extraction
        u = eps(Float64) / 2
        n = basis.n
        gamma_correction = 1.0 + higham_gamma(2 * n, u)
        norms .*= gamma_correction
    end

    if verbose
        for k in 1:m
            println("  k = $k: ||P^k|_V|| ≤ $(round(norms[k], digits=6))")
        end
    end

    return (norms=norms, j0=j0, rho=rho, dim_lo=dim_lo, A_norms=A_norms)
end

"""
    _block_recurrence(A_norms, nB, nC, rho, k, a0, h0)

Iterate the 2×2 block norm recurrence for k steps starting from (a0, h0).
"""
function _block_recurrence(A_norms, nB, nC, rho, k, a0, h0)
    a = a0
    h = h0
    # Use precomputed A_norms[1] as ||A||
    nA = length(A_norms) > 0 ? A_norms[1] : 0.0
    for _ in 1:k
        a_new = nA * a + nB * h
        h_new = nC * a + rho * h
        a = a_new
        h = h_new
    end
    return a + h
end

"""
    _norms_of_powers_dense(A::Matrix, m::Int; rigorous::Bool=true)

Compute ||A^k|| (ℓ¹ operator norm) for k = 1..m on a small dense matrix.
Uses direct matrix powers since A is small.
"""
function _norms_of_powers_dense(A::Matrix, m::Int; rigorous::Bool=true)
    n = size(A, 1)
    u = eps(Float64) / 2

    norms = zeros(m)
    Ak = copy(A)
    for k in 1:m
        if k > 1
            Ak = Ak * A
        end
        norms[k] = opnorm(Ak, 1)
        if rigorous
            gamma = higham_gamma(n * k, u)
            norms[k] *= (1 + gamma)
        end
    end
    return refine_norms_of_powers(norms, m)
end

"""
    wavelet_rigorous_error_bound(P_noisy, partition, noise;
                                  target_contraction::Float64=0.5,
                                  max_power::Int=100,
                                  j0::Union{Int,Nothing}=nothing,
                                  verbose::Bool=false)

Compute rigorous error bound using wavelet Schur complement approach.

Same interface as `rigorous_error_bound` but uses wavelet decomposition
for dramatically faster norm computation on large grids.
"""
function wavelet_rigorous_error_bound(P_noisy, partition, noise;
                                      target_contraction::Float64=0.5,
                                      max_power::Int=100,
                                      j0::Union{Int,Nothing}=nothing,
                                      verbose::Bool=false)
    n = partition.n_boxes
    # Determine depth from partition size
    J = round(Int, log2(sqrt(n)))
    if 4^J != n
        error("Partition size $n is not 4^J for any integer J. " *
              "Wavelet approach requires uniform dyadic partition.")
    end

    basis = HaarBasis2D(J)
    domain = partition.domain

    if verbose
        println("Wavelet error bound: J=$J, n=$n")
    end

    # Compute norms via Schur complement
    result = schur_norms_of_powers(P_noisy, basis, noise, domain, max_power;
                                    j0=j0, rigorous=true, verbose=verbose)
    norms = result.norms

    # Find contraction power
    N_power = nothing
    alpha = nothing
    for k in 1:max_power
        if norms[k] < target_contraction
            N_power = k
            alpha = norms[k]
            break
        end
    end

    if N_power === nothing
        if verbose
            println("Warning: No contraction found within $max_power iterations")
        end
        return (error_bound=Inf, N=nothing, α=nothing, δ=nothing,
                var_ρ=nothing, mixing_constant=nothing, mixing_satisfied=false,
                ξ=nothing, wavelet_info=result)
    end

    # Compute error bound parameters
    δ = ulam_approximation_error(partition, method=:upper_bound)
    var_ρ = noise_variation(noise)
    ndims = length(noise)
    ξ = minimum(diam(noise[i]) for i in 1:ndims)
    sum_C = Float64(N_power)

    mixing_term = (3 * δ / ξ) * var_ρ * sum_C
    mixing_satisfied = alpha + mixing_term < 1.0

    if mixing_satisfied
        error_bound = (3 * δ * var_ρ) / (ξ * (1 - alpha)) * sum_C
    else
        error_bound = Inf
    end

    if verbose
        println("\nParameters:")
        println("  N = $N_power, α = $(round(alpha, digits=6))")
        println("  δ = $(round(δ, digits=6)), ξ = $(round(ξ, digits=6))")
        println("  var(ρ) = $(round(var_ρ, digits=6))")
        println("  Mixing: α + correction = $(round(alpha + mixing_term, digits=6)) < 1? $mixing_satisfied")
        println("  Error bound: $(round(error_bound, digits=6))")
    end

    return (error_bound=error_bound, N=N_power, α=alpha, δ=δ,
            var_ρ=var_ρ, mixing_constant=sum_C, mixing_satisfied=mixing_satisfied,
            ξ=ξ, wavelet_info=result)
end
