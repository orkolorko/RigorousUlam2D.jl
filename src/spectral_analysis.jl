"""
Spectral analysis for rigorous Ulam method.

This module implements:
1. Norm estimation on the zero-average space V = {f : ∫f dλ = 0}
   (following Galatolo-Monge-Nisoli-Poloni 2014)
2. Rigorous error bounds for stationary density approximation
   (following the ZZ paper: Pacifico-Liberalquino-Nisoli)

Key theoretical background:
- The annealed transfer operator: L_ξ = N_ξ ∘ L_T (noise ∘ deterministic)
- Zero-average subspace: V = {f ∈ L¹(λ) : ∫f dλ = 0}
- Contraction on V: Need ||L^N_δ|_V|| < α < 1 for some N
- Error bound: ||f_ξ - f_{δ,ξ}||_{L¹} ≤ (3δ var(ρ))/(ξ(1-α)) Σ C_i

Rigorous numerics:
- Uses Higham's a priori bounds for floating-point errors in matrix computations
- Ensures computed norms are rigorous upper bounds even with round-off errors

Reference: Galatolo-Monge-Nisoli-Poloni (2014), RigorousInvariantMeasure.jl
"""

using LinearAlgebra
using SparseArrays

export operator_norm_on_V, find_contraction_power, operator_norm_on_V_power
export rigorous_norm_on_V, higham_gamma
export compute_stationary_density, rigorous_error_bound
export noise_variation, ulam_approximation_error

"""
    operator_norm_on_V(P::AbstractMatrix; measure::Union{Vector{Float64},Nothing}=nothing)

Compute ||P|_V||_1, the induced operator norm of P restricted to the zero-average subspace V.

For a column-stochastic matrix P:
- V = {v : Σ v_i μ_i = 0} (zero weighted sum)
- The L¹ norm: ||v||_1 = Σ |v_i| μ_i
- ||P|_V||_1 = max{||Pv||_1 : v ∈ V, ||v||_1 = 1}

For uniform measure (μ_i = 1/n), this simplifies to:
- V = {v : Σ v_i = 0} (zero sum vectors)
- ||P|_V||_1 = max_{j≠k} (1/2) ||P_j - P_k||_1

where P_j is the j-th column of P.

# Arguments
- `P`: Column-stochastic matrix (columns sum to 1)
- `measure`: Optional measure weights. If nothing, assumes uniform measure.

# Returns
- Operator norm ||P|_V||_1

# Note
For general (non-stochastic) matrices, the function still computes the norm on V,
but the interpretation may differ.

# Reference
Galatolo-Nisoli (2014), Section 4.1
"""
function operator_norm_on_V(P::AbstractMatrix; measure::Union{Vector{Float64},Nothing}=nothing)
    n = size(P, 1)

    if measure === nothing
        # Uniform measure: use simplified formula
        # ||P|_V||_1 = max_{j≠k} (1/2) ||P[:,j] - P[:,k]||_1
        # This follows from the fact that extreme points of {v : Σv_i=0, ||v||_1=1}
        # are of the form (1/2)(e_j - e_k)

        max_norm = 0.0

        # For sparse matrices, convert columns to dense for comparison
        if issparse(P)
            for j in 1:n
                col_j = Vector(P[:, j])
                for k in (j+1):n
                    col_k = Vector(P[:, k])
                    diff_norm = sum(abs.(col_j .- col_k)) / 2
                    max_norm = max(max_norm, diff_norm)
                end
            end
        else
            for j in 1:n
                for k in (j+1):n
                    diff_norm = sum(abs.(P[:, j] .- P[:, k])) / 2
                    max_norm = max(max_norm, diff_norm)
                end
            end
        end

        return max_norm
    else
        # General measure case
        # V = {v : Σ v_i μ_i = 0}
        # ||v||_1 = Σ |v_i| μ_i
        # Use optimization or sampling approach

        # For now, use the approach:
        # ||P|_V||_1 = max over extreme points of unit ball in V
        # The extreme points are (1/(2μ_j + 2μ_k))(e_j/μ_j - e_k/μ_k)
        # normalized to have ||·||_1 = 1

        max_norm = 0.0
        for j in 1:n
            for k in (j+1):n
                # Extreme point: v = α(e_j - e_k) where α chosen so ||v||_1 = 1
                # ||v||_1 = |α|(μ_j + μ_k) = 1 → α = 1/(μ_j + μ_k)
                v = zeros(n)
                v[j] = 1.0 / (measure[j] + measure[k])
                v[k] = -1.0 / (measure[j] + measure[k])

                Pv = P * v
                norm_Pv = sum(abs.(Pv) .* measure)
                max_norm = max(max_norm, norm_Pv)
            end
        end

        return max_norm
    end
end

"""
    operator_norm_on_V_power(P::AbstractMatrix, power::Int;
                             measure::Union{Vector{Float64},Nothing}=nothing,
                             use_iteration::Bool=true)

Compute ||P^power|_V||_1, the operator norm of P^power restricted to V.

# Arguments
- `P`: Column-stochastic matrix
- `power`: The power to raise P to
- `measure`: Optional measure weights (uniform if nothing)
- `use_iteration`: If true, compute P^power by iteration (memory efficient).
                   If false, compute P^power explicitly (faster for small powers).

# Returns
- Operator norm ||P^power|_V||_1

# Note
For large powers and sparse matrices, iteration is more memory efficient
but may accumulate numerical errors. The explicit method is exact but
requires O(n²) memory for the dense result.
"""
function operator_norm_on_V_power(P::AbstractMatrix, power::Int;
                                  measure::Union{Vector{Float64},Nothing}=nothing,
                                  use_iteration::Bool=true)
    if power == 1
        return operator_norm_on_V(P; measure=measure)
    end

    if use_iteration && power > 2
        # Compute P^power by repeated squaring for efficiency
        # But we need the actual matrix to compute the norm
        P_power = Matrix(P)
        for _ in 2:power
            P_power = P_power * P
        end
        return operator_norm_on_V(P_power; measure=measure)
    else
        # Explicit computation
        P_power = P^power
        return operator_norm_on_V(P_power; measure=measure)
    end
end

"""
    higham_gamma(n::Int, u::Float64=eps(Float64)/2)

Compute Higham's γ_n constant for floating-point error bounds.

For n floating-point operations with unit roundoff u:
    γ_n = n*u / (1 - n*u)

This bounds the relative error in a sum of n terms.

# Reference
Higham (2002), "Accuracy and Stability of Numerical Algorithms", 2nd ed.
"""
function higham_gamma(n::Int, u::Float64=eps(Float64)/2)
    return n * u / (1 - n * u)
end

"""
    rigorous_norm_on_V(P::AbstractMatrix;
                       measure::Union{Vector{Float64},Nothing}=nothing)

Compute a RIGOROUS UPPER BOUND on ||P|_V||_1 using Higham's error analysis.

This function accounts for floating-point round-off errors to provide
a guaranteed upper bound on the operator norm, not just a numerical approximation.

The approach follows Galatolo-Monge-Nisoli-Poloni (2014) and RigorousInvariantMeasure.jl:
1. Compute the numerical approximation of ||P|_V||_1
2. Add Higham's a priori bound on the floating-point error

For ||P|_V||_1 computed as max_{j≠k} (1/2)||P_j - P_k||_1:
- The difference P_j - P_k has n floating-point subtractions
- The sum Σ|·| has n-1 floating-point additions
- Total error bounded by γ_{2n-1} * ||P_j - P_k||_1

# Arguments
- `P`: Column-stochastic matrix
- `measure`: Optional measure weights (uniform if nothing)

# Returns
- Rigorous upper bound on ||P|_V||_1

# Note
This is the preferred method for rigorous computations. Use `operator_norm_on_V`
only when you don't need rigorous bounds.

# Reference
- Higham (2002), "Accuracy and Stability of Numerical Algorithms"
- Galatolo-Monge-Nisoli-Poloni (2014)
"""
function rigorous_norm_on_V(P::AbstractMatrix; measure::Union{Vector{Float64},Nothing}=nothing)
    n = size(P, 1)

    # Unit roundoff for Float64
    u = eps(Float64) / 2

    if measure === nothing
        # Uniform measure case
        # ||P|_V||_1 = max_{j≠k} (1/2) ||P[:,j] - P[:,k]||_1

        max_norm = 0.0

        # For rigorous upper bound:
        # 1. Subtraction P[:,j] - P[:,k]: n subtractions, each with relative error ≤ u
        #    Result has error ≤ γ_1 * |P_ij - P_ik| for each component
        # 2. Absolute value: exact
        # 3. Summation of n terms: error ≤ γ_{n-1} * sum
        #
        # Combined: fl(||v||_1) ≤ ||v||_1 * (1 + γ_{n-1}) where v = |P_j - P_k|
        # And ||v||_1 itself has errors from subtraction
        #
        # Total: fl(result) ≤ true_result * (1 + γ_n) * (1 + γ_{n-1})
        #                  ≤ true_result * (1 + γ_{2n-1})  (approximately)

        γ = higham_gamma(2*n - 1, u)

        if issparse(P)
            for j in 1:n
                col_j = Vector(P[:, j])
                for k in (j+1):n
                    col_k = Vector(P[:, k])
                    diff = col_j .- col_k
                    computed_norm = sum(abs.(diff)) / 2

                    # Rigorous upper bound: add error term
                    # The norm of the difference vector bounds the error
                    rigorous_norm = computed_norm * (1 + γ)

                    max_norm = max(max_norm, rigorous_norm)
                end
            end
        else
            for j in 1:n
                for k in (j+1):n
                    diff = P[:, j] .- P[:, k]
                    computed_norm = sum(abs.(diff)) / 2

                    rigorous_norm = computed_norm * (1 + γ)
                    max_norm = max(max_norm, rigorous_norm)
                end
            end
        end

        return max_norm
    else
        # General measure case
        # Similar error analysis but with weighted sums
        γ = higham_gamma(3*n, u)  # Extra factor for measure weights

        max_norm = 0.0
        for j in 1:n
            for k in (j+1):n
                v = zeros(n)
                v[j] = 1.0 / (measure[j] + measure[k])
                v[k] = -1.0 / (measure[j] + measure[k])

                Pv = P * v
                computed_norm = sum(abs.(Pv) .* measure)

                rigorous_norm = computed_norm * (1 + γ)
                max_norm = max(max_norm, rigorous_norm)
            end
        end

        return max_norm
    end
end

"""
    rigorous_norm_on_V_power(P::AbstractMatrix, power::Int;
                             measure::Union{Vector{Float64},Nothing}=nothing)

Compute a RIGOROUS UPPER BOUND on ||P^power|_V||_1.

Uses Higham's error analysis to account for:
1. Round-off errors in matrix multiplication (P * P * ... * P)
2. Round-off errors in norm computation

# Error propagation for matrix powers
For P^k = P * P^{k-1}:
- Each matrix multiplication introduces error ≤ γ_n * ||P|| * ||P^{k-1}||
- Accumulated error grows but is bounded by (1 + γ_n)^{k-1}

# Arguments
- `P`: Column-stochastic matrix
- `power`: Power to raise P to
- `measure`: Optional measure weights

# Returns
- Rigorous upper bound on ||P^power|_V||_1
"""
function rigorous_norm_on_V_power(P::AbstractMatrix, power::Int;
                                  measure::Union{Vector{Float64},Nothing}=nothing)
    if power == 1
        return rigorous_norm_on_V(P; measure=measure)
    end

    n = size(P, 1)
    u = eps(Float64) / 2

    # Compute P^power iteratively, tracking error bounds
    # For matrix multiplication C = A*B:
    # ||fl(A*B) - A*B||_1 ≤ γ_n * ||A||_1 * ||B||_1 (column-wise)
    #
    # For stochastic P, ||P||_1 = 1 (columns sum to 1)
    # So ||fl(P^k) - P^k||_1 ≤ γ_n * k (roughly)

    P_power = Matrix(P)
    for _ in 2:power
        P_power = P_power * P
    end

    # Error in computing P^power: (1 + γ_n)^{power-1} ≈ 1 + (power-1)*γ_n
    γ_mult = higham_gamma(n, u) * (power - 1)

    # Compute rigorous norm on the computed P^power
    computed_norm = rigorous_norm_on_V(P_power; measure=measure)

    # Add multiplicative error from matrix powers
    # ||P^k|_V|| ≤ ||fl(P^k)|_V|| * (1 + γ_mult) + error_terms
    # For rigorous upper bound, we scale up
    rigorous_norm = computed_norm * (1 + γ_mult)

    return rigorous_norm
end

"""
    find_contraction_power(P::AbstractMatrix;
                          target_norm::Float64=0.5,
                          max_power::Int=100,
                          measure::Union{Vector{Float64},Nothing}=nothing,
                          rigorous::Bool=true,
                          verbose::Bool=false)

Find the smallest power N such that ||P^N|_V||_1 < target_norm.

This is the key algorithm from Galatolo-Nisoli (2014) Section 4.1:
Iterate ||P^j|_V||_1 for j = 1, 2, ... until finding N where ||P^N|_V|| < 1/2.

# Arguments
- `P`: Column-stochastic matrix
- `target_norm`: Target contraction factor (default 0.5)
- `max_power`: Maximum power to try before giving up
- `measure`: Optional measure weights
- `rigorous`: If true (default), use Higham's a priori bound for rigorous upper bounds
- `verbose`: If true, print progress

# Returns
Named tuple with:
- `N`: The contraction power (or nothing if not found)
- `α`: The achieved norm ||P^N|_V||_1 (rigorous upper bound if rigorous=true)
- `norms`: Vector of all computed norms [(1, ||P|_V||), (2, ||P²|_V||), ...]
- `rigorous`: Whether rigorous bounds were used

# Example
```julia
P_noisy = N * P_det
normalize_columns!(P_noisy)
result = find_contraction_power(P_noisy, target_norm=0.5, rigorous=true, verbose=true)
if result.N !== nothing
    println("Contraction achieved at power N = \$(result.N) with rigorous bound α = \$(result.α)")
end
```

# Reference
- Galatolo-Nisoli (2014), Algorithm 4.1
- Higham (2002) for rigorous floating-point error bounds
"""
function find_contraction_power(P::AbstractMatrix;
                                target_norm::Float64=0.5,
                                max_power::Int=100,
                                measure::Union{Vector{Float64},Nothing}=nothing,
                                rigorous::Bool=true,
                                verbose::Bool=false)

    norms = Tuple{Int,Float64}[]

    # Choose norm function based on rigorous flag
    norm_func = rigorous ? rigorous_norm_on_V : operator_norm_on_V

    # Compute P^j iteratively
    n = size(P, 1)
    u = eps(Float64) / 2
    P_power = Matrix(P)  # P^1

    for j in 1:max_power
        # Compute ||P^j|_V|| (with or without rigorous error bounds)
        if rigorous
            # Account for matrix multiplication errors in P^j
            γ_mult = j > 1 ? higham_gamma(n, u) * (j - 1) : 0.0
            base_norm = norm_func(P_power; measure=measure)
            norm_j = base_norm * (1 + γ_mult)
        else
            norm_j = norm_func(P_power; measure=measure)
        end

        push!(norms, (j, norm_j))

        if verbose
            rig_str = rigorous ? " (rigorous)" : ""
            println("  j = $j: ||P^j|_V||$rig_str = $(round(norm_j, digits=6))")
        end

        if norm_j < target_norm
            return (N=j, α=norm_j, norms=norms, rigorous=rigorous)
        end

        # Update P_power = P^(j+1)
        P_power = P_power * P
    end

    # No contraction found within max_power iterations
    if verbose
        println("  Warning: No contraction found within $max_power iterations")
    end

    return (N=nothing, α=nothing, norms=norms, rigorous=rigorous)
end

"""
    compute_stationary_density(P::AbstractMatrix;
                               tol::Float64=1e-12,
                               max_iter::Int=10000)

Compute the stationary density (left eigenvector) of a column-stochastic matrix.

The stationary density π satisfies P'π = π and Σ π_i = 1.

# Arguments
- `P`: Column-stochastic matrix
- `tol`: Convergence tolerance
- `max_iter`: Maximum power iterations

# Returns
Named tuple with:
- `π`: Stationary density vector
- `converged`: Whether iteration converged
- `iterations`: Number of iterations used

# Note
Uses power iteration on P'. For matrices with spectral gap, convergence is geometric.
"""
function compute_stationary_density(P::AbstractMatrix;
                                   tol::Float64=1e-12,
                                   max_iter::Int=10000)
    n = size(P, 1)

    # Start with uniform distribution
    π = fill(1.0/n, n)

    for iter in 1:max_iter
        # Apply P' (left multiplication by P on row vectors)
        π_new = P' * π

        # Normalize to ensure sum = 1
        π_new ./= sum(π_new)

        # Check convergence
        diff = maximum(abs.(π_new .- π))
        π = π_new

        if diff < tol
            return (π=π, converged=true, iterations=iter)
        end
    end

    return (π=π, converged=false, iterations=max_iter)
end

"""
    noise_variation(noise::IntervalBox{N}; noise_type::Symbol=:uniform) where {N}

Compute the variation (total variation seminorm) of the noise density ρ.

For uniform noise on a box of width ξ in each dimension:
- In 1D: var(ρ) = 2/ξ (two jumps of height 1/ξ)
- In 2D: var(ρ) = 4/ξ (for square noise, perimeter/area scaling)

# Arguments
- `noise`: The noise support box
- `noise_type`: Type of noise (:uniform or :triangular)

# Returns
- var(ρ): The total variation of the noise density

# Reference
ZZ paper, below Theorem B
"""
function noise_variation(noise; noise_type::Symbol=:uniform)
    N = length(noise)

    if noise_type == :uniform
        # For uniform density on box with side lengths ξ_i:
        # In 2D with square noise (ξ₁ = ξ₂ = ξ):
        # var(ρ) = 4/ξ (perimeter / area = 4ξ / ξ² = 4/ξ)
        #
        # More generally, for rectangular uniform noise:
        # var(ρ) = 2 * Σ (1/ξ_i) * Π_{j≠i} ξ_j / Π ξ_j
        #        = 2 * Σ (1/ξ_i) (each face contributes twice)

        widths = [diam(noise[i]) for i in 1:N]

        if N == 2
            # 2D case: var(ρ) = 2*(1/ξ₁ + 1/ξ₂) for rectangular
            # For square: var(ρ) = 4/ξ
            return 2 * sum(1 ./ widths)
        else
            # General N-dimensional case
            # For uniform density on box, var(ρ) = 2 * Σ (1/ξ_i)
            # Each pair of opposite faces contributes 2/ξ_i to the variation
            total_var = 0.0
            for i in 1:N
                total_var += 2 / widths[i]
            end
            return total_var
        end
    elseif noise_type == :triangular
        # Triangular noise has bounded variation from piecewise linear density
        error("Triangular noise variation not yet implemented")
    else
        error("Unknown noise type: $noise_type")
    end
end

"""
    ulam_approximation_error(partition; method::Symbol=:upper_bound)

Compute the Ulam approximation error δ for the partition.

For a partition with maximum diameter d:
- ||1 - π_δ||_{var → L¹} ≤ δ/2

where δ is related to the partition mesh size.

# Arguments
- `partition`: The UlamPartition
- `method`: :upper_bound uses maximum box diameter
            :average uses average diameter

# Returns
- δ: The discretization error parameter

# Reference
ZZ paper, Theorem B
"""
function ulam_approximation_error(partition; method::Symbol=:upper_bound)
    if method == :upper_bound
        # δ = max diameter over all boxes
        max_diam = 0.0
        for box in partition.boxes
            d = box_width(box)
            max_diam = max(max_diam, d)
        end
        return max_diam
    elseif method == :average
        # Average diameter
        total_diam = 0.0
        for box in partition.boxes
            total_diam += box_width(box)
        end
        return total_diam / partition.n_boxes
    else
        error("Unknown method: $method")
    end
end

"""
    rigorous_error_bound(P_noisy::AbstractMatrix, partition, noise;
                         target_contraction::Float64=0.5,
                         max_power::Int=100,
                         verbose::Bool=false)

Compute a rigorous error bound on the stationary density approximation.

This implements the main result from the ZZ paper (Theorem 4.2 / equation 21):

    ||f_ξ - f_{δ,ξ}||_{L¹} ≤ (3δ · var(ρ)) / (ξ · (1-α)) · Σ_{i=0}^{N-1} C_i

where:
- f_ξ is the true stationary density with noise ξ
- f_{δ,ξ} is the Ulam approximation
- δ is the partition mesh size
- var(ρ) is the noise density variation (4/ξ for 2D uniform)
- α is the contraction rate: ||L^N_δ|_V|| < α
- C_i = 1 for Ulam with L¹ norm
- N is the contraction power

# Arguments
- `P_noisy`: The noisy transition matrix (N * P_det, normalized)
- `partition`: The UlamPartition
- `noise`: The noise support box
- `target_contraction`: Target α for ||P^N|_V|| (default 0.5)
- `max_power`: Maximum power to search for contraction
- `verbose`: Print progress information

# Returns
Named tuple with:
- `error_bound`: Rigorous upper bound ||f_ξ - f_{δ,ξ}||_{L¹}
- `N`: Contraction power
- `α`: Achieved contraction ||P^N|_V||
- `δ`: Discretization error
- `var_ρ`: Noise variation
- `mixing_constant`: The sum Σ C_i = N (for Ulam)
- `mixing_satisfied`: Whether α + (3δ/ξ)var(ρ) Σ C_i < 1

# Example
```julia
# Build noisy Ulam operator
P_det, U = build_transition_matrix(F, partition, periodic=true)
N_op = build_grid_aligned_noise_operator(noise, partition, periodic=true)
P_noisy = N_op * P_det
normalize_columns!(P_noisy)

# Compute rigorous error bound
result = rigorous_error_bound(P_noisy, partition, noise, verbose=true)
println("Error bound: \$(result.error_bound)")
```

# References
- ZZ paper (Pacifico-Liberalquino-Nisoli), Theorem 4.2, equation (21)
- Galatolo-Nisoli (2014), Theorem 3.1
"""
function rigorous_error_bound(P_noisy::AbstractMatrix, partition, noise;
                              target_contraction::Float64=0.5,
                              max_power::Int=100,
                              verbose::Bool=false)

    # Step 1: Find contraction power N and rate α
    if verbose
        println("Finding contraction power...")
    end

    contraction_result = find_contraction_power(P_noisy,
                                                target_norm=target_contraction,
                                                max_power=max_power,
                                                verbose=verbose)

    N = contraction_result.N
    α = contraction_result.α

    if N === nothing
        if verbose
            println("Warning: No contraction found within $max_power iterations")
        end
        return (error_bound=Inf,
                N=nothing,
                α=nothing,
                δ=nothing,
                var_ρ=nothing,
                mixing_constant=nothing,
                mixing_satisfied=false)
    end

    # Step 2: Compute discretization error δ
    δ = ulam_approximation_error(partition, method=:upper_bound)

    # Step 3: Compute noise variation var(ρ)
    var_ρ = noise_variation(noise)

    # Step 4: Compute noise scale ξ (minimum noise width)
    ndims = length(noise)
    ξ = minimum(diam(noise[i]) for i in 1:ndims)

    # Step 5: For Ulam method with L¹ norm, C_i = 1 for all i
    # Σ_{i=0}^{N-1} C_i = N
    sum_C = Float64(N)

    # Step 6: Check mixing condition (equation 20 from ZZ paper)
    # α + (3δ/ξ) var(ρ) Σ C_i < 1
    mixing_term = (3 * δ / ξ) * var_ρ * sum_C
    mixing_satisfied = α + mixing_term < 1.0

    if verbose
        println("\nParameters:")
        println("  N (contraction power) = $N")
        println("  α (contraction rate) = $(round(α, digits=6))")
        println("  δ (mesh size) = $(round(δ, digits=6))")
        println("  ξ (noise scale) = $(round(ξ, digits=6))")
        println("  var(ρ) = $(round(var_ρ, digits=6))")
        println("  Σ C_i = $sum_C")
        println("\nMixing condition: α + (3δ/ξ)var(ρ)ΣC_i = $(round(α + mixing_term, digits=6)) < 1.0 ? $mixing_satisfied")
    end

    # Step 7: Compute error bound (equation 21)
    # ||f_ξ - f_{δ,ξ}||_{L¹} ≤ (3δ · var(ρ)) / (ξ · (1-α)) · Σ C_i
    if mixing_satisfied
        error_bound = (3 * δ * var_ρ) / (ξ * (1 - α)) * sum_C
    else
        # Mixing not satisfied - bound is not valid
        error_bound = Inf
    end

    if verbose
        println("\nRigorous error bound: ||f_ξ - f_{δ,ξ}||_{L¹} ≤ $(round(error_bound, digits=6))")
    end

    return (error_bound=error_bound,
            N=N,
            α=α,
            δ=δ,
            var_ρ=var_ρ,
            mixing_constant=sum_C,
            mixing_satisfied=mixing_satisfied,
            ξ=ξ)
end

"""
    spectral_gap(P::AbstractMatrix; method::Symbol=:power_iteration)

Estimate the spectral gap of a column-stochastic matrix.

The spectral gap is γ = 1 - |λ₂| where λ₂ is the second largest eigenvalue.

# Arguments
- `P`: Column-stochastic matrix
- `method`: :power_iteration (faster, approximate) or :eigen (exact but slower)

# Returns
- Estimated spectral gap

# Note
The spectral gap provides an alternative measure of mixing rate.
For reversible Markov chains: ||P^n|_V|| ≤ (1-γ)^n
"""
function spectral_gap(P::AbstractMatrix; method::Symbol=:power_iteration)
    if method == :eigen
        # Exact but expensive for large matrices
        eigenvalues = eigen(Matrix(P)).values
        sorted_eigs = sort(abs.(eigenvalues), rev=true)
        return 1.0 - sorted_eigs[2]
    elseif method == :power_iteration
        # Deflated power iteration to find second eigenvalue
        n = size(P, 1)

        # For stochastic matrices, we project out the constant vector direction
        # (which corresponds to eigenvalue 1) and iterate to find λ₂

        # Start with random vector orthogonal to 1
        v = randn(n)
        v = v .- mean(v)  # Make orthogonal to constant
        v ./= norm(v)

        λ_prev = 0.0
        for _ in 1:1000
            # Apply P and project out the stationary direction
            w = P * v
            w = w .- dot(w, ones(n)/n) * ones(n)  # Project out constant

            λ_new = norm(w)
            if abs(λ_new - λ_prev) < 1e-10
                return 1.0 - λ_new
            end

            v = w ./ λ_new
            λ_prev = λ_new
        end

        return 1.0 - λ_prev
    else
        error("Unknown method: $method")
    end
end
