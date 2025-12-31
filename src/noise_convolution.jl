"""
Noise convolution utilities for stochastic dynamical systems.

This module provides functions for handling additive noise in the rigorous
Ulam method, including:
- Minkowski sum of boxes (for rectangular/uniform noise)
- Convolution of transition matrices
- Noisy image enclosures
"""

using IntervalArithmetic, IntervalBoxes
using SparseArrays

export minkowski_sum, noisy_image
export convolve_transition_matrix, add_noise_to_partition

"""
    minkowski_sum(A::IntervalBox{N}, B::IntervalBox{N}) where {N}

Compute the Minkowski sum A ⊕ B = {a + b : a ∈ A, b ∈ B}.

For interval boxes, this is simply the sum of corresponding intervals:
(A ⊕ B)[i] = A[i] + B[i]

This operation is used to add rectangular/uniform noise to image enclosures.

# Example
```julia
box = IntervalBox(interval(0.0, 1.0), interval(0.0, 1.0))
noise = IntervalBox(interval(-0.1, 0.1), interval(-0.1, 0.1))
noisy_box = minkowski_sum(box, noise)
```
"""
function minkowski_sum(A::IntervalBox{N}, B::IntervalBox{N}) where {N}
    result_intervals = [A[i] + B[i] for i in 1:N]
    return IntervalBox(result_intervals...)
end

"""
    minkowski_sum(A::Interval, B::Interval)

Compute the Minkowski sum for individual intervals.
"""
minkowski_sum(A::Interval, B::Interval) = A + B

"""
    minkowski_sum(A::Tuple, B::IntervalBox{N}) where {N}

Compute Minkowski sum when A is a tuple of intervals (from enclosure functions).
"""
function minkowski_sum(A::Tuple, B::IntervalBox{N}) where {N}
    A_box = IntervalBox(A...)
    return minkowski_sum(A_box, B)
end

"""
    noisy_image(F, box::IntervalBox{N}, noise::IntervalBox{N};
                enclosure_method=first_order_enclosure) where {N}

Compute a rigorous enclosure of the image of `box` under the noisy map
F̃(x) = F(x) + ξ where ξ ∈ noise.

This is equivalent to computing F(box) ⊕ noise (Minkowski sum).

# Arguments
- `F`: The deterministic part of the map
- `box`: The source box
- `noise`: The noise box (support of the noise distribution)
- `enclosure_method`: Method for computing F(box)

# Returns
An IntervalBox enclosing all possible images under the noisy dynamics.

# Example
```julia
F(x, y) = (2*x*(1-x), 0.5*y)
box = IntervalBox(interval(0.0, 0.5), interval(0.0, 1.0))
noise = IntervalBox(interval(-0.01, 0.01), interval(-0.01, 0.01))
image = noisy_image(F, box, noise)
```
"""
function noisy_image(F, box::IntervalBox{N}, noise::IntervalBox{N};
                    enclosure_method=first_order_enclosure) where {N}

    # Compute deterministic image
    deterministic_image = enclosure_method(F, box)

    # Add noise via Minkowski sum
    return minkowski_sum(deterministic_image, noise)
end

"""
    noisy_rel_size(F, Q_source::IntervalBox{N}, Q_target::IntervalBox{N},
                   noise::IntervalBox{N}; kwargs...) where {N}

Compute the relative size for a system with additive noise.

This modifies the standard `rel_size` algorithm to account for noise by
using `noisy_image` instead of the deterministic enclosure.

See `rel_size` for additional keyword arguments.
"""
function noisy_rel_size(F, Q_source::IntervalBox{N}, Q_target::IntervalBox{N},
                       noise::IntervalBox{N};
                       enclosure_method=first_order_enclosure,
                       ε=1e-6, maxdepth=30, depth=0) where {N}

    # Base case: empty source
    if box_measure(Q_source) == 0
        return (0.0, 0.0, IntervalBox{N}[])
    end

    # Compute noisy image enclosure
    image_box = noisy_image(F, Q_source, noise; enclosure_method=enclosure_method)

    # Case 1: Image definitely contained in target
    if issubset_box(image_box, Q_target)
        measure = box_measure(Q_source)
        return (measure, measure, IntervalBox{N}[])
    end

    # Case 2: Image definitely disjoint from target
    if isdisjoint_box(image_box, Q_target)
        return (0.0, 0.0, IntervalBox{N}[])
    end

    # Case 3: Uncertain - need to refine
    if depth >= maxdepth || box_width(Q_source) < ε
        measure = box_measure(Q_source)
        return (0.0, measure, [Q_source])
    end

    # Bisect and recurse
    if N == 2
        sub_boxes = quadrisect(Q_source)
    else
        sub_boxes = collect(bisect_box(Q_source))
    end

    total_lower = 0.0
    total_upper = 0.0
    all_uncertain = IntervalBox{N}[]

    for sub_box in sub_boxes
        lower, upper, uncertain = noisy_rel_size(F, sub_box, Q_target, noise;
                                                 enclosure_method=enclosure_method,
                                                 ε=ε, maxdepth=maxdepth, depth=depth+1)
        total_lower += lower
        total_upper += upper
        append!(all_uncertain, uncertain)
    end

    return (total_lower, total_upper, all_uncertain)
end

"""
    build_noisy_transition_matrix(F, partition::UlamPartition{N},
                                  noise::IntervalBox{N}; kwargs...) where {N}

Build a transition matrix for a system with additive noise.

This is similar to `build_transition_matrix` but uses `noisy_rel_size`
to account for the noise in the dynamics.

# Arguments
- `F`: Deterministic part of the map
- `partition`: Phase space partition
- `noise`: Noise box (support of uniform noise distribution)
- Additional kwargs passed to `noisy_rel_size`

# Returns
- `P`: Sparse transition matrix
- `U`: Uncertainty matrix

# Example
```julia
F(x, y) = (2*x*(1-x), 0.5*y)
partition = UlamPartition(boxes, domain)
noise = IntervalBox(interval(-0.01, 0.01), interval(-0.01, 0.01))
P, U = build_noisy_transition_matrix(F, partition, noise)
```
"""
function build_noisy_transition_matrix(F, partition::UlamPartition{N},
                                      noise::IntervalBox{N};
                                      enclosure_method=first_order_enclosure,
                                      ε=1e-6, maxdepth=30,
                                      use_lower_bound=true) where {N}

    n = partition.n_boxes
    I_indices = Int[]
    J_indices = Int[]
    values = Float64[]
    uncertainties_I = Int[]
    uncertainties_J = Int[]
    uncertainties_vals = Float64[]

    for j in 1:n
        Q_source = partition.boxes[j]
        source_measure = box_measure(Q_source)

        if source_measure == 0
            continue
        end

        for i in 1:n
            Q_target = partition.boxes[i]

            lower, upper, _ = noisy_rel_size(F, Q_source, Q_target, noise;
                                            enclosure_method=enclosure_method,
                                            ε=ε, maxdepth=maxdepth)

            transition_measure = use_lower_bound ? lower : upper
            uncertainty = upper - lower

            if transition_measure > 0
                push!(I_indices, i)
                push!(J_indices, j)
                push!(values, transition_measure / source_measure)
            end

            if uncertainty > 0
                push!(uncertainties_I, i)
                push!(uncertainties_J, j)
                push!(uncertainties_vals, uncertainty / source_measure)
            end
        end
    end

    P = sparse(I_indices, J_indices, values, n, n)
    U = sparse(uncertainties_I, uncertainties_J, uncertainties_vals, n, n)

    return P, U
end

"""
    convolve_transition_matrix(P::AbstractMatrix, noise_kernel::AbstractMatrix)

Convolve a transition matrix with a noise kernel.

This is an alternative approach to handling noise: instead of computing
noisy images directly, first compute the deterministic transition matrix,
then convolve with a discretized noise distribution.

# Arguments
- `P`: Deterministic transition matrix (n × n)
- `noise_kernel`: Discretized noise distribution (m × m, typically small)

# Returns
A convolved transition matrix accounting for the noise.

# Note
This method is more efficient when the noise is small and can be
represented on a coarse grid. For larger noise, use `build_noisy_transition_matrix`.
"""
function convolve_transition_matrix(P::AbstractMatrix, noise_kernel::AbstractMatrix)
    # TODO: Implement proper convolution for sparse matrices
    # This is a placeholder for future implementation
    error("Convolution method not yet implemented. Use build_noisy_transition_matrix instead.")
end

"""
    uniform_noise_box(noise_level::Real, N::Int)

Create a symmetric uniform noise box [-ε, ε]^N.

# Arguments
- `noise_level`: Half-width of the noise interval (ε)
- `N`: Dimension of the phase space

# Returns
An N-dimensional IntervalBox representing uniform noise.

# Example
```julia
noise = uniform_noise_box(0.01, 2)  # 2D noise with ±0.01 in each direction
```
"""
function uniform_noise_box(noise_level::Real, N::Int)
    noise_interval = interval(-noise_level, noise_level)
    return IntervalBox([noise_interval for _ in 1:N]...)
end

"""
    anisotropic_noise_box(noise_levels::Vector{<:Real})

Create an anisotropic noise box with different noise levels per dimension.

# Arguments
- `noise_levels`: Vector of half-widths for each dimension

# Example
```julia
noise = anisotropic_noise_box([0.01, 0.005])  # Different noise in x and y
```
"""
function anisotropic_noise_box(noise_levels::Vector{<:Real})
    N = length(noise_levels)
    intervals = [interval(-ε, ε) for ε in noise_levels]
    return IntervalBox(intervals...)
end
