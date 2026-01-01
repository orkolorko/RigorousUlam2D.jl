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

export minkowski_sum, noisy_image, noisy_rel_size
export uniform_noise_box, anisotropic_noise_box
export build_noise_convolution_operator

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
    left, right = bisect_box(Q_source)
    sub_boxes = [left, right]

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
    build_noise_convolution_operator(noise::IntervalBox{N}, partition::UlamPartition{N};
                                     periodic::Bool=false,
                                     transition_graph::Union{SparseMatrixCSC{Bool},Nothing}=nothing) where {N}

Build the noise convolution operator N on a partition.

This operator represents the spreading of mass due to additive noise. Given a probability
distribution μ on the partition, N*μ gives the distribution after convolving with noise.

For uniform noise ξ ∼ Uniform(noise), the operator N[i,j] represents the fraction of
box j that, after adding noise ξ, lands in box i.

# Arguments
- `noise`: Noise support box (e.g., from uniform_noise_box)
- `partition`: The partition of phase space
- `periodic`: If true, apply periodic boundary conditions (wrap to torus)
- `transition_graph`: Optional sparsity pattern. If provided, only computes entries
                      where boxes can overlap after noise addition.

# Returns
- `N`: Sparse noise convolution operator (n × n)

# Example
```julia
noise = uniform_noise_box(0.01, 2)
partition, graph = build_adaptive_partition(F, domain, noise, max_diameter=0.05, periodic=true)
N = build_noise_convolution_operator(noise, partition, periodic=true, transition_graph=graph)
```

# Note
The noise operator is independent of the dynamics F. It only depends on the noise
distribution and the partition geometry.
"""
function build_noise_convolution_operator(noise::IntervalBox{N}, partition::UlamPartition{N};
                                         periodic::Bool=false,
                                         transition_graph::Union{SparseMatrixCSC{Bool},Nothing}=nothing) where {N}

    n = partition.n_boxes
    I_indices = Int[]
    J_indices = Int[]
    values = Float64[]

    # Helper: compute fraction of source_box + noise that intersects target_box
    function noise_overlap(source_box::IntervalBox{N}, target_box::IntervalBox{N})
        # The noisy box is source_box ⊕ noise (Minkowski sum)
        noisy_box = minkowski_sum(source_box, noise)

        # Apply boundary conditions
        if periodic
            noisy_box = apply_periodic_bc(noisy_box, partition.domain)
        end

        # Compute intersection with target
        intersection = intersect_boxes(noisy_box, target_box)

        if is_empty_box(intersection)
            return 0.0
        end

        # Fraction = measure(intersection) / measure(noisy_box)
        return box_measure(intersection) / box_measure(noisy_box)
    end

    if transition_graph !== nothing
        # Use graph to skip zero entries
        rows = rowvals(transition_graph)
        for j in 1:n
            source_box = partition.boxes[j]

            for idx in nzrange(transition_graph, j)
                i = rows[idx]
                target_box = partition.boxes[i]

                overlap = noise_overlap(source_box, target_box)

                if overlap > 0
                    push!(I_indices, i)
                    push!(J_indices, j)
                    push!(values, overlap)
                end
            end
        end
    else
        # No graph: check all pairs
        for j in 1:n
            source_box = partition.boxes[j]

            for i in 1:n
                target_box = partition.boxes[i]

                overlap = noise_overlap(source_box, target_box)

                if overlap > 0
                    push!(I_indices, i)
                    push!(J_indices, j)
                    push!(values, overlap)
                end
            end
        end
    end

    N_op = sparse(I_indices, J_indices, values, n, n)
    return N_op
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
