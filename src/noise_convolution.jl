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
export uniform_noise_box, anisotropic_noise_box
export build_noise_convolution_operator, build_topological_graph
export build_grid_aligned_noise_operator

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

# DEPRECATED: noisy_rel_size has been removed.
# This function computed the annealed operator (F followed by noise), which violates
# the operator decomposition architecture where dynamics F and noise N are separate.
#
# Use the operator composition architecture instead:
#
#   # Build deterministic transition matrix
#   P, U = build_transition_matrix(F, partition, periodic=false, transition_graph=graph)
#
#   # Build noise convolution operator
#   N = build_noise_convolution_operator(noise, partition, periodic=false, transition_graph=graph)
#
#   # Compose to get noisy transition operator
#   P_noisy = N * P
#
# This follows the operator decomposition: P_noisy = N ∘ P
# where N is the noise convolution operator and P is the deterministic transition operator.

"""
    compute_noise_overlap(source_box::IntervalBox{N}, target_box::IntervalBox{N},
                         noise::IntervalBox{N}, domain::IntervalBox{N},
                         periodic::Bool; ε::Float64=1e-6, max_iterations::Int=20) where {N}

Rigorously compute the fraction of source_box that, after adding uniform noise, lands in target_box.

For uniform noise ξ ∼ Uniform(noise), this uses adaptive refinement to compute rigorous
lower and upper bounds on the probability that x + ξ ∈ target_box for x ∈ source_box.

The algorithm subdivides source_box until:
- Lower bound: All sub-boxes whose entire noisy image is contained in target
- Upper bound: All sub-boxes whose noisy image intersects target
- The difference (upper - lower) becomes negligible or max_iterations is reached

# Arguments
- `source_box`: The box from which mass originates
- `target_box`: The box to which mass may transition
- `noise`: The noise support (uniform distribution)
- `domain`: The domain (used for periodic BC)
- `periodic`: If true, wrap noisy box to torus
- `ε`: Tolerance for subdivision (stop when box width < ε)
- `max_iterations`: Maximum number of bisection iterations

# Returns
Fraction in [0, 1] representing the (rigorous lower bound on) overlap probability.

# Note
This returns the LOWER BOUND for verified transitions. The difference between upper and
lower bounds represents epistemic uncertainty from the interval arithmetic enclosure.
"""
function compute_noise_overlap(source_box::IntervalBox{N}, target_box::IntervalBox{N},
                               noise::IntervalBox{N}, domain::IntervalBox{N},
                               periodic::Bool; ε::Float64=1e-6, max_iterations::Int=20) where {N}

    # Base case: empty source
    source_measure = box_measure(source_box)
    if source_measure == 0
        return 0.0
    end

    # Compute noisy image enclosure
    noisy_box = minkowski_sum(source_box, noise)
    if periodic
        noisy_box = apply_periodic_bc(noisy_box, domain)
    end

    # Case 1: Entire noisy image contained in target (verified transition)
    if issubset_box(noisy_box, target_box)
        return 1.0  # All mass from source lands in target
    end

    # Case 2: Noisy image disjoint from target (no transition)
    if isdisjoint_box(noisy_box, target_box)
        return 0.0
    end

    # Case 3: Uncertain - need adaptive refinement
    # Use a simple iterative subdivision approach
    total_lower = 0.0
    uncertain_boxes = [source_box]

    for _ in 1:max_iterations
        if isempty(uncertain_boxes)
            break
        end

        # Stop if boxes are too small
        if all(box_width(box) < ε for box in uncertain_boxes)
            break
        end

        new_uncertain = IntervalBox{N}[]

        for box in uncertain_boxes
            # Subdivide the box
            left, right = bisect_box(box)

            for sub_box in [left, right]
                sub_measure = box_measure(sub_box)

                # Compute noisy image of sub-box
                sub_noisy = minkowski_sum(sub_box, noise)
                if periodic
                    sub_noisy = apply_periodic_bc(sub_noisy, domain)
                end

                # Check containment
                if issubset_box(sub_noisy, target_box)
                    # Verified transition - add to lower bound
                    total_lower += sub_measure / source_measure
                elseif !isdisjoint_box(sub_noisy, target_box)
                    # Still uncertain - keep for next iteration
                    push!(new_uncertain, sub_box)
                end
                # If disjoint, we skip (contributes 0)
            end
        end

        uncertain_boxes = new_uncertain
    end

    # Return lower bound (verified transitions only)
    return total_lower
end

"""
    build_topological_graph(partition::UlamPartition{N}, noise::IntervalBox{N};
                           periodic::Bool=false,
                           dynamical_graph::Union{SparseMatrixCSC{Bool},Nothing}=nothing) where {N}

Build a sparse boolean matrix indicating which partition boxes can overlap after
adding noise. This is a **topological** connectivity graph based on distance.

For noise convolution, box i can receive mass from box j if their Minkowski sum
overlaps: `box_j ⊕ noise` intersects `box_i`.

If a dynamical_graph is provided, the result will be the union of topological and
dynamical connections. This is because noise can spread mass both locally (topology)
and following the dynamics.

# Arguments
- `partition`: The partition of phase space
- `noise`: The noise support box
- `periodic`: If true, account for periodic boundary conditions
- `dynamical_graph`: Optional dynamical transition graph from build_adaptive_partition.
                     If provided, result includes both topological and dynamical connections.

# Returns
Sparse boolean matrix `graph` where `graph[i,j] = true` if boxes i and j are
connected (either topologically within noise distance, or dynamically via F, or both).

# Example
```julia
noise = uniform_noise_box(0.01, 2)
partition, dyn_graph = build_adaptive_partition(F, domain, noise, max_diameter=0.05)

# Build combined graph: topological + dynamical
noise_graph = build_topological_graph(partition, noise, dynamical_graph=dyn_graph)
N = build_noise_convolution_operator(noise, partition, transition_graph=noise_graph)
```

# Note
The combined graph ensures correct noise propagation: mass spreads both locally
(within noise distance) and follows the dynamics (F then adds noise).

# NOTE: As of version 0.2, build_adaptive_partition returns the topological graph directly.
# This standalone function is kept for backward compatibility but is DEPRECATED for adaptive partitions.
#
# Preferred usage:
#   partition, dyn_graph, topo_graph = build_adaptive_partition(F, domain, noise, ...)
#   N = build_noise_convolution_operator(noise, partition, transition_graph=topo_graph, ...)
#
# This function can still be used if you built the partition with build_uniform_partition:
#   partition = build_uniform_partition(domain, depth)
#   topo_graph = build_topological_graph(partition, noise, ...)
"""
function build_topological_graph(partition::UlamPartition{N}, noise::IntervalBox{N};
                                periodic::Bool=false,
                                dynamical_graph::Union{SparseMatrixCSC{Bool},Nothing}=nothing) where {N}
    n = partition.n_boxes
    I_indices = Int[]
    J_indices = Int[]

    for j in 1:n
        source_box = partition.boxes[j]

        # Compute the extent of source + noise
        noisy_source = minkowski_sum(source_box, noise)
        if periodic
            noisy_source = apply_periodic_bc(noisy_source, partition.domain)
        end

        # Find all boxes that could overlap with noisy source
        for i in 1:n
            target_box = partition.boxes[i]

            # Quick check: could the boxes possibly overlap?
            if !isdisjoint_box(noisy_source, target_box)
                push!(I_indices, i)
                push!(J_indices, j)
            end
        end
    end

    topo_graph = sparse(I_indices, J_indices, true, n, n)

    # If dynamical graph provided, take union (logical OR)
    if dynamical_graph !== nothing
        return topo_graph .| dynamical_graph
    else
        return topo_graph
    end
end

"""
    build_noise_convolution_operator(noise::IntervalBox{N}, partition::UlamPartition{N};
                                     periodic::Bool=false,
                                     transition_graph::Union{SparseMatrixCSC{Bool},Nothing}=nothing,
                                     ε::Float64=1e-6, max_iterations::Int=20) where {N}

Build the noise convolution operator N on a partition using rigorous adaptive refinement.

This operator represents the spreading of mass due to additive noise. Given a probability
distribution μ on the partition, N*μ gives the distribution after convolving with noise.

For uniform noise ξ ∼ Uniform(noise), the operator N[i,j] represents the **rigorously verified**
fraction of box j that, after adding noise ξ, lands in box i. The computation uses adaptive
subdivision to distinguish between verified transitions and uncertain ones.

# Arguments
- `noise`: Noise support box (e.g., from uniform_noise_box)
- `partition`: The partition of phase space
- `periodic`: If true, apply periodic boundary conditions (wrap to torus)
- `transition_graph`: Optional sparsity pattern. If provided, only computes entries
                      where boxes can overlap after noise addition.
- `ε`: Tolerance for adaptive refinement (stop when box width < ε)
- `max_iterations`: Maximum number of subdivision iterations per entry

# Returns
- `N`: Sparse noise convolution operator (n × n) with **rigorously verified lower bounds**

# Example
```julia
noise = uniform_noise_box(0.01, 2)
partition, graph = build_adaptive_partition(F, domain, noise, max_diameter=0.05, periodic=true)
N = build_noise_convolution_operator(noise, partition, periodic=true, transition_graph=graph,
                                     ε=1e-6, max_iterations=20)
```

# Note
- The noise operator is independent of the dynamics F. It only depends on the noise
  distribution and the partition geometry.
- Matrix entries are **lower bounds** (verified transitions only). This ensures rigorous
  approximations but may be conservative.
- Increase `max_iterations` or decrease `ε` for tighter bounds (more computation).

# Future Optimization: Grid-Aligned Noise
When noise size matches the partition grid (noise_side = domain_side / 2^k), the convolution
operator can be computed exactly without subdivision:
- Each box spreads mass uniformly to its topological neighbors
- No adaptive refinement needed - just follow topological graph edges
- Compute exact overlap fractions from geometry
- Would be orders of magnitude faster for grid-aligned noise
"""
function build_noise_convolution_operator(noise::IntervalBox{N}, partition::UlamPartition{N};
                                         periodic::Bool=false,
                                         transition_graph::Union{SparseMatrixCSC{Bool},Nothing}=nothing,
                                         ε::Float64=1e-6, max_iterations::Int=20) where {N}

    n = partition.n_boxes
    I_indices = Int[]
    J_indices = Int[]
    values = Float64[]

    if transition_graph !== nothing
        # Use graph to skip zero entries
        rows = rowvals(transition_graph)
        for j in 1:n
            source_box = partition.boxes[j]

            for idx in nzrange(transition_graph, j)
                i = rows[idx]
                target_box = partition.boxes[i]

                overlap = compute_noise_overlap(source_box, target_box, noise,
                                               partition.domain, periodic,
                                               ε=ε, max_iterations=max_iterations)

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

                overlap = compute_noise_overlap(source_box, target_box, noise,
                                               partition.domain, periodic,
                                               ε=ε, max_iterations=max_iterations)

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

"""
    build_grid_aligned_noise_operator(noise::IntervalBox{N}, partition::UlamPartition{N};
                                      periodic::Bool=false,
                                      topological_graph::Union{SparseMatrixCSC{Bool},Nothing}=nothing) where {N}

Build noise convolution operator for grid-aligned noise (FAST EXACT method).

This function computes the noise operator EXACTLY without subdivision when the noise
is aligned with the partition grid, i.e., when noise_width = domain_width / 2^k for some k.

For grid-aligned uniform noise, mass spreads uniformly to topological neighbors with
exact geometric overlap fractions that can be computed directly.

# Grid Alignment Requirement

For each dimension i, the noise must satisfy:
```
noise_width[i] = (domain_width[i] / 2^k[i]) for some non-negative integer k[i]
```

This ensures that noise boundaries align with partition box boundaries, allowing
exact computation of overlap fractions.

**Supports anisotropic noise**: Each dimension is checked independently, so you can use
`anisotropic_noise_box([1/32, 1/16])` for rectangular noise on a rectangular domain.

# Arguments
- `noise`: Noise support box (must be grid-aligned)
- `partition`: The partition of phase space
- `periodic`: If true, apply periodic boundary conditions
- `topological_graph`: Topological connectivity graph from build_adaptive_partition.
                       If not provided, will compute it (slower).

# Returns
- `N`: Exact noise convolution operator (n × n sparse matrix)

# Example
```julia
# Domain [0,1]² partitioned to depth 5 → boxes of size 1/32
domain = IntervalBox(interval(0.0, 1.0), interval(0.0, 1.0))
partition, dyn_graph, topo_graph = build_adaptive_partition(F, domain, nothing,
                                                            initial_depth=5,
                                                            refinement_levels=0)

# Grid-aligned noise: 1/32 = domain_width/2^5
noise = uniform_noise_box(1/64, 2)  # ±1/64 → width 1/32 ✓ grid-aligned

# Fast exact computation (no subdivision!)
N = build_grid_aligned_noise_operator(noise, partition,
                                      topological_graph=topo_graph)
```

# Performance
Orders of magnitude faster than adaptive refinement approach:
- No subdivision iterations
- Exact geometric computation
- Just follows topological graph edges

# Validation
The function validates grid alignment and errors if noise is not grid-aligned.
Use build_noise_convolution_operator for non-aligned noise.
"""
function build_grid_aligned_noise_operator(noise::IntervalBox{N}, partition::UlamPartition{N};
                                          periodic::Bool=false,
                                          topological_graph::Union{SparseMatrixCSC{Bool},Nothing}=nothing) where {N}

    # Validate grid alignment
    domain = partition.domain
    for i in 1:N
        noise_width = diam(noise[i])
        domain_width = diam(domain[i])

        # Check if noise_width = domain_width / 2^k for some k
        ratio = domain_width / noise_width
        k = round(log2(ratio))
        expected_width = domain_width / 2^k

        if abs(noise_width - expected_width) > 1e-10
            error("Noise is not grid-aligned in dimension $i: " *
                  "noise_width=$noise_width, expected=$(domain_width / 2^k) for k=$k")
        end
    end

    # Build topological graph if not provided
    if topological_graph === nothing
        topological_graph = build_topological_graph(partition, noise, periodic=periodic)
    end

    n = partition.n_boxes

    # Multithreaded: each thread accumulates its own COO triples
    nt = Threads.maxthreadid()
    thread_I = [Int[] for _ in 1:nt]
    thread_J = [Int[] for _ in 1:nt]
    thread_V = [Float64[] for _ in 1:nt]

    # For each source box, compute exact overlap with neighbors
    rows = rowvals(topological_graph)
    Threads.@threads :static for j in 1:n
        tid = Threads.threadid()
        source_box = partition.boxes[j]

        # Compute noisy box: source ⊕ noise
        noisy_box = minkowski_sum(source_box, noise)

        if periodic
            # For grid-aligned periodic noise, handle wraparound explicitly
            wraps_around = false
            for i in 1:N
                inf_wrapped = wrap_to_torus(inf(noisy_box[i]), inf(domain[i]), sup(domain[i]))
                sup_wrapped = wrap_to_torus(sup(noisy_box[i]), inf(domain[i]), sup(domain[i]))
                if inf_wrapped > sup_wrapped
                    wraps_around = true
                    break
                end
            end

            if wraps_around
                # Wraparound case: enumerate all wrapped components
                components = [noisy_box]
                for dim in 1:N
                    period = diam(domain[dim])
                    noisy_interval = noisy_box[dim]

                    if diam(noisy_interval) < period
                        inf_wrapped = wrap_to_torus(inf(noisy_interval), inf(domain[dim]), sup(domain[dim]))
                        sup_wrapped = wrap_to_torus(sup(noisy_interval), inf(domain[dim]), sup(domain[dim]))

                        if inf_wrapped > sup_wrapped
                            new_components = IntervalBox{N}[]
                            for comp in components
                                intervals1 = [comp[k] for k in 1:N]
                                intervals2 = [comp[k] for k in 1:N]
                                intervals1[dim] = interval(inf(domain[dim]), sup_wrapped)
                                intervals2[dim] = interval(inf_wrapped, sup(domain[dim]))
                                push!(new_components, IntervalBox(intervals1...))
                                push!(new_components, IntervalBox(intervals2...))
                            end
                            components = new_components
                        end
                    end
                end

                # Process each component separately
                # Duplicate (i,j) entries are summed by sparse() constructor
                noisy_volume = box_measure(noisy_box)
                for comp in components
                    for idx in nzrange(topological_graph, j)
                        i = rows[idx]
                        target_box = partition.boxes[i]

                        overlap_box = intersect_boxes(comp, target_box)
                        if !is_empty_box(overlap_box)
                            overlap_volume = box_measure(overlap_box)
                            fraction = overlap_volume / noisy_volume

                            if fraction > 0
                                push!(thread_I[tid], i)
                                push!(thread_J[tid], j)
                                push!(thread_V[tid], fraction)
                            end
                        end
                    end
                end
                continue  # Skip the non-periodic case
            else
                noisy_box = apply_periodic_bc(noisy_box, domain)
            end
        end

        # Non-periodic or non-wrapping case
        noisy_volume = box_measure(noisy_box)
        for idx in nzrange(topological_graph, j)
            i = rows[idx]
            target_box = partition.boxes[i]

            overlap_box = intersect_boxes(noisy_box, target_box)

            if !is_empty_box(overlap_box)
                overlap_volume = box_measure(overlap_box)
                fraction = overlap_volume / noisy_volume

                if fraction > 0
                    push!(thread_I[tid], i)
                    push!(thread_J[tid], j)
                    push!(thread_V[tid], fraction)
                end
            end
        end
    end

    # Reduce: concatenate thread-local triples
    # sparse() with + combiner handles duplicate (i,j) from periodic wraparound
    I_indices = reduce(vcat, thread_I)
    J_indices = reduce(vcat, thread_J)
    values = reduce(vcat, thread_V)

    N_op = sparse(I_indices, J_indices, values, n, n)
    return N_op
end
