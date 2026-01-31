"""
Sparse matrix construction for the rigorous Ulam method.

This module provides functions to construct sparse transition matrices
for Markov approximations of dynamical systems using rigorous enclosures.
"""

using IntervalArithmetic, IntervalBoxes
using SparseArrays
using LinearAlgebra

export build_transition_matrix, normalize_columns!
export UlamPartition, partition_index

"""
    UlamPartition{N}

Structure to manage a partition of the phase space for the Ulam method.

# Fields
- `boxes::Vector{IntervalBox{N}}`: The partition elements
- `domain::IntervalBox{N}`: The overall domain
- `n_boxes::Int`: Number of partition elements

# Constructor
    UlamPartition(boxes::Vector{IntervalBox{N}}, domain::IntervalBox{N})
"""
struct UlamPartition{N,T,I}
    boxes::Vector{IntervalBox{N,T,I}}
    domain::IntervalBox{N,T,I}
    n_boxes::Int

    function UlamPartition(boxes::Vector{IntervalBox{N,T,I}}, domain::IntervalBox{N,T,I}) where {N,T,I}
        new{N,T,I}(boxes, domain, length(boxes))
    end
end

"""
    partition_index(partition::UlamPartition, point::IntervalBox)

Find the index of the partition element containing the given point (box).
Returns the index of the first box containing the point, or nothing if not found.

For rigorous computations, this checks if the point is a subset of any partition element.
"""
function partition_index(partition::UlamPartition{N}, point::IntervalBox{N}) where {N}
    for (i, box) in enumerate(partition.boxes)
        if issubset_box(point, box)
            return i
        end
    end
    return nothing
end

"""
    build_transition_matrix(F, partition::UlamPartition{N},
                           noise::Union{IntervalBox{N}, Nothing}=nothing;
                           enclosure_method=first_order_enclosure,
                           ε=1e-6, maxdepth=30,
                           use_lower_bound=true,
                           periodic::Bool=false,
                           transition_graph::Union{SparseMatrixCSC{Bool},Nothing}=nothing) where {N}

Construct a sparse transition matrix for the Ulam approximation of the map F.

This unified function handles both deterministic and noisy systems, with optional
sparsity pattern optimization via a transition graph.

Returns a sparse matrix P where P[i,j] represents the (rigorous) transition
probability from partition element j to partition element i.

# Arguments
- `F`: The dynamical system (function mapping R^N → R^N)
- `partition`: The partition of the phase space
- `noise`: Noise support box (optional, for stochastic systems)
- `enclosure_method`: Method for computing rigorous image enclosures
- `ε`: Tolerance for adaptive refinement (box diameter threshold)
- `max_iterations`: Maximum iterations for adaptive refinement
- `use_lower_bound`: If true, use lower bounds (verified transitions only)
                     If false, use upper bounds (includes uncertainties)
- `periodic`: If true, apply periodic boundary conditions (wrap images to torus)
- `transition_graph`: Optional boolean sparsity pattern from build_adaptive_partition.
                      If provided, only computes rel_size for potential transitions,
                      dramatically reducing computation time.

# Returns
- `P`: Sparse transition matrix (n_boxes × n_boxes)
- `U`: Uncertainty matrix (upper - lower bounds)

# Examples
```julia
# Deterministic system with absorbing BC
F(x, y) = (2*x*(1-x), 0.5*y)
partition = build_uniform_partition(domain, 4)
P, U = build_transition_matrix(F, partition)

# Noisy system with periodic BC
partition, graph = build_adaptive_partition(F, domain, noise, max_diameter=0.05, periodic=true)
P, U = build_transition_matrix(F, partition, noise, periodic=true, transition_graph=graph)
```

# Note
The matrix is constructed in column-major order (transitions FROM column index).
Use `normalize_columns!` to convert to a column-stochastic matrix.
"""
function build_transition_matrix(F, partition::UlamPartition{N},
                                noise::Union{IntervalBox{N}, Nothing}=nothing;
                                enclosure_method=first_order_enclosure,
                                ε=1e-6, max_iterations=30,
                                use_lower_bound=true,
                                periodic::Bool=false,
                                transition_graph::Union{SparseMatrixCSC{Bool},Nothing}=nothing) where {N}

    n = partition.n_boxes
    I_indices = Int[]
    J_indices = Int[]
    values = Float64[]
    uncertainties_I = Int[]
    uncertainties_J = Int[]
    uncertainties_vals = Float64[]

    # Check for deprecated usage: noise should be applied via operator composition
    if noise !== nothing
        error("""
            Passing noise to build_transition_matrix is deprecated.

            Use the operator composition architecture instead:
              # Build deterministic transition matrix
              P, U = build_transition_matrix(F, partition, periodic=$periodic, transition_graph=graph)

              # Build noise convolution operator
              N = build_noise_convolution_operator(noise, partition, periodic=$periodic, transition_graph=graph)

              # Compose to get noisy transition operator
              P_noisy = N * P

            This follows the operator decomposition: P_noisy = N ∘ P
            where N is the noise convolution operator and P is the deterministic transition operator.
            """)
    end

    # Select appropriate rel_size function based on boundary conditions
    compute_rel_size = if periodic
        # Deterministic with periodic BC - wrap images to torus
        # Uses subdivision for accurate overlap computation
        (F, Q_source, Q_target) -> begin
            source_measure = box_measure(Q_source)
            if source_measure == 0
                return (0.0, 0.0, IntervalBox{N}[])
            end

            # Initialize accumulators
            lower_measure = 0.0

            # Process boxes via simple subdivision
            to_process = [Q_source]
            max_depth = max_iterations

            for _ in 1:max_depth
                if isempty(to_process)
                    break
                end

                next_round = IntervalBox{N}[]

                for box in to_process
                    box_vol = box_measure(box)

                    # Compute image and wrap
                    img = enclosure_method(F, box)
                    image_box = img isa Tuple ? IntervalBox(img...) : img
                    wrapped = apply_periodic_bc(image_box, partition.domain)

                    # Check containment
                    if issubset_box(wrapped, Q_target)
                        lower_measure += box_vol
                    elseif isdisjoint_box(wrapped, Q_target)
                        # Discard - no contribution
                    else
                        # Partial overlap - check if we can compute geometric overlap
                        is_full_domain = all(
                            abs(diam(wrapped[i]) - diam(partition.domain[i])) < 1e-10
                            for i in 1:N
                        )

                        if is_full_domain && box_width(box) > ε
                            # Full domain fallback - need to subdivide
                            left, right = bisect_box(box)
                            push!(next_round, left)
                            push!(next_round, right)
                        elseif is_full_domain
                            # Box too small but still full domain - use geometric estimate
                            # Mass distributes over entire domain proportionally
                            overlap_fraction = box_measure(Q_target) / box_measure(partition.domain)
                            lower_measure += box_vol * overlap_fraction
                        else
                            # Wrapped box is accurate, compute geometric overlap
                            intersection = intersect_boxes(wrapped, Q_target)
                            if !is_empty_box(intersection)
                                overlap_fraction = box_measure(intersection) / box_measure(wrapped)
                                lower_measure += box_vol * overlap_fraction
                            end
                        end
                    end
                end

                to_process = next_round
            end

            # Any remaining boxes are uncertain
            uncertain_measure = sum(box_measure(box) for box in to_process; init=0.0)

            lower_rel = lower_measure / source_measure
            upper_rel = (lower_measure + uncertain_measure) / source_measure

            return (lower_rel, upper_rel, to_process)
        end
    else
        # Absorbing boundary conditions (default)
        (F, Q_source, Q_target) -> rel_size(F, Q_source, Q_target;
                                            enclosure_method=enclosure_method,
                                            ε=ε, max_iterations=max_iterations)
    end

    # If transition graph is provided, use it to skip zero entries
    if transition_graph !== nothing
        rows = rowvals(transition_graph)
        for j in 1:n
            Q_source = partition.boxes[j]
            source_measure = box_measure(Q_source)

            if source_measure == 0
                continue
            end

            # Only check boxes that can potentially receive mass from j
            for idx in nzrange(transition_graph, j)
                i = rows[idx]
                Q_target = partition.boxes[i]

                lower, upper, _ = compute_rel_size(F, Q_source, Q_target)

                transition_measure = use_lower_bound ? lower : upper
                uncertainty = upper - lower

                if transition_measure > 0
                    push!(I_indices, i)
                    push!(J_indices, j)
                    # rel_size already returns relative measures (fractions in [0,1])
                    # so we multiply by source_measure to get absolute measures
                    push!(values, transition_measure * source_measure)
                end

                if uncertainty > 0
                    push!(uncertainties_I, i)
                    push!(uncertainties_J, j)
                    push!(uncertainties_vals, uncertainty * source_measure)
                end
            end
        end
    else
        # No graph: check all pairs (original behavior)
        for j in 1:n
            Q_source = partition.boxes[j]
            source_measure = box_measure(Q_source)

            if source_measure == 0
                continue
            end

            for i in 1:n
                Q_target = partition.boxes[i]

                lower, upper, _ = compute_rel_size(F, Q_source, Q_target)

                transition_measure = use_lower_bound ? lower : upper
                uncertainty = upper - lower

                if transition_measure > 0
                    push!(I_indices, i)
                    push!(J_indices, j)
                    # rel_size already returns relative measures (fractions in [0,1])
                    # so we multiply by source_measure to get absolute measures
                    push!(values, transition_measure * source_measure)
                end

                if uncertainty > 0
                    push!(uncertainties_I, i)
                    push!(uncertainties_J, j)
                    push!(uncertainties_vals, uncertainty * source_measure)
                end
            end
        end
    end

    # Create sparse matrices
    P = sparse(I_indices, J_indices, values, n, n)
    U = sparse(uncertainties_I, uncertainties_J, uncertainties_vals, n, n)

    return P, U
end

"""
    normalize_columns!(P::SparseMatrixCSC)

Normalize each column of the sparse matrix P to sum to 1.
This converts the transition measure matrix to a column-stochastic matrix.

Modifies P in-place and returns the column sums before normalization.
"""
function normalize_columns!(P::SparseMatrixCSC)
    n = size(P, 2)
    col_sums = zeros(n)

    # Compute column sums
    for j in 1:n
        col_sums[j] = sum(P[:, j])
    end

    # Normalize non-zero columns
    for j in 1:n
        if col_sums[j] > 0
            P[:, j] ./= col_sums[j]
        end
    end

    return col_sums
end

"""
    build_normalized_transition_matrix(F, partition::UlamPartition{N}; kwargs...) where {N}

Build and normalize a transition matrix in one step.

Returns `(P, U, col_sums)` where:
- `P`: Column-stochastic transition matrix
- `U`: Uncertainty matrix
- `col_sums`: Column sums before normalization (diagnostic)

See `build_transition_matrix` for keyword arguments.
"""
function build_normalized_transition_matrix(F, partition::UlamPartition{N};
                                          kwargs...) where {N}
    P, U = build_transition_matrix(F, partition; kwargs...)
    col_sums = normalize_columns!(P)
    return P, U, col_sums
end

"""
    estimate_sparsity(P::SparseMatrixCSC)

Estimate the sparsity pattern of a sparse matrix.

Returns a NamedTuple with:
- `nnz`: Number of non-zero entries
- `total`: Total number of entries
- `sparsity`: Fraction of zero entries
- `avg_per_column`: Average non-zeros per column
"""
function estimate_sparsity(P::SparseMatrixCSC)
    total = prod(size(P))
    nz = nnz(P)
    n_cols = size(P, 2)

    return (
        nnz = nz,
        total = total,
        sparsity = 1.0 - nz/total,
        avg_per_column = nz / n_cols
    )
end
