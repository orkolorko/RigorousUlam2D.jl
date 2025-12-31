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
    build_transition_matrix(F, partition::UlamPartition{N};
                           enclosure_method=first_order_enclosure,
                           ε=1e-6, maxdepth=30,
                           use_lower_bound=true) where {N}

Construct a sparse transition matrix for the Ulam approximation of the map F.

Returns a sparse matrix P where P[i,j] represents the (rigorous) transition
probability from partition element j to partition element i.

# Arguments
- `F`: The dynamical system (function mapping R^N → R^N)
- `partition`: The partition of the phase space
- `enclosure_method`: Method for computing rigorous image enclosures
- `ε`: Tolerance for adaptive refinement in rel_size
- `maxdepth`: Maximum depth for adaptive refinement
- `use_lower_bound`: If true, use lower bounds (verified transitions only)
                     If false, use upper bounds (includes uncertainties)

# Returns
- `P`: Sparse transition matrix (n_boxes × n_boxes)
- `uncertainties`: Matrix of uncertainty measures (upper - lower bounds)

# Example
```julia
F(x, y) = (2*x*(1-x), 0.5*y)
partition = UlamPartition(boxes, domain)
P, uncertainties = build_transition_matrix(F, partition)
```

# Note
The matrix is initially constructed in column-major order (transitions FROM column index).
Use `normalize_columns!` to convert to a column-stochastic matrix.
"""
function build_transition_matrix(F, partition::UlamPartition{N};
                                enclosure_method=first_order_enclosure,
                                ε=1e-6, maxdepth=30,
                                use_lower_bound=true) where {N}

    n = partition.n_boxes

    # Use sparse matrix for efficiency
    # Preallocate with estimated non-zeros (assume ~10% sparsity)
    I_indices = Int[]
    J_indices = Int[]
    values = Float64[]
    uncertainties_I = Int[]
    uncertainties_J = Int[]
    uncertainties_vals = Float64[]

    # Build the matrix column by column
    # P[i,j] = measure of Q_j that maps to Q_i
    for j in 1:n
        Q_source = partition.boxes[j]
        source_measure = box_measure(Q_source)

        if source_measure == 0
            continue  # Skip empty boxes
        end

        for i in 1:n
            Q_target = partition.boxes[i]

            # Compute rigorous transition measure
            lower, upper, uncertain_boxes = rel_size(F, Q_source, Q_target;
                                                     enclosure_method=enclosure_method,
                                                     ε=ε, maxdepth=maxdepth)

            # Choose bound based on preference
            transition_measure = use_lower_bound ? lower : upper
            uncertainty = upper - lower

            # Only add non-zero entries
            if transition_measure > 0
                push!(I_indices, i)
                push!(J_indices, j)
                push!(values, transition_measure / source_measure)
            end

            # Track uncertainties
            if uncertainty > 0
                push!(uncertainties_I, i)
                push!(uncertainties_J, j)
                push!(uncertainties_vals, uncertainty / source_measure)
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
