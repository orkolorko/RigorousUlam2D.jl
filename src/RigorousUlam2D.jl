module RigorousUlam2D

using ForwardDiff
using IntervalArithmetic, IntervalBoxes
using SparseArrays
using LinearAlgebra

# Export enclosure methods
export zeroth_order_enclosure, first_order_enclosure

# Include submodules
include("partition_utils.jl")
include("adaptive_refinement.jl")
include("sparse_ulam.jl")
include("noise_convolution.jl")
include("boundary_conditions.jl")
include("spectral_analysis.jl")
include("transfer_matvec.jl")
include("haar_wavelet.jl")
include("schur_norm_bounds.jl")
include("transition_graph.jl")

# Re-export from submodules
# Partition utilities
export bisect_interval, bisect_box, quadrisect
export issubset_box, isdisjoint_box, intersects, is_empty_box
export box_measure, box_width, box_radius
export expand_interval, intersect_boxes
export build_uniform_partition, build_adaptive_partition

# Adaptive refinement
export rel_size

# Sparse Ulam
export build_transition_matrix, normalize_columns!, build_normalized_transition_matrix
export UlamPartition, partition_index, estimate_sparsity

# Noise convolution
export minkowski_sum, noisy_image
export uniform_noise_box, anisotropic_noise_box
export build_noise_convolution_operator, noise_matvec, build_grid_index_maps

# Boundary conditions
export apply_periodic_bc, wrap_to_torus
export noisy_image_periodic

# Spectral analysis (Galatolo-Nisoli / ZZ paper)
export operator_norm_on_V, operator_norm_on_V_power, find_contraction_power
export rigorous_norm_on_V, rigorous_norm_on_V_power, higham_gamma
export norms_of_powers_sparse, norms_of_powers_matvec, refine_norms_of_powers, norms_of_powers_from_coarser_grid
export compute_stationary_density, spectral_gap
export noise_variation, ulam_approximation_error, rigorous_error_bound

# Haar wavelet
export HaarBasis2D, haar_forward!, haar_inverse!, haar_forward, haar_inverse
export wavelet_matvec, wavelet_scale_ranges

# Schur complement norm bounds
export wavelet_operator_blocks, noise_attenuation_bound
export schur_norms_of_powers, wavelet_rigorous_error_bound

"""
    zeroth_order_enclosure(F, box::IntervalBox)

Compute a rigorous enclosure of the image of the rectangle `box` under the
function `F` by evaluating `F` directly on the intervals defining the box.
This corresponds to the 0-th order (pure interval) method.
"""
zeroth_order_enclosure(F, box::IntervalBox) = F(box...)

"""
    first_order_enclosure(F, box::IntervalBox{2})

Compute a rigorous first-order (mean value) enclosure of the image of the
rectangle `box` under the function `F`. The Jacobian matrix is obtained via
automatic differentiation using `ForwardDiff` and evaluated over the interval
box, providing a tighter enclosure than the 0-th order method in many cases.
"""
function first_order_enclosure(F, box::IntervalBox{2})
    intervals = Tuple(box)
    mids = Interval.(map(mid, intervals))

    center_value = F(mids...)
    center_intervals = _to_vector(center_value)
    
    function wrapper(x::AbstractVector{T}) where {T}
        values = _to_vector(F(x...))
        return T.(values)
    end

    jacobian = ForwardDiff.jacobian(wrapper, collect(intervals))
    offsets = [intervals[i] - mids[i] for i in 1:2]
    enclosure_vec = center_intervals .+ jacobian * offsets

    return _from_vector(center_value, enclosure_vec)
end

function _to_vector(value)
    if value isa Number
        return [value]
    elseif value isa Tuple
        return [value[i] for i in eachindex(value)]
    elseif value isa AbstractVector
        return Vector(value)
    else
        throw(ArgumentError("Unsupported output type $(typeof(value))"))
    end
end

function _from_vector(prototype, values::AbstractVector)
    if prototype isa Number
        return values[1]
    elseif prototype isa Tuple
        return tuple(values...)
    elseif prototype isa AbstractVector
        result = similar(prototype, eltype(values))
        for i in eachindex(result, values)
            result[i] = values[i]
        end
        return result
    else
        throw(ArgumentError("Unsupported output type $(typeof(prototype))"))
    end
end

end