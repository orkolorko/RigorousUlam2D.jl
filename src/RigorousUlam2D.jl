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
include("domain_restriction.jl")
include("boundary_conditions.jl")
include("adaptive_enclosure.jl")

# Re-export from submodules
# Partition utilities
export bisect_interval, bisect_box, quadrisect
export issubset_box, isdisjoint_box, intersects, is_empty_box
export box_measure, box_width, box_radius

# Adaptive refinement
export rel_size, adaptive_partition

# Sparse Ulam
export build_transition_matrix, normalize_columns!, build_normalized_transition_matrix
export UlamPartition, partition_index, estimate_sparsity

# Noise convolution
export minkowski_sum, noisy_image, noisy_rel_size
export build_noisy_transition_matrix
export uniform_noise_box, anisotropic_noise_box

# Domain restriction
export compute_forward_image_enclosure, compute_effective_domain
export restrict_partition_to_domain, create_restricted_partition
export hull, intersect_boxes

# Boundary conditions
export apply_periodic_bc, wrap_to_torus
export noisy_image_periodic, noisy_rel_size_periodic
export build_noisy_transition_matrix_periodic

# Adaptive enclosure
export adaptive_enclosure, adaptive_enclosure_auto
export AdaptiveEnclosureMethod, FixedDepth, ErrorBased, ExpansionBased
export compare_enclosure_methods

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

using IntervalArithmetic, StaticArrays
export interval

const Ia = I"1.4"
const Ib = I"0.3"
const I2 = I"2"
const I0 = I"0"
const I1 = I"1"

# Map
H(x, y; a=Ia, b=Ib) = [I"1" - a * (x^2) + y;  b * x]
H(u; a=Ia, b=Ib)    = H(u[1], u[2]; a=a, b=b)

# Jacobian (interval-valued)
DH(x, y; a=Ia, b=Ib) = @SMatrix [-I2*a*x   I1; b        I0]
    
# Enclosure

function H_centered(u; a = Ia, b = Ib)
    xm, ym = interval(mid(u[1])), interval(mid(u[2]))
    δx, δy = u[1]-xm, u[2]-ym

    Jm = DH(xm, ym; a=a, b=b)                 # point Jacobian
    v  = H(xm, ym; a=a, b=b) + Jm*@SVector[δx, δy]

    # remainder: only on the first component, |R1| ≤ a * rad(x)^2
    rx = radius(u[1])
    R1 = sup(a) * rx^2
    v1 = v[1] + interval(-R1, R1)
    @SVector [v1, v[2]]   
end

export H, DH, H_centered

using IterTools: product

# 2D: split a box Qi = [Ix, Iy] into 4 sub-boxes
function quadsect(Qi)
    x1, x2 = bisect(Qi[1])
    y1, y2 = bisect(Qi[2])
    return [[x, y] for (x, y) in product((x1, x2), (y1, y2))]
end

using IntervalArithmetic

# 1D interval subset (closed, inclusive)
@inline subset1D(a::Interval, b::Interval) = (inf(b) <= inf(a)) & (sup(a) <= sup(b))

# Empty box if any coord interval is empty
@inline is_empty_box(Q) = any(isempty_interval, Q)

# N-D subset for boxes Q = (I₁, …, I_d) or [I₁, …, I_d]
function issubset(Qi, Qj)
    @assert length(Qi) == length(Qj) "Box dims must match"
    if is_empty_box(Qi)             # ∅ ⊆ anything
        return true
    elseif is_empty_box(Qj)         # nonempty ⊆ ∅ is false
        return is_empty_box(Qi)
    end
    return all(subset1D.(Qi, Qj))
end

# Disjointness: strict gap in ANY coordinate
@inline function isdisjoint(Qi, Qj)
    @assert length(Qi) == length(Qj)
    return any(@. (sup(Qi) < inf(Qj)) | (sup(Qj) < inf(Qi)))
end

@inline intersects(Qi, Qj) = !isdisjoint(Qi, Qj)

width1D(I::Interval) = interval(sup(I)) - interval(inf(I))
area(Q) = width1D(Q[1]) * width1D(Q[2])  

function rel_size(Q_i, Q_j; ε = 1e-6, maxdepth = 30)
    val = I0        # guaranteed IN area
    uncertain = I0  # unresolved boundary belt area

    # worklist holds (box, depth)
    candidates = [(Q_i, 0)]

    while !isempty(candidates)
        Q, d = pop!(candidates)

        imQ = H_centered(Q)  # your enclosure of H(Q); keep as-is

        if issubset(imQ, Q_j)
            val += area(Q)
        elseif isdisjoint(imQ, Q_j)
            # OUT: nothing to add
        else
            if area(Q) <= ε || d >= maxdepth
                uncertain += area(Q)
            else
                # subdivide
                for child in quadsect(Q)
                    push!(candidates, (child, d + 1))
                end
            end
        end
    end

    return val, uncertain   # inner measure, unresolved belt; outer ≤ val + uncertain
end

export rel_size


end