"""
The certified Ulam approximation of the stationary density of a map of the two-torus with additive
uniform noise, and the certified L¹ distance between the computed density and the stationary one:
the computations of Section 6 ("Examples") of Pacifico, Bioni Liberalquino, Nisoli, "Stationary
densities of random maps with additive BV noise: existence, regularity and certified Ulam
approximation".

The stages, each in its own file:

- `grid.jl`: the uniform dyadic grids and the two orders of their cells;
- `transition_graph.jl`: the enclosure `L_lo ≤ L ≤ L_lo + U` of the Ulam matrix of a map
  (`UlamTree`, `refine!`, `ulam_matrices`, `fibred_area`);
- `maps.jl`: the maps of the paper's examples;
- `noisy_operator.jl`: the noisy Ulam operator `P_h = π N_ξ π L π`, matrix-free, with its ℓ¹
  rounding model;
- `norms_of_powers.jl`: `C_j ≥ ‖P_h^j|_{V_h}‖₁` by iterating every column;
- `l1_error.jl`: the computed density, its residuals, the coarse-fine recursion and the L¹ bounds;
- `experiments.jl`: the stages the drivers in `bench/` are thin over, and the names of their files;
- `archive.jl`, `deposit.jl`: the archival copies and the deposit of the data.
"""
module RigorousUlam2D

using ForwardDiff
using IntervalArithmetic, IntervalBoxes
using SparseArrays
using LinearAlgebra
import Serialization

export zeroth_order_enclosure, first_order_enclosure

include("grid.jl")
include("transition_graph.jl")
include("maps.jl")
include("noisy_operator.jl")
include("norms_of_powers.jl")
include("l1_error.jl")
include("experiments.jl")
include("archive.jl")
include("deposit.jl")

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