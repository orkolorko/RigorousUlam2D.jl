module RigorousUlam2D

using ForwardDiff
using IntervalArithmetic

export zeroth_order_enclosure, first_order_enclosure

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
    mids = map(mid, intervals)

    center_value = F(mids...)
    center_vec = _to_vector(center_value)
    center_intervals = Interval.(center_vec)

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
