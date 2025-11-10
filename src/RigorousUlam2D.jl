module RigorousUlam2D

using IntervalArithmetic

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