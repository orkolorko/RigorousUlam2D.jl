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



end
