"""
2D Haar wavelet transform for dyadic partitions.

The partition uses Z-order (Morton curve) from `quadrisect`: [SW, SE, NW, NE].
At depth J, there are n = 4^J boxes on a 2^J × 2^J grid.

The Haar basis on this grid has:
- 1 scaling function (constant, index 1)
- 3 detail functions per scale per spatial position (H, V, D types)
  Total detail coefficients: 3·(4^0 + 4^1 + ... + 4^{J-1}) = 4^J - 1

Layout of coefficient vector (length n = 4^J):
  [phi; scale 0 (3 coeffs); scale 1 (12 coeffs); ...; scale J-1 (3·4^{J-1} coeffs)]

The transform operates in-place by processing groups of 4 (matching quadrisect order).
"""

export HaarBasis2D, haar_forward!, haar_inverse!, haar_forward, haar_inverse
export wavelet_matvec, wavelet_scale_ranges

"""
    HaarBasis2D

2D Haar wavelet basis on a dyadic grid of depth J (n = 4^J boxes).
"""
struct HaarBasis2D
    depth::Int   # J
    n::Int       # 4^J
end

function HaarBasis2D(depth::Int)
    depth >= 0 || throw(ArgumentError("depth must be non-negative"))
    HaarBasis2D(depth, 4^depth)
end

"""
    wavelet_scale_ranges(basis::HaarBasis2D)

Return a vector of (scale, UnitRange) pairs giving the index range for each
wavelet scale in the coefficient vector.

Scale -1 is the single scaling coefficient (index 1).
Scale j (j = 0, ..., J-1) has 3 · 4^j detail coefficients.
"""
function wavelet_scale_ranges(basis::HaarBasis2D)
    J = basis.depth
    ranges = Tuple{Int,UnitRange{Int}}[]
    push!(ranges, (-1, 1:1))  # scaling coefficient
    offset = 2
    for j in 0:(J - 1)
        count = 3 * 4^j
        push!(ranges, (j, offset:(offset + count - 1)))
        offset += count
    end
    return ranges
end

# ---------- Forward transform (grid → wavelet) ----------

"""
    haar_forward!(v::AbstractVector, basis::HaarBasis2D)

In-place 2D Haar forward transform. Converts grid-order values (Z-order,
length 4^J) into wavelet coefficients.

The transform processes the vector in blocks of 4 at each scale,
matching the [SW, SE, NW, NE] quadrisect ordering:

    phi  = (SW + SE + NW + NE) / 2   (scaling, ÷2 for L2 normalization)
    H    = (SW + SE - NW - NE) / 2   (horizontal detail: y-variation)
    V    = (SW - SE + NW - NE) / 2   (vertical detail: x-variation)
    D    = (SW - SE - NW + NE) / 2   (diagonal detail)
"""
function haar_forward!(v::AbstractVector, basis::HaarBasis2D)
    J = basis.depth
    n = basis.n
    length(v) == n || throw(DimensionMismatch("expected length $(n), got $(length(v))"))
    J == 0 && return v

    # Work buffer
    buf = similar(v)

    # Process from fine to coarse: at each level, the first 4^level entries
    # are the "scaling" part that gets split into scaling + detail
    block = n  # number of entries to process at this level
    for level in 1:J
        nblocks = block ÷ 4
        # Read from v[1:block], write scaling to buf[1:nblocks], detail to buf[nblocks+1:block]
        for b in 0:(nblocks - 1)
            i = 4 * b
            sw = v[i + 1]
            se = v[i + 2]
            nw = v[i + 3]
            ne = v[i + 4]

            buf[b + 1]              = (sw + se + nw + ne) / 2  # phi (scaling)
            buf[nblocks + 3*b + 1]  = (sw + se - nw - ne) / 2  # H
            buf[nblocks + 3*b + 2]  = (sw - se + nw - ne) / 2  # V
            buf[nblocks + 3*b + 3]  = (sw - se - nw + ne) / 2  # D
        end
        # Copy scaling part back, keep detail in place
        copyto!(v, 1, buf, 1, block)
        block = nblocks
    end
    return v
end

# ---------- Inverse transform (wavelet → grid) ----------

"""
    haar_inverse!(c::AbstractVector, basis::HaarBasis2D)

In-place 2D Haar inverse transform. Converts wavelet coefficients back to
grid-order values.

Inverse of the forward transform:
    SW = (phi + H + V + D) / 2
    SE = (phi + H - V - D) / 2
    NW = (phi - H + V - D) / 2
    NE = (phi - H - V + D) / 2
"""
function haar_inverse!(c::AbstractVector, basis::HaarBasis2D)
    J = basis.depth
    n = basis.n
    length(c) == n || throw(DimensionMismatch("expected length $(n), got $(length(c))"))
    J == 0 && return c

    buf = similar(c)

    # Process from coarse to fine
    block = 1  # number of scaling entries at current level
    for level in 1:J
        nblocks = block
        newblock = 4 * block
        for b in 0:(nblocks - 1)
            phi = c[b + 1]
            h   = c[nblocks + 3*b + 1]
            v   = c[nblocks + 3*b + 2]
            d   = c[nblocks + 3*b + 3]

            i = 4 * b
            buf[i + 1] = (phi + h + v + d) / 2  # SW
            buf[i + 2] = (phi + h - v - d) / 2  # SE
            buf[i + 3] = (phi - h + v - d) / 2  # NW
            buf[i + 4] = (phi - h - v + d) / 2  # NE
        end
        # Copy detail coefficients at finer scales unchanged
        if newblock < n
            copyto!(buf, newblock + 1, c, newblock + 1, n - newblock)
        end
        copyto!(c, 1, buf, 1, n)
        block = newblock
    end
    return c
end

# ---------- Non-mutating wrappers ----------

"""
    haar_forward(v::AbstractVector, basis::HaarBasis2D)

Return wavelet coefficients for grid values `v` (non-mutating).
"""
function haar_forward(v::AbstractVector, basis::HaarBasis2D)
    c = copy(v)
    haar_forward!(c, basis)
    return c
end

"""
    haar_inverse(c::AbstractVector, basis::HaarBasis2D)

Return grid values from wavelet coefficients `c` (non-mutating).
"""
function haar_inverse(c::AbstractVector, basis::HaarBasis2D)
    v = copy(c)
    haar_inverse!(v, basis)
    return v
end

# ---------- Wavelet-coordinate matrix-vector product ----------

"""
    wavelet_matvec(c::AbstractVector, P, basis::HaarBasis2D)

Apply the operator P in wavelet coordinates: W · P · W⁻¹ · c.

1. Inverse transform c → grid values v
2. Apply P: w = P * v
3. Forward transform w → wavelet coefficients
"""
function wavelet_matvec(c::AbstractVector, P, basis::HaarBasis2D)
    v = haar_inverse(c, basis)
    w = P * v
    return haar_forward(w, basis)
end
