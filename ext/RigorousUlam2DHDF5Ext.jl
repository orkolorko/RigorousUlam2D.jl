module RigorousUlam2DHDF5Ext

using RigorousUlam2D
using HDF5: HDF5, h5open, attributes

function RigorousUlam2D.write_density_hdf5(path::AbstractString, record; source::AbstractString = "",
        domain::AbstractVector{<:Real})
    isfile(path) && throw(ArgumentError("$path exists; write_density_hdf5 does not overwrite"))
    m = Vector{Float64}(record.density)
    length(m) == 4^record.depth ||
        throw(DimensionMismatch("the record holds $(length(m)) masses, a grid of depth $(record.depth) has $(4^record.depth)"))
    length(domain) == 4 || throw(ArgumentError("domain must be [xmin, xmax, ymin, ymax]"))
    a = Rational{BigInt}(record.a)
    h5open(path, "w") do f
        f["masses"] = m
        at = attributes(f)
        at["map"] = String(record.map)
        at["depth"] = Int64(record.depth)
        at["half_width"] = "$(numerator(a))/$(denominator(a))"
        at["half_width_numerator"] = Int64(numerator(a))
        at["half_width_denominator"] = Int64(denominator(a))
        at["domain"] = Float64.(collect(domain))
        at["ordering"] = RigorousUlam2D.DENSITY_HDF5_ORDERING
        at["source"] = String(source)
        at["header"] = String(join(vcat(record.header), "\n"))
    end
    return path
end

end
