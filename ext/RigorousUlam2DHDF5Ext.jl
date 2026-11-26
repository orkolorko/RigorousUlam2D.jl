module RigorousUlam2DHDF5Ext

using RigorousUlam2D
using HDF5: HDF5, h5open, attributes, create_dataset, datatype, dataspace
using SparseArrays: SparseMatrixCSC, nnz

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

function _write_compressed(f, name, v::Vector, deflate)
    d = create_dataset(f, name, datatype(eltype(v)), dataspace(size(v));
                       chunk = (clamp(length(v), 1, 2^20),), shuffle = true, deflate = deflate)
    write(d, v)
end

function RigorousUlam2D.write_block_hdf5(path::AbstractString, block, field::Symbol;
        perm::AbstractVector{<:Integer}, source::AbstractString = "", deflate::Integer = 6)
    isfile(path) && throw(ArgumentError("$path exists; write_block_hdf5 does not overwrite"))
    field in (:L_lo, :U) || throw(ArgumentError("field must be :L_lo or :U"))
    A = getfield(block, field)
    size(A, 1) == size(A, 2) == length(perm) ||
        throw(DimensionMismatch("the matrix is $(size(A)), the permutation has length $(length(perm))"))
    B = SparseMatrixCSC{Float64,Int64}(A[perm, perm])
    h5open(path, "w") do f
        _write_compressed(f, "colptr", Vector{Int64}(B.colptr), deflate)
        _write_compressed(f, "rowval", Vector{Int64}(B.rowval), deflate)
        _write_compressed(f, "nzval", Vector{Float64}(B.nzval), deflate)
        at = attributes(f)
        at["matrix"] = String(field)
        at["size"] = Int64[size(B)...]
        at["nnz"] = Int64(nnz(B))
        at["format"] = RigorousUlam2D.BLOCK_HDF5_FORMAT
        at["enclosure"] = "L_lo <= L <= L_lo + U entrywise, L the Ulam matrix of the map on the grid of this depth"
        at["source"] = String(source)
        for k in (:map, :depth, :store_level, :store_maxlevel, :tag, :levels, :d0, :d1, :host, :script,
                  :julia, :threads, :commits, :header, :environment)
            hasproperty(block, k) || continue
            v = getproperty(block, k)
            at[String(k)] = v isa Integer ? Int64(v) : string(v)
        end
    end
    return path
end

end
