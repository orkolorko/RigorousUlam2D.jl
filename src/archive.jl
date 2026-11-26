"""
Archival copies of the computed stationary densities, for deposit in a data repository.

The records of `bench/stationary_density.jl` are files of the `Serialization` standard library,
which is not guaranteed to be readable by another Julia version.  `write_density_hdf5` writes the
same cell masses, bit for bit, to an HDF5 file with the metadata needed to read them without this
package.  It is implemented in the package extension `RigorousUlam2DHDF5Ext`, loaded with `HDF5`.
"""

export write_density_hdf5, DENSITY_HDF5_ORDERING

"The ordering of the cell masses, written into every archival file as an attribute."
const DENSITY_HDF5_ORDERING = "masses[(ix - 1) * 2^depth + iy] is the mass of the cell in column ix " *
    "(along x) and row iy (along y), 1-based, of the uniform grid of 2^depth x 2^depth cells of the " *
    "domain; on each cell the density is the mass divided by the area of the cell"

"""
    write_density_hdf5(path, record; source = "", domain) -> path

Write the computed density of a `stationary_density` record (the named tuple stored by
`bench/stationary_density.jl`, with fields `map`, `depth`, `a`, `density`, `header`) to the HDF5 file
`path`: the dataset `masses`, the vector `record.density` unchanged (`Float64`, length `4^depth`),
and the attributes `map`, `depth`, `half_width` (the noise half-width as the string `"p/q"`),
`half_width_numerator`, `half_width_denominator`, `domain` (`[xmin, xmax, ymin, ymax]`),
`ordering` ([`DENSITY_HDF5_ORDERING`](@ref)), `source` (the name of the record file) and `header`
(the header lines of the record, which name the driver, the host and the commit).  Requires
`using HDF5`.  Refuses to overwrite an existing file.
"""
function write_density_hdf5 end
