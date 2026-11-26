using HDF5

@testset "write_density_hdf5" begin
    masses = collect(range(1.0, 16.0; length = 16)) ./ 136
    record = (; map = "lorenz", depth = 2, a = 77 // 256, density = masses,
                header = "# bench/stationary_density.jl on test\n# RigorousUlam2D abc1234")
    dir = mktempdir()
    path = joinpath(dir, "density.h5")
    @test write_density_hdf5(path, record; source = "stationary_density.test.jls",
                             domain = [-0.5, 0.5, -0.5, 0.5]) == path
    h5open(path, "r") do f
        @test read(f["masses"]) == masses                      # bit for bit
        at = attributes(f)
        @test read(at["map"]) == "lorenz"
        @test read(at["depth"]) == 2
        @test read(at["half_width"]) == "77/256"
        @test read(at["half_width_numerator"]) == 77
        @test read(at["half_width_denominator"]) == 256
        @test read(at["domain"]) == [-0.5, 0.5, -0.5, 0.5]
        @test read(at["ordering"]) == DENSITY_HDF5_ORDERING
        @test read(at["source"]) == "stationary_density.test.jls"
        @test occursin("abc1234", read(at["header"]))
    end
    @test_throws ArgumentError write_density_hdf5(path, record; domain = [-0.5, 0.5, -0.5, 0.5])
    @test_throws DimensionMismatch write_density_hdf5(joinpath(dir, "bad.h5"),
        merge(record, (; density = masses[1:8])); domain = [-0.5, 0.5, -0.5, 0.5])
end

@testset "write_block_hdf5" begin
    using SparseArrays
    A = sparse([1, 3, 2, 4, 4], [1, 1, 2, 3, 4], [0.5, 0.25, 1.0, 2.0^-30, 0.125], 4, 4)
    U = sparse([1, 2], [1, 3], [1e-12, 3e-13], 4, 4)
    blk = (; map = "lorenz", depth = 1, store_level = 14, tag = "xinv", L_lo = A, U = U, graph = nothing,
             host = "test", header = "# bench/tree_deepen.jl on test", commits = (; RigorousUlam2D = "abc1234", dirty = false))
    perm = [3, 1, 4, 2]
    dir = mktempdir()
    for (field, M) in ((:L_lo, A), (:U, U))
        path = joinpath(dir, "$field.h5")
        @test write_block_hdf5(path, blk, field; perm, source = "blk.jls") == path
        h5open(path, "r") do f
            B = SparseMatrixCSC(4, 4, read(f["colptr"]), read(f["rowval"]), read(f["nzval"]))
            @test B == M[perm, perm]                                  # entries unchanged, permuted
            at = attributes(f)
            @test read(at["matrix"]) == String(field)
            @test read(at["size"]) == [4, 4] && read(at["nnz"]) == nnz(M)
            @test read(at["format"]) == BLOCK_HDF5_FORMAT
            @test read(at["source"]) == "blk.jls" && read(at["map"]) == "lorenz" && read(at["depth"]) == 1
            @test occursin("abc1234", read(at["commits"])) && !haskey(at, "graph")
            @test any(x -> x isa HDF5.Filters.Deflate, HDF5.get_create_properties(f["nzval"]).filters)
        end
    end
    @test_throws ArgumentError write_block_hdf5(joinpath(dir, "U.h5"), blk, :U; perm)
    @test_throws ArgumentError write_block_hdf5(joinpath(dir, "g.h5"), blk, :graph; perm)
    @test_throws DimensionMismatch write_block_hdf5(joinpath(dir, "p.h5"), blk, :U; perm = [1, 2])
end
