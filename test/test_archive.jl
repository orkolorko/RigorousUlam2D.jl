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
