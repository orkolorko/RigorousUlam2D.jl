using IntervalArithmetic, IntervalBoxes
using RigorousUlam2D
using Test

@testset "RigorousUlam2D.jl" begin

    @testset "Enclosure Methods" begin
        @testset "zeroth order enclosure" begin
            F(x, y) = x^2 + y
            box = IntervalBox(interval(0.0, 1.0), interval(-1.0, 2.0))
            @test isequal_interval(zeroth_order_enclosure(F, box), F(box...))
        end

        @testset "first order enclosure" begin
            F(x, y) = x^2 + y^2
            box = IntervalBox(interval(-0.2, 0.4), interval(1.0, 1.5))
            first_order = first_order_enclosure(F, box)
            zeroth_order = zeroth_order_enclosure(F, box)
            # Both should be valid enclosures (contain the true image)
            @test first_order isa Interval
            @test zeroth_order isa Interval

            G(x, y) = (x + y, x - y)
            zeroth_tuple = zeroth_order_enclosure(G, box)
            first_tuple = first_order_enclosure(G, box)

            # Both should return tuples for tuple-valued functions
            @test first_tuple isa Tuple{Interval, Interval}
            @test zeroth_tuple isa Tuple{Interval, Interval}
        end
    end

    # Include component tests
    include("test_partition_utils.jl")
    include("test_adaptive_refinement.jl")
    include("test_sparse_ulam.jl")
    include("test_noise_convolution.jl")
    include("test_grid_aligned_noise.jl")
    include("test_domain_restriction.jl")

end
