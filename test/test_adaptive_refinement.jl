using IntervalArithmetic, IntervalBoxes
using RigorousUlam2D
using Test

@testset "Adaptive Refinement" begin

    @testset "rel_size basic" begin
        # Simple map: identity
        F_id(x, y) = (x, y)

        source = IntervalBox(interval(0.0, 0.5), interval(0.0, 0.5))
        target = IntervalBox(interval(0.0, 1.0), interval(0.0, 1.0))

        lower, upper, uncertain = rel_size(F_id, source, target, maxdepth=5)

        # Identity map: source ⊆ target, so all points map in
        @test lower ≈ box_measure(source)
        @test upper ≈ box_measure(source)
        @test length(uncertain) == 0
    end

    @testset "rel_size disjoint" begin
        # Map that shifts outside
        F_shift(x, y) = (x + 2.0, y + 2.0)

        source = IntervalBox(interval(0.0, 0.5), interval(0.0, 0.5))
        target = IntervalBox(interval(0.0, 1.0), interval(0.0, 1.0))

        lower, upper, uncertain = rel_size(F_shift, source, target, maxdepth=5)

        # Shifted map: no points map into target
        @test lower == 0.0
        @test upper == 0.0
        @test length(uncertain) == 0
    end

    @testset "rel_size partial overlap" begin
        # Scaling map
        F_scale(x, y) = (2.0 * x, 2.0 * y)

        source = IntervalBox(interval(0.0, 1.0), interval(0.0, 1.0))
        target = IntervalBox(interval(0.0, 1.0), interval(0.0, 1.0))

        lower, upper, uncertain = rel_size(F_scale, source, target, maxdepth=10, ε=1e-4)

        # Only part of source maps to target (roughly 1/4)
        @test 0.0 <= lower <= 0.25
        @test 0.25 <= upper <= 1.0
        @test lower <= upper
    end

    @testset "adaptive_partition" begin
        # Simple expanding map
        F_expand(x, y) = (2.0 * x, 2.0 * y)

        domain = IntervalBox(interval(0.0, 1.0), interval(0.0, 1.0))

        partition = adaptive_partition(domain, F_expand, initial_depth=2, max_refinement_depth=4)

        @test length(partition) >= 4^2  # At least initial grid
        @test all(issubset_box(box, domain) for box in partition)

        # Check partition covers domain (approximately)
        total_measure = sum(box_measure(box) for box in partition)
        @test total_measure ≈ box_measure(domain) atol=1e-10
    end

    @testset "rel_size with zeroth order" begin
        # Test with zeroth order enclosure
        F(x, y) = (x^2, y^2)

        source = IntervalBox(interval(0.0, 0.5), interval(0.0, 0.5))
        target = IntervalBox(interval(0.0, 0.5), interval(0.0, 0.5))

        lower, upper, _ = rel_size(F, source, target,
                                   enclosure_method=zeroth_order_enclosure,
                                   maxdepth=8)

        @test 0.0 <= lower <= upper <= box_measure(source)
    end
end
