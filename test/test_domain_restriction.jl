using IntervalArithmetic, IntervalBoxes
using RigorousUlam2D
using Test

@testset "Domain Restriction" begin

    @testset "hull operations" begin
        # Hull of intervals
        I1 = interval(0.0, 0.5)
        I2 = interval(0.3, 0.8)
        I3 = interval(-0.2, 0.1)

        h = RigorousUlam2D.hull(I1, I2, I3)
        @test inf(h) == -0.2
        @test sup(h) == 0.8

        # Hull of boxes
        box1 = IntervalBox(interval(0.0, 0.5), interval(0.0, 0.5))
        box2 = IntervalBox(interval(0.3, 0.8), interval(0.3, 0.8))
        box3 = IntervalBox(interval(-0.1, 0.2), interval(0.6, 1.0))

        h_box = RigorousUlam2D.hull(box1, box2, box3)
        @test inf(h_box[1]) == -0.1
        @test sup(h_box[1]) == 0.8
        @test inf(h_box[2]) == 0.0
        @test sup(h_box[2]) == 1.0
    end

    @testset "intersect_boxes" begin
        A = IntervalBox(interval(0.0, 1.0), interval(0.0, 1.0))
        B = IntervalBox(interval(0.5, 1.5), interval(0.5, 1.5))

        C = intersect_boxes(A, B)
        @test inf(C[1]) == 0.5
        @test sup(C[1]) == 1.0
        @test inf(C[2]) == 0.5
        @test sup(C[2]) == 1.0

        # Disjoint boxes
        D = IntervalBox(interval(2.0, 3.0), interval(2.0, 3.0))
        E = intersect_boxes(A, D)
        @test is_empty_box(E)
    end

    # DEPRECATED: restrict_partition_to_domain was removed in favor of build_adaptive_partition
    # @testset "restrict_partition_to_domain" begin
    #     # Create a large partition
    #     domain_large = IntervalBox(interval(0.0, 1.0), interval(0.0, 1.0))
    #     boxes = quadrisect(domain_large)
    #     for _ in 1:2
    #         new_boxes = eltype(boxes)[]
    #         for box in boxes
    #             append!(new_boxes, quadrisect(box))
    #         end
    #         boxes = new_boxes
    #     end
    #
    #     # 64 boxes total
    #     @test length(boxes) == 64
    #
    #     # Restrict to smaller domain
    #     restricted_domain = IntervalBox(interval(0.3, 0.7), interval(0.3, 0.7))
    #     restricted_boxes = restrict_partition_to_domain(boxes, restricted_domain)
    #
    #     # Should have fewer boxes
    #     @test length(restricted_boxes) < length(boxes)
    #
    #     # All restricted boxes should intersect the domain
    #     @test all(!isdisjoint_box(box, restricted_domain) for box in restricted_boxes)
    #
    #     # None of the excluded boxes should be inside restricted domain
    #     # (can't use setdiff because it calls union! which isn't supported for intervals)
    #     excluded = [box for box in boxes if !(box in restricted_boxes)]
    #     @test all(isdisjoint_box(box, restricted_domain) for box in excluded)
    # end

    # DEPRECATED: compute_effective_partition_adaptive was removed
    # All tests for this function have been commented out
end
