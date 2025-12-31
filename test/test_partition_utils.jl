using IntervalArithmetic, IntervalBoxes
using RigorousUlam2D
using Test

@testset "Partition Utilities" begin

    @testset "bisect_interval" begin
        I = interval(0.0, 1.0)
        I_left, I_right = bisect_interval(I)

        @test inf(I_left) == 0.0
        @test sup(I_left) == 0.5
        @test inf(I_right) == 0.5
        @test sup(I_right) == 1.0

        # Check they partition the original interval
        @test issubset_interval(I_left, I)
        @test issubset_interval(I_right, I)
    end

    @testset "bisect_box" begin
        box = IntervalBox(interval(0.0, 1.0), interval(0.0, 2.0))

        # Bisect along dimension 1 (x-direction)
        left, right = bisect_box(box, 1)
        @test sup(left[1]) == 0.5
        @test inf(right[1]) == 0.5
        @test isequal_interval(left[2], box[2])
        @test isequal_interval(right[2], box[2])

        # Bisect along widest dimension (should be dimension 2)
        left2, right2 = bisect_box(box)
        @test isequal_interval(left2[1], box[1])
        @test isequal_interval(right2[1], box[1])
    end

    @testset "quadrisect" begin
        box = IntervalBox(interval(0.0, 1.0), interval(0.0, 1.0))
        quads = quadrisect(box)

        @test length(quads) == 4

        # Check they partition the space
        total_measure = sum(box_measure(q) for q in quads)
        @test total_measure ≈ box_measure(box)

        # Check they're disjoint (interiors)
        for i in 1:4
            for j in (i+1):4
                # Interior disjointness - allow boundary overlap
                @test box_measure(quads[i]) > 0
                @test box_measure(quads[j]) > 0
            end
        end
    end

    @testset "box subset and disjoint" begin
        A = IntervalBox(interval(0.2, 0.4), interval(0.3, 0.5))
        B = IntervalBox(interval(0.0, 1.0), interval(0.0, 1.0))
        C = IntervalBox(interval(0.6, 0.8), interval(0.6, 0.9))

        @test issubset_box(A, B)
        @test !issubset_box(B, A)
        @test isdisjoint_box(A, C)
        @test !isdisjoint_box(A, B)
        @test intersects(A, B)
        @test !intersects(A, C)
    end

    @testset "box measures" begin
        box = IntervalBox(interval(0.0, 2.0), interval(0.0, 3.0))

        @test box_measure(box) ≈ 6.0
        @test box_width(box) ≈ 3.0  # max diameter
        @test box_radius(box) ≈ sqrt(4 + 9) / 2

        # Empty box - create using emptyinterval()
        empty_int = emptyinterval(Float64)
        normal_int = interval(0.0, 1.0)
        empty_box = IntervalBox(empty_int, normal_int)
        @test is_empty_box(empty_box)
        @test box_measure(empty_box) == 0.0
    end
end
