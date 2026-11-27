using IntervalArithmetic, IntervalBoxes
using RigorousUlam2D
using Test

@testset "grids" begin

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
        q = quadrisect(box)
        @test length(q) == 4
        ends = [(inf(b[1]), sup(b[1]), inf(b[2]), sup(b[2])) for b in q]
        @test ends == [(0.0, 0.5, 0.0, 0.5), (0.5, 1.0, 0.0, 0.5), (0.0, 0.5, 0.5, 1.0), (0.5, 1.0, 0.5, 1.0)]
    end

    @testset "build_uniform_partition: 4^d cells with exact ends, covering the domain" begin
        dom = IntervalBox(interval(-1.0, 1.0), interval(-0.5, 0.5))
        for d in 0:3
            P = build_uniform_partition(dom, d)
            @test P.n_boxes == 4^d && length(P.boxes) == 4^d
            @test sum(diam(b[1]) * diam(b[2]) for b in P.boxes) == diam(dom[1]) * diam(dom[2])
            @test all(b -> inf(b[1]) >= -1 && sup(b[1]) <= 1 && inf(b[2]) >= -0.5 && sup(b[2]) <= 0.5, P.boxes)
        end
        @test_throws ErrorException build_uniform_partition(dom, -1)
    end

    @testset "wrap_to_torus" begin
        @test wrap_to_torus(1.25, 0.0, 1.0) == 0.25
        @test wrap_to_torus(-0.25, 0.0, 1.0) == 0.75
        @test wrap_to_torus(0.75, -0.5, 0.5) == -0.25
    end
end
