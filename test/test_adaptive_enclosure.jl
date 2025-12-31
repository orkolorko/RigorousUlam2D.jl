using IntervalArithmetic, IntervalBoxes
using RigorousUlam2D
using Test

@testset "Adaptive Enclosure" begin

    @testset "FixedDepth" begin
        # Linear map - subdivision shouldn't help much
        F_linear(x, y) = (2*x, 3*y)
        box = IntervalBox(interval(0.0, 1.0), interval(0.0, 1.0))

        enc0 = adaptive_enclosure(F_linear, box, FixedDepth(0))
        enc2 = adaptive_enclosure(F_linear, box, FixedDepth(2))

        # Both should give valid enclosures
        @test enc0 isa IntervalBox
        @test enc2 isa IntervalBox

        # For linear map, subdivision might not improve much
        # But enc2 should not be worse than enc0
        @test box_measure(enc2) <= box_measure(enc0) * 1.1
    end

    @testset "ErrorBased" begin
        # Nonlinear map - subdivision should help
        F_nonlinear(x, y) = (x^2 + y^2, x*y)
        box = IntervalBox(interval(0.0, 1.0), interval(0.0, 1.0))

        # Coarse tolerance
        enc_coarse = adaptive_enclosure(F_nonlinear, box,
                                       ErrorBased(0.5, 10))

        # Fine tolerance - should give tighter bounds
        enc_fine = adaptive_enclosure(F_nonlinear, box,
                                     ErrorBased(0.1, 10))

        @test box_measure(enc_fine) <= box_measure(enc_coarse)
    end

    @testset "ExpansionBased" begin
        # Expanding map
        F_expand(x, y) = (4*x*(1-x), 2*y)
        box = IntervalBox(interval(0.2, 0.4), interval(0.3, 0.5))

        # Allow large expansion
        enc_loose = adaptive_enclosure(F_expand, box,
                                      ExpansionBased(10.0, 5))

        # Restrict expansion - should subdivide more
        enc_tight = adaptive_enclosure(F_expand, box,
                                      ExpansionBased(2.0, 8))

        # Tighter restriction should give better enclosure
        @test box_measure(enc_tight) <= box_measure(enc_loose)
    end

    @testset "adaptive_enclosure_auto" begin
        # Test automatic method selection
        F(x, y) = (x^2 - y^2, 2*x*y)  # Complex Squaring
        box = IntervalBox(interval(0.5, 0.7), interval(0.5, 0.7))

        enc_auto = adaptive_enclosure_auto(F, box, target_improvement=2.0)

        # Should produce valid enclosure
        @test enc_auto isa IntervalBox
        @test box_measure(enc_auto) > 0
    end

    @testset "return_boxes option" begin
        F(x, y) = (x^2, y^2)
        box = IntervalBox(interval(0.0, 1.0), interval(0.0, 1.0))

        enc, boxes, images = adaptive_enclosure(F, box, FixedDepth(2),
                                               return_boxes=true)

        # Should return subdivision info
        @test length(boxes) == 4^2  # 2 levels of quadrisection
        @test length(images) == length(boxes)
        @test all(img isa IntervalBox for img in images)

        # Hull of images should equal enclosure
        @test enc == RigorousUlam2D.hull(images...)
    end

    @testset "Integration with domain restriction" begin
        # Use adaptive enclosure in forward image computation
        F(x, y) = (0.5*x + 0.25, 0.5*y + 0.25)
        domain = IntervalBox(interval(0.0, 1.0), interval(0.0, 1.0))

        # Compare standard vs adaptive
        img_standard = compute_forward_image_enclosure(F, domain, subdivision_depth=2)

        # Adaptive with similar settings
        img_adaptive = adaptive_enclosure(F, domain, FixedDepth(2))

        # Should give comparable results
        @test abs(box_measure(img_adaptive) - box_measure(img_standard)) < 0.1
    end

    @testset "compare_enclosure_methods" begin
        F(x, y) = (2*x*(1-x), 0.5*y)
        box = IntervalBox(interval(0.2, 0.6), interval(0.3, 0.7))

        methods = [
            FixedDepth(0),
            FixedDepth(1),
            ErrorBased(0.1, 5),
            ExpansionBased(2.0, 5)
        ]

        results = compare_enclosure_methods(F, box, methods=methods)

        # Should have result for each method
        @test length(results) == length(methods)

        # Each result should have required fields
        for result in results
            @test hasfield(typeof(result), :method)
            @test hasfield(typeof(result), :volume)
            @test hasfield(typeof(result), :n_boxes)
            @test result.volume > 0
            @test result.n_boxes > 0
        end

        # More subdivision should generally give tighter bounds
        @test results[2].volume <= results[1].volume  # Depth 1 <= Depth 0
    end

    @testset "Adaptive with zeroth order base" begin
        # Test with zeroth-order as base
        F(x, y) = (x + y, x - y)
        box = IntervalBox(interval(0.0, 0.5), interval(0.0, 0.5))

        enc = adaptive_enclosure(F, box, FixedDepth(2),
                                base_enclosure=zeroth_order_enclosure)

        @test enc isa IntervalBox
        @test box_measure(enc) > 0
    end
end
