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

    @testset "compute_forward_image_enclosure basic" begin
        # Identity map
        F_id(x, y) = (x, y)
        domain = IntervalBox(interval(0.0, 1.0), interval(0.0, 1.0))

        image = compute_forward_image_enclosure(F_id, domain)
        @test isequal_interval(image[1], domain[1])
        @test isequal_interval(image[2], domain[2])

        # Contracting map
        F_contract(x, y) = (0.5*x + 0.25, 0.5*y + 0.25)
        image_contract = compute_forward_image_enclosure(F_contract, domain)

        # Should be roughly [0.25, 0.75] × [0.25, 0.75]
        @test inf(image_contract[1]) >= 0.2
        @test sup(image_contract[1]) <= 0.8
        @test inf(image_contract[2]) >= 0.2
        @test sup(image_contract[2]) <= 0.8

        # Image should be smaller than domain
        @test box_measure(image_contract) < box_measure(domain)
    end

    @testset "compute_forward_image_enclosure with subdivision" begin
        # Nonlinear map - subdivision should give tighter bounds
        F(x, y) = (x^2, y^2)
        domain = IntervalBox(interval(0.0, 1.0), interval(0.0, 1.0))

        image0 = compute_forward_image_enclosure(F, domain, subdivision_depth=0)
        image2 = compute_forward_image_enclosure(F, domain, subdivision_depth=2)

        # Both should be valid enclosures
        @test inf(image0[1]) <= 0.0
        @test sup(image0[1]) >= 1.0
        @test inf(image2[1]) <= 0.0
        @test sup(image2[1]) >= 1.0

        # Subdivision should give smaller or equal enclosure
        @test box_measure(image2) <= box_measure(image0)
    end

    @testset "compute_effective_domain without noise" begin
        # Strongly contracting map
        F(x, y) = (0.4*x + 0.3, 0.4*y + 0.3)
        domain = IntervalBox(interval(0.0, 1.0), interval(0.0, 1.0))

        eff_domain, img_clean, img_noisy = compute_effective_domain(F, domain, nothing,
                                                                     subdivision_depth=1,
                                                                     safety_margin=1.0)

        # Effective domain should be smaller than initial
        @test box_measure(eff_domain) < box_measure(domain)

        # Should be roughly [0.3, 0.7] × [0.3, 0.7]
        @test inf(eff_domain[1]) >= 0.25
        @test sup(eff_domain[1]) <= 0.75

        # img_noisy should equal img_clean when no noise
        @test isequal_interval(img_clean[1], img_noisy[1])
        @test isequal_interval(img_clean[2], img_noisy[2])
    end

    @testset "compute_effective_domain with noise" begin
        F(x, y) = (0.5*x + 0.25, 0.5*y + 0.25)
        domain = IntervalBox(interval(0.0, 1.0), interval(0.0, 1.0))
        noise = uniform_noise_box(0.05, 2)

        eff_domain, img_clean, img_noisy = compute_effective_domain(F, domain, noise,
                                                                     subdivision_depth=1,
                                                                     safety_margin=1.1)

        # Noisy image should be larger than clean image
        @test box_measure(img_noisy) > box_measure(img_clean)

        # Effective domain should include noisy image
        @test issubset_box(img_noisy, eff_domain)

        # But still smaller than initial domain
        @test box_measure(eff_domain) < box_measure(domain)
    end

    @testset "create_restricted_partition" begin
        # Contracting map
        F(x, y) = (0.6*x + 0.2, 0.6*y + 0.2)
        domain = IntervalBox(interval(0.0, 1.0), interval(0.0, 1.0))

        partition, eff_domain, info = create_restricted_partition(
            F, domain, nothing,
            partition_depth=3,
            subdivision_depth=1,
            verbose=false
        )

        # Check reduction achieved
        @test info.reduction_factor > 1.0

        # Partition should have boxes
        @test partition.n_boxes > 0
        @test partition.n_boxes == 4^3  # 3 levels of quadrisection

        # All boxes should be in effective domain
        @test all(issubset_box(box, eff_domain) for box in partition.boxes)

        # Effective domain should be smaller than initial
        @test box_measure(eff_domain) < box_measure(domain)
    end

    @testset "create_restricted_partition with noise" begin
        F(x, y) = (0.5*x + 0.25, 0.5*y + 0.25)
        domain = IntervalBox(interval(0.0, 1.0), interval(0.0, 1.0))
        noise = uniform_noise_box(0.02, 2)

        partition, eff_domain, info = create_restricted_partition(
            F, domain, noise,
            partition_depth=2,
            subdivision_depth=1,
            safety_margin=1.2,
            verbose=false
        )

        # Should have significant reduction
        @test info.reduction_factor > 1.5

        # Noise should have expanded the image
        @test box_measure(info.image_noisy) > box_measure(info.image_clean)

        # Partition should cover effective domain
        total_coverage = sum(box_measure(box) for box in partition.boxes)
        @test total_coverage ≈ box_measure(eff_domain) atol=1e-10
    end

    @testset "restrict_partition_to_domain" begin
        # Create a large partition
        domain_large = IntervalBox(interval(0.0, 1.0), interval(0.0, 1.0))
        boxes = quadrisect(domain_large)
        for _ in 1:2
            new_boxes = eltype(boxes)[]
            for box in boxes
                append!(new_boxes, quadrisect(box))
            end
            boxes = new_boxes
        end

        # 64 boxes total
        @test length(boxes) == 64

        # Restrict to smaller domain
        restricted_domain = IntervalBox(interval(0.3, 0.7), interval(0.3, 0.7))
        restricted_boxes = restrict_partition_to_domain(boxes, restricted_domain)

        # Should have fewer boxes
        @test length(restricted_boxes) < length(boxes)

        # All restricted boxes should intersect the domain
        @test all(!isdisjoint_box(box, restricted_domain) for box in restricted_boxes)

        # None of the excluded boxes should be inside restricted domain
        # (can't use setdiff because it calls union! which isn't supported for intervals)
        excluded = [box for box in boxes if !(box in restricted_boxes)]
        @test all(isdisjoint_box(box, restricted_domain) for box in excluded)
    end

    @testset "domain restriction efficiency comparison" begin
        # Show the computational savings
        F(x, y) = (0.4*x + 0.3, 0.4*y + 0.3)  # Strong contraction
        domain = IntervalBox(interval(0.0, 1.0), interval(0.0, 1.0))

        # Full domain partition
        depth_full = 3
        n_boxes_full = 4^depth_full  # 64 boxes

        # Restricted domain partition
        partition_restricted, eff_dom, info = create_restricted_partition(
            F, domain, nothing,
            partition_depth=3,
            subdivision_depth=2,
            verbose=false
        )

        # Matrix size comparison
        # Full matrix would be n_boxes_full × n_boxes_full
        # Restricted is n_restricted × n_restricted
        n_restricted = partition_restricted.n_boxes

        matrix_size_full = n_boxes_full^2
        matrix_size_restricted = n_restricted^2

        # Should have significant savings (both in partition size and matrix size)
        @test n_restricted <= n_boxes_full
        @test matrix_size_restricted <= matrix_size_full

        println("  Efficiency comparison:")
        println("    Full partition: $n_boxes_full boxes, matrix size $matrix_size_full")
        println("    Restricted: $n_restricted boxes, matrix size $matrix_size_restricted")
        println("    Reduction factor: $(round(info.reduction_factor, digits=2))×")
        println("    Matrix size reduction: $(round(matrix_size_full/matrix_size_restricted, digits=2))×")
    end

    @testset "diameter-based partition (PDF error bound)" begin
        # Test diameter-based stopping criterion from PDF theory
        F(x, y) = (0.5*x + 0.25, 0.5*y + 0.25)
        domain = IntervalBox(interval(0.0, 1.0), interval(0.0, 1.0))

        # Create partition with diameter constraint
        target_diameter = 0.1
        partition, eff_dom, info = create_restricted_partition(
            F, domain, nothing,
            max_diameter=target_diameter,
            verbose=false
        )

        # Check that partition diameter satisfies constraint
        @test info.partition_diameter <= target_diameter
        @test info.partition_diameter > 0

        # Verify all boxes satisfy diameter constraint
        for box in partition.boxes
            @test box_width(box) <= target_diameter + 1e-10  # small tolerance for floating point
        end

        println("  Diameter-based partition:")
        println("    Target diameter: $target_diameter")
        println("    Achieved diameter: $(round(info.partition_diameter, digits=6))")
        println("    Number of boxes: $(info.n_boxes)")
    end

    @testset "diameter vs depth comparison" begin
        # Compare diameter-based vs depth-based approaches
        F(x, y) = (0.4*x + 0.3, 0.4*y + 0.3)
        domain = IntervalBox(interval(0.0, 1.0), interval(0.0, 1.0))

        # Depth-based
        partition_depth, _, info_depth = create_restricted_partition(
            F, domain, nothing,
            partition_depth=4,
            verbose=false
        )

        # Diameter-based (target similar diameter)
        target_diam = info_depth.partition_diameter
        partition_diam, _, info_diam = create_restricted_partition(
            F, domain, nothing,
            max_diameter=target_diam,
            verbose=false
        )

        # Should achieve similar partition sizes
        @test abs(info_depth.partition_diameter - info_diam.partition_diameter) < 0.1

        println("  Comparison:")
        println("    Depth-based (depth=4): $(info_depth.n_boxes) boxes, diam=$(round(info_depth.partition_diameter, digits=6))")
        println("    Diameter-based (target=$(round(target_diam, digits=6))): $(info_diam.n_boxes) boxes, diam=$(round(info_diam.partition_diameter, digits=6))")
    end

    @testset "diameter-based with noise" begin
        # Test diameter-based approach with noise
        F(x, y) = (0.5*x + 0.25, 0.5*y + 0.25)
        domain = IntervalBox(interval(0.0, 1.0), interval(0.0, 1.0))
        noise = uniform_noise_box(0.05, 2)

        target_diameter = 0.05
        partition, eff_dom, info = create_restricted_partition(
            F, domain, noise,
            max_diameter=target_diameter,
            verbose=false
        )

        # Verify constraint
        @test info.partition_diameter <= target_diameter

        # Check that noise was properly incorporated
        @test box_measure(info.image_noisy) > box_measure(info.image_clean)

        println("  Diameter-based with noise:")
        println("    Target diameter: $target_diameter")
        println("    Achieved diameter: $(round(info.partition_diameter, digits=6))")
        println("    Boxes: $(info.n_boxes)")
        println("    Noise expansion: $(round(box_measure(info.image_noisy)/box_measure(info.image_clean), digits=2))×")
    end

    @testset "compute_effective_partition_adaptive - fixed levels" begin
        println("\n--- Testing adaptive refinement with fixed levels ---")

        F(x, y) = (0.5*x + 0.25, 0.5*y + 0.25)
        domain = IntervalBox(interval(0.0, 1.0), interval(0.0, 1.0))
        noise = uniform_noise_box(0.01, 2)

        # Test with fixed refinement levels
        partition, info = compute_effective_partition_adaptive(
            F, domain, noise,
            initial_depth=2,
            refinement_levels=2,
            verbose=true
        )

        # Basic checks
        @test partition.n_boxes > 0
        @test info.refinement_levels == 2
        @test info.reduction_factor > 1.0  # Effective domain should be smaller

        println("  Fixed levels test:")
        println("    Initial depth: 2, Refinement levels: 2")
        println("    Final boxes: $(partition.n_boxes)")
        println("    Reduction: $(round(info.reduction_factor, digits=2))×")
    end

    @testset "compute_effective_partition_adaptive - diameter-based" begin
        println("\n--- Testing adaptive refinement with diameter stopping ---")

        F(x, y) = (0.5*x + 0.25, 0.5*y + 0.25)
        domain = IntervalBox(interval(0.0, 1.0), interval(0.0, 1.0))
        noise = uniform_noise_box(0.01, 2)

        target_diameter = 0.05

        # Test with diameter-based stopping
        partition, info = compute_effective_partition_adaptive(
            F, domain, noise,
            initial_depth=2,
            max_diameter=target_diameter,
            verbose=true
        )

        # Verify diameter constraint
        @test info.partition_diameter <= target_diameter

        # Basic properties
        @test partition.n_boxes > 0
        @test info.reduction_factor > 1.0  # Effective domain should be smaller

        println("  Diameter-based test:")
        println("    Target diameter: $target_diameter")
        println("    Achieved diameter: $(round(info.partition_diameter, digits=6))")
        println("    Final boxes: $(partition.n_boxes)")
        println("    Refinement levels used: $(info.refinement_levels)")
    end

    @testset "compute_effective_partition_adaptive - comparison" begin
        println("\n--- Comparing refinement strategies ---")

        F(x, y) = (0.4*x + 0.3, 0.4*y + 0.3)
        domain = IntervalBox(interval(0.0, 1.0), interval(0.0, 1.0))

        # Fixed levels
        partition_fixed, info_fixed = compute_effective_partition_adaptive(
            F, domain, nothing,
            initial_depth=3,
            refinement_levels=2,
            verbose=false
        )

        # Diameter-based
        target_diameter = 0.03
        partition_diameter, info_diameter = compute_effective_partition_adaptive(
            F, domain, nothing,
            initial_depth=3,
            max_diameter=target_diameter,
            verbose=false
        )

        # Both should reduce the domain
        @test info_fixed.reduction_factor > 1.0
        @test info_diameter.reduction_factor > 1.0

        # Diameter-based should respect constraint
        @test info_diameter.partition_diameter <= target_diameter

        println("  Comparison:")
        println("    Fixed (2 levels): $(partition_fixed.n_boxes) boxes, diameter=$(round(info_fixed.partition_diameter, digits=6))")
        println("    Diameter-based: $(partition_diameter.n_boxes) boxes, diameter=$(round(info_diameter.partition_diameter, digits=6))")
    end

    @testset "compute_effective_partition_adaptive - error handling" begin
        F(x, y) = (0.5*x, 0.5*y)
        domain = IntervalBox(interval(0.0, 1.0), interval(0.0, 1.0))

        # Should error if neither stopping criterion specified
        @test_throws ErrorException compute_effective_partition_adaptive(
            F, domain, nothing,
            initial_depth=2
        )
    end

    @testset "compute_effective_partition_adaptive - convergence" begin
        println("\n--- Testing convergence behavior ---")

        F(x, y) = (0.3*x + 0.35, 0.3*y + 0.35)
        domain = IntervalBox(interval(0.0, 1.0), interval(0.0, 1.0))

        # Test multiple diameter targets
        diameters = [0.1, 0.05, 0.025]

        for target_diam in diameters
            partition, info = compute_effective_partition_adaptive(
                F, domain, nothing,
                initial_depth=2,
                max_diameter=target_diam,
                verbose=false
            )

            @test info.partition_diameter <= target_diam
            println("  Target=$target_diam: achieved=$(round(info.partition_diameter, digits=6)), boxes=$(info.n_boxes_final), levels=$(info.refinement_levels)")
        end

        # Finer diameter should require more boxes
        partition_coarse, info_coarse = compute_effective_partition_adaptive(
            F, domain, nothing,
            initial_depth=2,
            max_diameter=0.1,
            verbose=false
        )

        partition_fine, info_fine = compute_effective_partition_adaptive(
            F, domain, nothing,
            initial_depth=2,
            max_diameter=0.02,
            verbose=false
        )

        @test info_fine.n_boxes_final >= info_coarse.n_boxes_final
    end
end
