using Test
using IntervalArithmetic, IntervalBoxes
using RigorousUlam2D
using SparseArrays
using LinearAlgebra

@testset "Ulam Discretization Core Tests" begin

    @testset "rel_size accuracy" begin
        @testset "identity map - full coverage" begin
            F_id(x, y) = (x, y)
            source = IntervalBox(interval(0.2, 0.4), interval(0.3, 0.5))
            target = IntervalBox(interval(0.0, 1.0), interval(0.0, 1.0))

            lower, upper, uncertain = rel_size(F_id, source, target, max_iterations=10, ε=1e-6)
            @test lower ≈ 1.0
            @test upper ≈ 1.0
            @test isempty(uncertain)
        end

        @testset "identity map - partial coverage" begin
            F_id(x, y) = (x, y)
            source = IntervalBox(interval(0.0, 1.0), interval(0.0, 1.0))
            target = IntervalBox(interval(0.0, 0.5), interval(0.0, 0.5))

            lower, upper, uncertain = rel_size(F_id, source, target, max_iterations=10, ε=1e-4)
            # 1/4 of source maps to target
            @test lower ≈ 0.25 atol=0.01
            @test upper >= lower
        end

        @testset "scaling map" begin
            F_scale(x, y) = (0.5 * x, 0.5 * y)
            source = IntervalBox(interval(0.0, 1.0), interval(0.0, 1.0))
            target = IntervalBox(interval(0.0, 0.5), interval(0.0, 0.5))

            lower, upper, uncertain = rel_size(F_scale, source, target, max_iterations=15, ε=1e-4)
            # All of source maps to target
            @test lower ≈ 1.0 atol=0.01
            @test upper >= lower
        end

        @testset "translation map - no overlap" begin
            F_trans(x, y) = (x + 5.0, y + 5.0)
            source = IntervalBox(interval(0.0, 1.0), interval(0.0, 1.0))
            target = IntervalBox(interval(0.0, 1.0), interval(0.0, 1.0))

            lower, upper, uncertain = rel_size(F_trans, source, target, max_iterations=5, ε=1e-4)
            @test lower == 0.0
            @test upper == 0.0
            @test isempty(uncertain)
        end
    end

    @testset "Transition matrix properties" begin
        @testset "identity map - diagonal dominance" begin
            F_id(x, y) = (x, y)
            domain = IntervalBox(interval(0.0, 1.0), interval(0.0, 1.0))
            partition = build_uniform_partition(domain, 2)  # 16 boxes

            P, U = build_transition_matrix(F_id, partition, max_iterations=10, ε=1e-4)

            # Identity map should be diagonal
            for i in 1:partition.n_boxes
                @test P[i, i] > 0
            end

            # Column sums should equal box measures
            for j in 1:partition.n_boxes
                col_sum = sum(P[:, j])
                box_m = box_measure(partition.boxes[j])
                @test col_sum ≈ box_m atol=1e-8
            end
        end

        @testset "contracting map - mass conservation" begin
            F(x, y) = (0.5*x + 0.25, 0.5*y + 0.25)
            domain = IntervalBox(interval(0.0, 1.0), interval(0.0, 1.0))

            partition, dyn_graph, _ = build_adaptive_partition(F, domain, nothing,
                max_diameter=0.1, initial_depth=2, verbose=false)

            P, U = build_transition_matrix(F, partition,
                transition_graph=dyn_graph, max_iterations=10, ε=1e-4)

            # Contracting map keeps all mass in domain
            for j in 1:partition.n_boxes
                col_sum = sum(P[:, j])
                box_m = box_measure(partition.boxes[j])
                @test col_sum ≈ box_m atol=1e-8
            end
        end

        @testset "normalized matrix is stochastic" begin
            F(x, y) = (0.5*x + 0.25, 0.5*y + 0.25)
            domain = IntervalBox(interval(0.0, 1.0), interval(0.0, 1.0))
            partition = build_uniform_partition(domain, 2)

            P, U = build_transition_matrix(F, partition, max_iterations=10, ε=1e-4)
            normalize_columns!(P)

            # All column sums should be 1
            for j in 1:partition.n_boxes
                @test sum(P[:, j]) ≈ 1.0 atol=1e-10
            end

            # All entries should be non-negative
            @test all(P.nzval .>= 0)
        end
    end

    @testset "Periodic BC correctness" begin
        @testset "wrap_to_torus" begin
            # Test scalar wrapping
            @test wrap_to_torus(1.2, 0.0, 1.0) ≈ 0.2
            @test wrap_to_torus(-0.3, 0.0, 1.0) ≈ 0.7
            @test wrap_to_torus(2.5, -1.0, 1.0) ≈ 0.5

            # Test interval wrapping (non-wrapping case)
            I = interval(0.2, 0.4)
            wrapped = wrap_to_torus(I, 0.0, 1.0)
            @test inf(wrapped) ≈ 0.2
            @test sup(wrapped) ≈ 0.4
        end

        @testset "periodic BC mass conservation" begin
            F(x, y) = (2*x, 2*y)  # Expanding map
            domain = IntervalBox(interval(-1.0, 1.0), interval(-1.0, 1.0))
            noise = uniform_noise_box(1/4, 2)

            partition, dyn_graph, topo_graph = build_adaptive_partition(
                F, domain, noise, max_diameter=0.5, initial_depth=2, periodic=true, verbose=false)

            P_det, U = build_transition_matrix(F, partition,
                transition_graph=dyn_graph, periodic=true, max_iterations=15, ε=1e-3)

            # With periodic BC, all mass should stay in domain
            col_sums = [sum(P_det[:, j]) for j in 1:size(P_det, 2)]
            box_measures = [box_measure(b) for b in partition.boxes]

            @test all(col_sums .> 0)
            ratios = col_sums ./ box_measures
            @test minimum(ratios) > 0.85  # Allow some numerical tolerance
        end
    end

    @testset "Operator composition" begin
        @testset "P_noisy = N * P" begin
            F(x, y) = (0.5*x + 0.25, 0.5*y + 0.25)
            domain = IntervalBox(interval(0.0, 1.0), interval(0.0, 1.0))
            noise = uniform_noise_box(1/32, 2)  # Grid-aligned noise

            partition, dyn_graph, topo_graph = build_adaptive_partition(
                F, domain, noise, max_diameter=0.1, initial_depth=2, verbose=false)

            # Build operators
            P_det, _ = build_transition_matrix(F, partition,
                transition_graph=dyn_graph, max_iterations=10, ε=1e-4)

            N = build_grid_aligned_noise_operator(noise, partition,
                topological_graph=topo_graph)

            # Compose
            P_noisy = N * P_det

            # Noise operator should spread mass
            @test nnz(P_noisy) >= nnz(P_det)

            # Verify noise operator is (nearly) column-stochastic
            n_col_sums = [sum(N[:, j]) for j in 1:size(N, 2)]
            @test minimum(n_col_sums) > 0.5  # Absorbing BC loses some mass at boundary
        end

        @testset "invariant measure convergence" begin
            F(x, y) = (0.5*x + 0.25, 0.5*y + 0.25)
            domain = IntervalBox(interval(0.0, 1.0), interval(0.0, 1.0))
            noise = uniform_noise_box(1/32, 2)

            partition, dyn_graph, topo_graph = build_adaptive_partition(
                F, domain, noise, max_diameter=0.05, initial_depth=2, verbose=false)

            P_det, _ = build_transition_matrix(F, partition,
                transition_graph=dyn_graph, max_iterations=10, ε=1e-4)
            N = build_grid_aligned_noise_operator(noise, partition,
                topological_graph=topo_graph)
            P_noisy = N * P_det

            normalize_columns!(P_noisy)

            # Compute eigenvalues (small matrix)
            if size(P_noisy, 1) <= 200
                eigvals_all = eigen(Matrix(P_noisy)).values
                sorted_eigs = sort(abs.(eigvals_all), rev=true)

                # Leading eigenvalue should be ~1
                @test sorted_eigs[1] ≈ 1.0 atol=1e-4

                # Second eigenvalue should be < 1 (mixing)
                if length(sorted_eigs) > 1
                    @test sorted_eigs[2] < 1.0
                end
            end
        end
    end

    @testset "Enclosure method comparison" begin
        @testset "both enclosures contain true image" begin
            # Both methods should produce valid enclosures
            F(x, y) = (x^2 + 0.1*y, 0.5*y + 0.1*x^2)
            box = IntervalBox(interval(0.2, 0.3), interval(0.2, 0.3))

            zeroth = zeroth_order_enclosure(F, box)
            first = first_order_enclosure(F, box)

            # Test a few sample points to verify they're contained in both enclosures
            for x_val in [0.2, 0.25, 0.3]
                for y_val in [0.2, 0.25, 0.3]
                    result = F(x_val, y_val)
                    # Check zeroth order contains result
                    @test inf(zeroth[1]) <= result[1] <= sup(zeroth[1])
                    @test inf(zeroth[2]) <= result[2] <= sup(zeroth[2])
                    # Check first order contains result
                    @test inf(first[1]) <= result[1] <= sup(first[1])
                    @test inf(first[2]) <= result[2] <= sup(first[2])
                end
            end
        end

        @testset "linear map - both methods equal" begin
            # For linear maps, both methods should give the same result
            F_linear(x, y) = (2*x + 3*y, x - y)
            box = IntervalBox(interval(0.0, 1.0), interval(0.0, 1.0))

            zeroth = zeroth_order_enclosure(F_linear, box)
            first = first_order_enclosure(F_linear, box)

            # Should be very close for linear maps
            @test inf(zeroth[1]) ≈ inf(first[1]) atol=1e-10
            @test sup(zeroth[1]) ≈ sup(first[1]) atol=1e-10
            @test inf(zeroth[2]) ≈ inf(first[2]) atol=1e-10
            @test sup(zeroth[2]) ≈ sup(first[2]) atol=1e-10
        end
    end

    @testset "Adaptive partition properties" begin
        @testset "forward flagging reduces boxes" begin
            # Strongly contracting map should have small effective domain
            F(x, y) = (0.2*x + 0.4, 0.2*y + 0.4)  # Contracts to (0.5, 0.5)
            domain = IntervalBox(interval(0.0, 1.0), interval(0.0, 1.0))

            partition_uniform = build_uniform_partition(domain, 4)  # 256 boxes

            partition_adaptive, _, _ = build_adaptive_partition(
                F, domain, nothing, max_diameter=0.1, initial_depth=2, verbose=false)

            # Adaptive should have fewer boxes than uniform (at same resolution)
            @test partition_adaptive.n_boxes < partition_uniform.n_boxes
        end

        @testset "partition covers effective domain" begin
            F(x, y) = (0.5*x + 0.25, 0.5*y + 0.25)
            domain = IntervalBox(interval(0.0, 1.0), interval(0.0, 1.0))

            partition, _, _ = build_adaptive_partition(
                F, domain, nothing, max_diameter=0.1, initial_depth=2, verbose=false)

            # Total measure should be <= domain measure
            total_measure = sum(box_measure(b) for b in partition.boxes)
            @test total_measure <= box_measure(domain) + 1e-10

            # All boxes should be subsets of domain
            for box in partition.boxes
                @test issubset_box(box, domain)
            end
        end
    end
end
