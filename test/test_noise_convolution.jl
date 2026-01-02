using IntervalArithmetic, IntervalBoxes
using RigorousUlam2D
using SparseArrays
using LinearAlgebra
using Test

@testset "Noise Convolution" begin

    @testset "minkowski_sum" begin
        A = IntervalBox(interval(0.0, 1.0), interval(0.0, 1.0))
        B = IntervalBox(interval(-0.1, 0.1), interval(-0.1, 0.1))

        C = minkowski_sum(A, B)

        @test inf(C[1]) ≈ -0.1
        @test sup(C[1]) ≈ 1.1
        @test inf(C[2]) ≈ -0.1
        @test sup(C[2]) ≈ 1.1

        # Measure should increase
        @test box_measure(C) > box_measure(A)
    end

    @testset "minkowski_sum with tuple" begin
        A_tuple = (interval(0.0, 1.0), interval(0.0, 1.0))
        B = IntervalBox(interval(-0.1, 0.1), interval(-0.1, 0.1))

        C = minkowski_sum(A_tuple, B)

        @test C isa IntervalBox
        @test inf(C[1]) ≈ -0.1
        @test sup(C[1]) ≈ 1.1
    end

    @testset "uniform_noise_box" begin
        noise = uniform_noise_box(0.05, 2)

        @test noise isa IntervalBox{2}
        @test inf(noise[1]) ≈ -0.05
        @test sup(noise[1]) ≈ 0.05
        @test inf(noise[2]) ≈ -0.05
        @test sup(noise[2]) ≈ 0.05
    end

    @testset "anisotropic_noise_box" begin
        noise = anisotropic_noise_box([0.1, 0.05])

        @test noise isa IntervalBox{2}
        @test diam(noise[1]) ≈ 0.2
        @test diam(noise[2]) ≈ 0.1
    end

    @testset "noisy_image" begin
        F(x, y) = (x, y)  # Identity map

        box = IntervalBox(interval(0.4, 0.6), interval(0.4, 0.6))
        noise = uniform_noise_box(0.1, 2)

        noisy = noisy_image(F, box, noise)

        # Should be deterministic image plus noise
        @test inf(noisy[1]) ≈ 0.3  # 0.4 - 0.1
        @test sup(noisy[1]) ≈ 0.7  # 0.6 + 0.1
        @test inf(noisy[2]) ≈ 0.3
        @test sup(noisy[2]) ≈ 0.7
    end

    # DEPRECATED: Tests for noisy_rel_size removed.
    # This function computed the annealed operator (F + noise) and violated the operator
    # decomposition architecture. Use operator composition instead: P_noisy = N * P

    @testset "build_noise_convolution_operator" begin
        noise = uniform_noise_box(0.05, 2)

        boxes = quadrisect(IntervalBox(interval(0.0, 1.0), interval(0.0, 1.0)))
        domain = IntervalBox(interval(0.0, 1.0), interval(0.0, 1.0))
        partition = UlamPartition(boxes, domain)

        N = build_noise_convolution_operator(noise, partition)

        @test size(N) == (4, 4)
        @test issparse(N)

        # Diagonal should dominate with small noise
        @test all(N[i, i] > 0 for i in 1:4)
    end

    @testset "noisy transition matrix via operator composition" begin
        F_id(x, y) = (x, y)

        boxes = quadrisect(IntervalBox(interval(0.0, 1.0), interval(0.0, 1.0)))
        domain = IntervalBox(interval(0.0, 1.0), interval(0.0, 1.0))
        partition = UlamPartition(boxes, domain)

        noise = uniform_noise_box(0.05, 2)

        # Build deterministic transition matrix
        P, U = build_transition_matrix(F_id, partition, max_iterations=5, ε=1e-3)

        # Build noise convolution operator
        N = build_noise_convolution_operator(noise, partition)

        # Compose: P_noisy = N * P
        P_noisy = N * P

        @test size(P_noisy) == (4, 4)
        @test issparse(P_noisy)

        # With identity + small noise, diagonal should still dominate
        @test all(P_noisy[i, i] > 0 for i in 1:4)

        # Noise causes diffusion - off-diagonal may have some mass
        total_off_diagonal = sum(P_noisy[i, j] for i in 1:4 for j in 1:4 if i != j)
        @test total_off_diagonal >= 0  # Just check it's non-negative
    end

    @testset "noisy vs deterministic matrix" begin
        F(x, y) = (0.5 * x + 0.5, 0.5 * y + 0.5)  # Contracting toward center

        boxes = quadrisect(IntervalBox(interval(0.0, 1.0), interval(0.0, 1.0)))
        domain = IntervalBox(interval(0.0, 1.0), interval(0.0, 1.0))
        partition = UlamPartition(boxes, domain)

        noise = uniform_noise_box(0.05, 2)

        # Deterministic
        P_det, _ = build_transition_matrix(F, partition, max_iterations=5, ε=1e-3)

        # Build noisy matrix using operator composition
        N = build_noise_convolution_operator(noise, partition)
        P_noisy = N * P_det

        # Noisy matrix should have more non-zero entries (diffusion)
        @test nnz(P_noisy) >= nnz(P_det)
    end

    @testset "contracting map stationary measure" begin
        # F(x,y) = (0.5x + 0.25, 0.5y + 0.25) contracts to fixed point (0.5, 0.5)
        # Theoretical stationary measure: noise kernel centered at (0.5, 0.5)
        F(x, y) = (0.5 * x + 0.25, 0.5 * y + 0.25)

        domain = IntervalBox(interval(0.0, 1.0), interval(0.0, 1.0))
        noise = uniform_noise_box(0.01, 2)  # Small noise

        # Build adaptive partition
        partition, graph = build_adaptive_partition(F, domain, noise,
                                                    initial_depth=2,
                                                    max_diameter=0.05)

        # Build operators
        P_det, _ = build_transition_matrix(F, partition, transition_graph=graph,
                                          max_iterations=8, ε=1e-4)
        N = build_noise_convolution_operator(noise, partition, transition_graph=graph,
                                            ε=1e-5, max_iterations=20)
        P_noisy = N * P_det

        # Normalize to get stochastic matrix
        normalize_columns!(P_noisy)

        # Compute invariant measure (leading eigenvector)
        eigvals, eigvecs = eigen(Matrix(P_noisy))

        # Find eigenvalue closest to 1
        idx = argmin(abs.(eigvals .- 1.0))
        invariant_measure = real(eigvecs[:, idx])
        invariant_measure ./= sum(invariant_measure)  # Normalize to probability

        # Theoretical: stationary measure should be concentrated near (0.5, 0.5)
        # Find boxes containing (0.5, 0.5)
        fixed_point = IntervalBox(interval(0.49, 0.51), interval(0.49, 0.51))
        boxes_near_fixed_point = findall(box -> !isdisjoint_box(box, fixed_point),
                                         partition.boxes)

        # Mass should concentrate near the fixed point
        mass_near_fixed_point = sum(invariant_measure[boxes_near_fixed_point])

        @test mass_near_fixed_point > 0.3  # Significant mass near fixed point
        @test abs(eigvals[idx] - 1.0) < 1e-6  # Leading eigenvalue should be ~1
    end
end
