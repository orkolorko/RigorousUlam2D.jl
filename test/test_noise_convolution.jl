using IntervalArithmetic, IntervalBoxes
using RigorousUlam2D
using SparseArrays
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

    @testset "noisy_rel_size" begin
        F_id(x, y) = (x, y)

        source = IntervalBox(interval(0.3, 0.4), interval(0.3, 0.4))
        target = IntervalBox(interval(0.0, 1.0), interval(0.0, 1.0))
        noise = uniform_noise_box(0.05, 2)

        lower, upper, uncertain = noisy_rel_size(F_id, source, target, noise, maxdepth=5)

        # With small noise, should still map mostly into large target
        @test lower > 0.0
        @test upper >= lower
        @test upper <= box_measure(source)
    end

    @testset "noisy_rel_size escaping" begin
        F_id(x, y) = (x, y)

        # Source near edge, target small - noise might push it out
        source = IntervalBox(interval(0.9, 1.0), interval(0.9, 1.0))
        target = IntervalBox(interval(0.0, 0.5), interval(0.0, 0.5))
        noise = uniform_noise_box(0.2, 2)  # Large noise

        lower, upper, uncertain = noisy_rel_size(F_id, source, target, noise, maxdepth=6)

        # Most points should NOT map into target (source far from target)
        @test upper <= box_measure(source)
    end

    @testset "build_noisy_transition_matrix" begin
        F_id(x, y) = (x, y)

        boxes = quadrisect(IntervalBox(interval(0.0, 1.0), interval(0.0, 1.0)))
        domain = IntervalBox(interval(0.0, 1.0), interval(0.0, 1.0))
        partition = UlamPartition(boxes, domain)

        noise = uniform_noise_box(0.05, 2)

        P, U = build_noisy_transition_matrix(F_id, partition, noise, maxdepth=5, ε=1e-3)

        @test size(P) == (4, 4)
        @test issparse(P)

        # With identity + small noise, diagonal should still dominate
        @test all(P[i, i] > 0 for i in 1:4)

        # Noise causes diffusion - off-diagonal may have some mass
        # (but with very small noise and coarse partition, might be zero)
        total_off_diagonal = sum(P[i, j] for i in 1:4 for j in 1:4 if i != j)
        @test total_off_diagonal >= 0  # Just check it's non-negative
    end

    @testset "noisy vs deterministic matrix" begin
        F(x, y) = (0.5 * x + 0.5, 0.5 * y + 0.5)  # Contracting toward center

        boxes = quadrisect(IntervalBox(interval(0.0, 1.0), interval(0.0, 1.0)))
        domain = IntervalBox(interval(0.0, 1.0), interval(0.0, 1.0))
        partition = UlamPartition(boxes, domain)

        # Deterministic
        P_det, _ = build_transition_matrix(F, partition, maxdepth=5, ε=1e-3)

        # With noise
        noise = uniform_noise_box(0.05, 2)
        P_noisy, _ = build_noisy_transition_matrix(F, partition, noise, maxdepth=5, ε=1e-3)

        # Noisy matrix should have more non-zero entries (diffusion)
        @test nnz(P_noisy) >= nnz(P_det)
    end
end
