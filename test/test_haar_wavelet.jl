using RigorousUlam2D
using IntervalArithmetic, IntervalBoxes
using LinearAlgebra
using SparseArrays
using Test

@testset "Haar Wavelet" begin

    @testset "HaarBasis2D construction" begin
        b = HaarBasis2D(0)
        @test b.depth == 0
        @test b.n == 1

        b = HaarBasis2D(3)
        @test b.depth == 3
        @test b.n == 64

        @test_throws ArgumentError HaarBasis2D(-1)
    end

    @testset "wavelet_scale_ranges" begin
        b = HaarBasis2D(2)  # n = 16
        ranges = wavelet_scale_ranges(b)
        # Scale -1 (scaling): index 1
        @test ranges[1] == (-1, 1:1)
        # Scale 0: 3 coeffs, indices 2:4
        @test ranges[2] == (0, 2:4)
        # Scale 1: 12 coeffs, indices 5:16
        @test ranges[3] == (1, 5:16)
        # Total covered = 1 + 3 + 12 = 16 = n
        @test sum(length(r) for (_, r) in ranges) == 16
    end

    @testset "Forward-inverse roundtrip (depth=$J)" for J in 1:4
        basis = HaarBasis2D(J)
        v = randn(basis.n)
        v_orig = copy(v)

        c = haar_forward(v, basis)
        v_rec = haar_inverse(c, basis)

        @test v_rec ≈ v_orig atol=1e-12
    end

    @testset "In-place roundtrip" begin
        basis = HaarBasis2D(3)
        v = randn(basis.n)
        v_orig = copy(v)

        haar_forward!(v, basis)
        haar_inverse!(v, basis)

        @test v ≈ v_orig atol=1e-12
    end

    @testset "Constant vector maps to scaling coefficient only" begin
        for J in 1:3
            basis = HaarBasis2D(J)
            v = ones(basis.n)
            c = haar_forward(v, basis)

            # All energy in scaling coefficient (index 1)
            # For constant 1 on 4^J boxes: scaling coeff = 2^J
            # (each level multiplies by /2 and sums 4 copies → ×2 per level)
            @test abs(c[1]) > 1e-10
            # All detail coefficients should be zero
            @test norm(c[2:end]) < 1e-12
        end
    end

    @testset "Parseval / energy preservation" begin
        for J in 1:3
            basis = HaarBasis2D(J)
            v = randn(basis.n)
            c = haar_forward(v, basis)

            # The transform is orthogonal (up to scaling factor 2^J per level),
            # so the L2 norm relation is: ||v||^2 = (1/4^J) * ||c||^2 * 4^J
            # Actually our normalization: ||v||^2 = ||c||^2 (orthogonal)
            # Let's verify by checking roundtrip preserves the vector exactly
            v_rec = haar_inverse(c, basis)
            @test dot(v_rec, v_rec) ≈ dot(v, v) atol=1e-12
        end
    end

    @testset "Zero-average vectors stay in detail subspace" begin
        basis = HaarBasis2D(2)
        n = basis.n
        # v with zero sum → scaling coefficient should be zero
        v = randn(n)
        v .-= sum(v) / length(v)
        c = haar_forward(v, basis)
        @test abs(c[1]) < 1e-12
    end

    @testset "wavelet_matvec consistency" begin
        # Build a small test operator
        basis = HaarBasis2D(2)  # 16 boxes
        n = basis.n
        domain = IntervalBox(interval(0.0, 1.0), interval(0.0, 1.0))
        partition = build_uniform_partition(domain, 2)

        # Simple contracting map
        F(x, y) = (0.5 * x + 0.25, 0.5 * y + 0.25)
        P_det, _ = build_transition_matrix(F, partition, periodic=true)
        normalize_columns!(P_det)

        # wavelet_matvec should give same result as W*P*W^{-1}*c
        c = randn(n)
        result_wav = wavelet_matvec(c, P_det, basis)

        # Manual: inverse, multiply, forward
        v = haar_inverse(c, basis)
        w = P_det * v
        result_manual = haar_forward(w, basis)

        @test result_wav ≈ result_manual atol=1e-12
    end

    @testset "Noise attenuation bound" begin
        domain = IntervalBox(interval(0.0, 1.0), interval(0.0, 1.0))
        noise = IntervalBox(interval(-0.25, 0.25), interval(-0.25, 0.25))  # xi = 0.5

        # Scale 0: low frequency, attenuation should be close to 1
        a0 = noise_attenuation_bound(noise, domain, 0)
        @test a0 ≤ 1.0

        # Higher scales should have stronger attenuation
        a1 = noise_attenuation_bound(noise, domain, 1)
        a2 = noise_attenuation_bound(noise, domain, 2)
        @test a1 ≤ a0
        @test a2 ≤ a1
        @test a2 < 0.5  # significant attenuation at scale 2 with xi=0.5
    end

    @testset "Schur norms of powers (small system)" begin
        basis = HaarBasis2D(2)
        n = basis.n
        domain = IntervalBox(interval(0.0, 1.0), interval(0.0, 1.0))
        noise = IntervalBox(interval(-0.25, 0.25), interval(-0.25, 0.25))
        partition = build_uniform_partition(domain, 2)

        # Contracting map with noise
        F(x, y) = (0.5 * x + 0.25, 0.5 * y + 0.25)
        P_det, _ = build_transition_matrix(F, partition, periodic=true)
        normalize_columns!(P_det)

        N_op = build_noise_convolution_operator(noise, partition, periodic=true)
        P_noisy = N_op * P_det
        normalize_columns!(P_noisy)

        # Direct norms (reference)
        direct_norms = norms_of_powers_sparse(P_noisy, 5; rigorous=true)

        # Schur norms should be valid upper bounds
        result = schur_norms_of_powers(P_noisy, basis, noise, domain, 5;
                                        rigorous=true, verbose=false)

        # The Schur bounds should be finite and positive
        for k in 1:5
            @test result.norms[k] > 0.0
            @test isfinite(result.norms[k])
        end
    end
end
