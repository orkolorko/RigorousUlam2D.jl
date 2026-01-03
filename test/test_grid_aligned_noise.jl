using Test
using RigorousUlam2D
using IntervalArithmetic, IntervalBoxes
using LinearAlgebra, SparseArrays

@testset "Grid-Aligned Noise Operator" begin

    @testset "grid alignment validation" begin
        # Simple uniform partition
        domain = IntervalBox(interval(0.0, 1.0), interval(0.0, 1.0))
        partition = build_uniform_partition(domain, 3)  # depth 3 → boxes of size 1/8

        # Grid-aligned noise: width = 1/8 = domain_width/2^3
        noise_aligned = uniform_noise_box(1/16, 2)  # ±1/16 → width 1/8 ✓

        # Should NOT error
        N = build_grid_aligned_noise_operator(noise_aligned, partition)
        @test size(N) == (partition.n_boxes, partition.n_boxes)

        # Non-aligned noise
        noise_misaligned = uniform_noise_box(0.01, 2)  # width 0.02 ≠ 1/2^k

        # Should error
        @test_throws ErrorException build_grid_aligned_noise_operator(noise_misaligned, partition)
    end

    @testset "exact vs adaptive comparison" begin
        # Build uniform partition
        domain = IntervalBox(interval(0.0, 1.0), interval(0.0, 1.0))
        partition = build_uniform_partition(domain, 4)  # depth 4 → boxes of size 1/16

        # Grid-aligned noise: width = 1/16
        noise = uniform_noise_box(1/32, 2)  # ±1/32 → width 1/16

        # Compute with grid-aligned method (exact)
        N_exact = build_grid_aligned_noise_operator(noise, partition)

        # Compute with adaptive method (should be very close)
        N_adaptive = build_noise_convolution_operator(noise, partition,
                                                      ε=1e-6, max_iterations=20)

        # Check they are very similar
        diff = norm(N_exact - N_adaptive, 1)
        @test diff < 0.01  # Should be nearly identical

        # Check column sums (should be ~1.0 for stochastic matrix)
        col_sums_exact = vec(sum(N_exact, dims=1))
        @test all(0.99 .< col_sums_exact .< 1.01)
    end

    @testset "noise spreading pattern" begin
        # Simple 2×2 grid
        domain = IntervalBox(interval(0.0, 1.0), interval(0.0, 1.0))
        partition = build_uniform_partition(domain, 1)  # depth 1 → 4 boxes

        # Grid-aligned noise: width = 0.5 = domain_width/2^1
        noise = uniform_noise_box(0.25, 2)  # ±0.25 → width 0.5

        N = build_grid_aligned_noise_operator(noise, partition)

        # All boxes should be connected (noise reaches all neighbors)
        @test nnz(N) >= 4  # At least diagonal entries

        # Check matrix is non-negative
        @test all(N.nzval .>= 0)

        # Check column sums approximately 1
        col_sums = vec(sum(N, dims=1))
        @test all(0.9 .< col_sums .< 1.1)
    end

    @testset "anisotropic grid-aligned noise" begin
        # Build uniform partition on rectangular domain
        domain = IntervalBox(interval(0.0, 1.0), interval(0.0, 2.0))
        partition = build_uniform_partition(domain, 3)  # depth 3

        # Anisotropic grid-aligned noise:
        # x direction: width = 1/8 = domain_x_width/2^3
        # y direction: width = 2/8 = domain_y_width/2^3
        noise = anisotropic_noise_box([1/16, 2/16])  # [±1/16, ±2/16]

        # Should work fine (each dimension grid-aligned independently)
        N = build_grid_aligned_noise_operator(noise, partition)
        @test size(N) == (partition.n_boxes, partition.n_boxes)

        # Check column sums
        col_sums = vec(sum(N, dims=1))
        @test all(0.9 .< col_sums .< 1.1)

        # Check matrix is non-negative
        @test all(N.nzval .>= 0)
    end

    @testset "periodic boundary conditions" begin
        # Build uniform partition
        domain = IntervalBox(interval(-1.0, 1.0), interval(-1.0, 1.0))
        partition = build_uniform_partition(domain, 2)  # depth 2 → boxes of size 0.5

        # Grid-aligned noise
        noise = uniform_noise_box(0.25, 2)  # ±0.25 → width 0.5

        # Non-periodic
        N_absorbing = build_grid_aligned_noise_operator(noise, partition, periodic=false)

        # Periodic
        N_periodic = build_grid_aligned_noise_operator(noise, partition, periodic=true)

        # Periodic should have more connections (wrapping around)
        @test nnz(N_periodic) >= nnz(N_absorbing)

        # Both should have valid column sums
        col_sums_abs = vec(sum(N_absorbing, dims=1))
        col_sums_per = vec(sum(N_periodic, dims=1))

        @test all(col_sums_abs .> 0)
        @test all(col_sums_per .> 0)
    end

    @testset "performance comparison" begin
        # Larger partition to see performance difference
        domain = IntervalBox(interval(0.0, 1.0), interval(0.0, 1.0))
        partition = build_uniform_partition(domain, 5)  # depth 5 → 1024 boxes

        # Grid-aligned noise
        noise = uniform_noise_box(1/64, 2)  # ±1/64 → width 1/32 = 1/2^5

        # Time grid-aligned method
        t_exact = @elapsed begin
            N_exact = build_grid_aligned_noise_operator(noise, partition)
        end

        # Time adaptive method
        t_adaptive = @elapsed begin
            N_adaptive = build_noise_convolution_operator(noise, partition,
                                                          ε=1e-4, max_iterations=10)
        end

        println("  Grid-aligned: $(round(t_exact, digits=3))s")
        println("  Adaptive: $(round(t_adaptive, digits=3))s")
        println("  Speedup: $(round(t_adaptive/t_exact, digits=1))×")

        # Grid-aligned should be significantly faster
        @test t_exact < t_adaptive
    end

end
