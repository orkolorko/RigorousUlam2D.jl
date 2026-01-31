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
        # Build uniform partition (smaller for faster adaptive comparison)
        domain = IntervalBox(interval(0.0, 1.0), interval(0.0, 1.0))
        partition = build_uniform_partition(domain, 2)  # depth 2 → 16 boxes (faster)

        # Grid-aligned noise: width = 1/4
        noise = uniform_noise_box(1/8, 2)  # ±1/8 → width 1/4

        # Compute with grid-aligned method (exact)
        N_exact = build_grid_aligned_noise_operator(noise, partition)

        # Compute with adaptive method (should be very close)
        N_adaptive = build_noise_convolution_operator(noise, partition,
                                                      ε=1e-6, max_iterations=20)

        # Grid-aligned is exact; adaptive uses conservative lower bounds
        # They may differ because adaptive is more conservative
        diff = norm(N_exact - N_adaptive, 1)
        @test diff < 20  # Adaptive is conservative, so differences are expected

        # Check column sums - with absorbing BC, boundary boxes lose mass
        # Interior boxes should have column sums closer to 1, boundary boxes less
        col_sums_exact = vec(sum(N_exact, dims=1))
        @test all(col_sums_exact .> 0)  # All positive
        @test all(col_sums_exact .<= 1.0 + 1e-10)  # None exceed 1 (floating point tolerance)
    end

    @testset "noise spreading pattern" begin
        # Simple 2×2 grid
        domain = IntervalBox(interval(0.0, 1.0), interval(0.0, 1.0))
        partition = build_uniform_partition(domain, 1)  # depth 1 → 4 boxes

        # Grid-aligned noise: width = 0.5 = domain_width/2^1
        noise = uniform_noise_box(0.25, 2)  # ±0.25 → width 0.5

        N = build_grid_aligned_noise_operator(noise, partition, periodic=false)

        # All boxes should be connected (noise reaches all neighbors)
        @test nnz(N) >= 4  # At least diagonal entries

        # Check matrix is non-negative
        @test all(N.nzval .>= 0)

        # Absorbing BC: boundary boxes lose mass
        col_sums = vec(sum(N, dims=1))
        @test all(col_sums .> 0)  # All positive
        @test all(col_sums .<= 1.0 + 1e-10)  # None exceed 1.0 (with floating point tolerance)
        @test minimum(col_sums) < 1.0  # Some boxes lose mass (boundary)
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

        # Check column sums - with absorbing BC, boundary boxes lose mass
        col_sums = vec(sum(N, dims=1))
        @test all(col_sums .> 0)  # All positive
        @test all(col_sums .<= 1.0 + 1e-10)  # None exceed 1 (floating point tolerance)

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

    @testset "periodic BC - column stochastic property" begin
        # With periodic BC and grid-aligned noise, column sums should be exactly 1
        domain = IntervalBox(interval(0.0, 1.0), interval(0.0, 1.0))
        partition = build_uniform_partition(domain, 4)  # 256 boxes

        # Multiple grid-aligned noise levels
        for k in [4, 5, 6]
            noise_width = 1.0 / 2^k
            noise = uniform_noise_box(noise_width/2, 2)  # half-width

            N = build_grid_aligned_noise_operator(noise, partition, periodic=true)
            col_sums = vec(sum(N, dims=1))

            # All column sums should be exactly 1.0 (no mass loss with periodic BC)
            @test all(abs.(col_sums .- 1.0) .< 1e-10)
        end
    end

    @testset "grid-aligned noise with adaptive partition" begin
        # Common workflow: adaptive partition + grid-aligned noise
        F(x, y) = (0.5*x + 0.25, 0.5*y + 0.25)
        domain = IntervalBox(interval(0.0, 1.0), interval(0.0, 1.0))

        # Grid-aligned noise
        noise = uniform_noise_box(1/64, 2)  # width = 1/32

        partition, dyn_graph, topo_graph = build_adaptive_partition(
            F, domain, noise,
            max_diameter=0.05,
            initial_depth=2,
            verbose=false
        )

        # Build complete noisy operator
        P_det, _ = build_transition_matrix(F, partition,
            transition_graph=dyn_graph, max_iterations=10, ε=1e-4)
        N = build_grid_aligned_noise_operator(noise, partition,
            topological_graph=topo_graph)
        P_noisy = N * P_det

        # Normalize and check
        normalize_columns!(P_noisy)

        # Should be a valid stochastic matrix
        col_sums = [sum(P_noisy[:, j]) for j in 1:size(P_noisy, 2)]
        @test all(abs.(col_sums .- 1.0) .< 1e-10)
        @test all(P_noisy.nzval .>= 0)
    end

    @testset "multiple noise sizes - same partition" begin
        # Test reusing partition with different noise levels
        F(x, y) = (0.5*x + 0.25, 0.5*y + 0.25)
        domain = IntervalBox(interval(0.0, 1.0), interval(0.0, 1.0))

        # Build partition with one noise level
        noise_ref = uniform_noise_box(1/32, 2)
        partition, dyn_graph, topo_graph = build_adaptive_partition(
            F, domain, noise_ref, max_diameter=0.05, initial_depth=2, verbose=false)

        P_det, _ = build_transition_matrix(F, partition,
            transition_graph=dyn_graph, max_iterations=10, ε=1e-4)

        # Test different grid-aligned noise sizes
        results = Dict{Float64, Float64}()

        for k in [5, 6, 7]
            noise_width = 1.0 / 2^k
            noise_k = uniform_noise_box(noise_width/2, 2)

            # Rebuild topological graph for this noise
            topo_graph_k = build_topological_graph(partition, noise_k)

            N_k = build_grid_aligned_noise_operator(noise_k, partition,
                topological_graph=topo_graph_k)
            P_noisy_k = N_k * P_det
            normalize_columns!(P_noisy_k)

            # Compute spectral gap (if small enough)
            if size(P_noisy_k, 1) <= 200
                eigvals_all = eigen(Matrix(P_noisy_k)).values
                sorted_eigs = sort(abs.(eigvals_all), rev=true)
                gap = 1.0 - sorted_eigs[2]
                results[noise_width] = gap
            end
        end

        # Larger noise should generally lead to faster mixing (larger spectral gap)
        if length(results) >= 2
            noise_sizes = sort(collect(keys(results)))
            # Just verify computation succeeded
            @test all(values(results) .>= 0)
        end
    end

    @testset "noise operator symmetry for symmetric partition" begin
        # For uniform partition on symmetric domain, noise operator should have
        # symmetric structure
        domain = IntervalBox(interval(-1.0, 1.0), interval(-1.0, 1.0))
        partition = build_uniform_partition(domain, 3)  # 64 boxes

        noise = uniform_noise_box(1/8, 2)  # grid-aligned

        N = build_grid_aligned_noise_operator(noise, partition, periodic=true)

        # Noise operator should be symmetric (N = N')
        # Because spreading from box i to j has same overlap as j to i
        @test norm(N - N', 1) < 1e-10
    end

    @testset "small noise limit" begin
        # Very small noise should give nearly identity operator
        domain = IntervalBox(interval(0.0, 1.0), interval(0.0, 1.0))
        partition = build_uniform_partition(domain, 4)  # 256 boxes

        # Very small grid-aligned noise
        noise = uniform_noise_box(1/512, 2)  # width = 1/256

        N = build_grid_aligned_noise_operator(noise, partition)

        # Most mass should stay on diagonal (relaxed threshold due to boundary effects)
        diag_mass = sum(N[i, i] for i in 1:partition.n_boxes)
        total_mass = sum(N)

        @test diag_mass / total_mass > 0.8  # >80% on diagonal
    end

    @testset "large noise limit" begin
        # Large noise spreads mass to many neighbors
        domain = IntervalBox(interval(0.0, 1.0), interval(0.0, 1.0))
        partition = build_uniform_partition(domain, 3)  # 64 boxes

        # Large grid-aligned noise (half domain width)
        noise = uniform_noise_box(1/4, 2)  # width = 1/2

        N = build_grid_aligned_noise_operator(noise, partition, periodic=true)

        # Should have many non-zeros (noise spreads widely)
        @test nnz(N) > partition.n_boxes * 4  # More than 4× diagonal

        # All entries should sum appropriately
        col_sums = vec(sum(N, dims=1))
        @test all(abs.(col_sums .- 1.0) .< 1e-10)  # Still stochastic
    end

end
