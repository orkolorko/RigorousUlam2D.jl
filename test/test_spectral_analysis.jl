using Test
using RigorousUlam2D
using IntervalArithmetic, IntervalBoxes
using LinearAlgebra, SparseArrays

@testset "Spectral Analysis" begin

    @testset "higham_gamma - error bound constant" begin
        # γ_n = n*u / (1 - n*u) where u = eps/2
        u = eps(Float64) / 2
        γ_10 = higham_gamma(10, u)

        @test γ_10 > 0
        @test γ_10 ≈ 10*u / (1 - 10*u)

        # For small n, γ_n ≈ n*u
        @test abs(γ_10 - 10*u) < 1e-20
    end

    @testset "rigorous_norm_on_V - guaranteed upper bounds" begin
        # Simple stochastic matrix
        P = [0.8 0.3;
             0.2 0.7]

        # Rigorous norm should be >= numerical norm
        numerical_norm = operator_norm_on_V(P)
        rigorous_norm = rigorous_norm_on_V(P)

        @test rigorous_norm >= numerical_norm
        # The difference should be very small (just floating-point error bounds)
        @test rigorous_norm - numerical_norm < 1e-13

        # For larger matrix, check the bound makes sense
        n = 50
        P_large = zeros(n, n)
        for j in 1:n
            P_large[:, j] = fill(1.0/n, n)
        end

        rig_norm_large = rigorous_norm_on_V(P_large)
        @test rig_norm_large >= 0
        @test rig_norm_large < 1e-10  # Nearly zero for uniform columns
    end

    @testset "operator_norm_on_V - basic properties" begin
        # Simple 2x2 stochastic matrix
        P = [0.8 0.3;
             0.2 0.7]

        norm_V = operator_norm_on_V(P)

        # For this matrix, ||P|_V|| should be |0.8-0.7| = 0.5 (difference of diagonals)
        # Actually: ||P|_V|| = max_{j≠k} (1/2)||P_j - P_k|| = (1/2)*||(0.8-0.3, 0.2-0.7)|| = 0.5
        @test norm_V ≈ 0.5 atol=1e-10

        # Identity matrix should have ||I|_V|| = 0 (all columns are the same when restricted to V)
        # Wait, identity columns are different: e_1, e_2, ...
        # ||I|_V|| = max_{j≠k} (1/2)||e_j - e_k|| = 1
        I_mat = Matrix(1.0I, 3, 3)
        @test operator_norm_on_V(I_mat) ≈ 1.0 atol=1e-10

        # Matrix with all equal columns has ||P|_V|| = 0
        P_equal = ones(3, 3) / 3
        @test operator_norm_on_V(P_equal) ≈ 0.0 atol=1e-10
    end

    @testset "operator_norm_on_V - contraction" begin
        # Create a strongly mixing matrix
        n = 4
        P = zeros(n, n)
        for j in 1:n
            # Each column is a convex combination close to uniform
            P[:, j] = fill(0.2, n)
            P[j, j] = 0.4
        end

        # First power
        norm_1 = operator_norm_on_V(P)
        @test norm_1 < 1.0  # Should be a contraction

        # Higher power should give smaller norm
        norm_2 = operator_norm_on_V_power(P, 2)
        @test norm_2 < norm_1

        # Very high power should approach 0
        norm_10 = operator_norm_on_V_power(P, 10)
        @test norm_10 < 0.1
    end

    @testset "find_contraction_power" begin
        # Create a mixing matrix
        n = 4
        P = zeros(n, n)
        for j in 1:n
            P[:, j] = fill(0.15, n)
            P[j, j] = 0.55
        end

        # Test with rigorous=false (faster)
        result_fast = find_contraction_power(P, target_norm=0.5, max_power=50, rigorous=false)

        @test result_fast.N !== nothing
        @test result_fast.N > 0
        @test result_fast.α < 0.5
        @test length(result_fast.norms) == result_fast.N
        @test result_fast.rigorous == false

        # Test with rigorous=true (default)
        result_rig = find_contraction_power(P, target_norm=0.5, max_power=50, rigorous=true)

        @test result_rig.N !== nothing
        @test result_rig.α < 0.5
        @test result_rig.rigorous == true

        # Rigorous bound should be >= non-rigorous (it's an upper bound)
        @test result_rig.α >= result_fast.α - 1e-15  # Small tolerance for comparison
    end

    @testset "compute_stationary_density" begin
        # Simple reversible matrix
        P = [0.8 0.3;
             0.2 0.7]

        result = compute_stationary_density(P)

        @test result.converged
        @test sum(result.π) ≈ 1.0
        @test all(result.π .>= 0)

        # Check it's actually stationary: P*π = π, π = (0.6, 0.4) for this column-stochastic P
        @test P * result.π ≈ result.π atol=1e-10
        @test result.π ≈ [0.6, 0.4] atol=1e-10

        # Doubly stochastic matrix has uniform stationary
        P_doubly = [0.5 0.5;
                    0.5 0.5]
        result_doubly = compute_stationary_density(P_doubly)
        @test result_doubly.π ≈ [0.5, 0.5] atol=1e-10
    end

    @testset "noise_coordinate_variations and noise_var12 - 2D uniform" begin
        # square noise of width 0.1: var_i = 2/0.1 each, var₁₂ = 4/0.1
        noise = uniform_noise_box(0.05, 2)
        v = noise_coordinate_variations(noise)
        @test all(x -> x >= 2 / big(0.1) - 1e-12 && x ≈ 20, v)
        @test noise_var12(noise) >= v[1] && isapprox(noise_var12(noise), 40; atol = 1e-10)
        # widths 0.2 and 0.1: var₁ = 10, var₂ = 20, var₁₂ = 30
        noise_rect = anisotropic_noise_box([0.1, 0.05])
        vr = noise_coordinate_variations(noise_rect)
        @test isapprox(vr[1], 10; atol = 1e-10) && isapprox(vr[2], 20; atol = 1e-10)
        @test noise_var12(noise_rect) ≈ 30.0 atol = 1e-10
        # upward: the bound is at least 2 over the exact width of the box
        @test all(i -> vr[i] >= 2 / (big(sup(noise_rect[i])) - big(inf(noise_rect[i]))), 1:2)
    end

    @testset "ulam_approximation_error" begin
        # Build uniform partition
        domain = IntervalBox(interval(0.0, 1.0), interval(0.0, 1.0))
        partition = build_uniform_partition(domain, 3)  # depth 3 → boxes of size 1/8

        δ = ulam_approximation_error(partition)

        # Maximum diameter should be √2 * (1/8) ≈ 0.177 for 2D square boxes
        # Actually box_width returns the maximum side length, not diameter
        # For uniform partition depth 3: box side = 1/8 = 0.125
        @test δ ≈ 0.125 atol=1e-10
    end

    @testset "rigorous_error_bound - contracting map" begin
        # Simple contracting map: F(x,y) = (0.5x, 0.5y)
        F(x, y) = (0.5*x + 0.25, 0.5*y + 0.25)

        domain = IntervalBox(interval(0.0, 1.0), interval(0.0, 1.0))
        partition = build_uniform_partition(domain, 3)  # 64 boxes

        # Grid-aligned noise
        noise = uniform_noise_box(1/16, 2)  # width = 1/8

        # Build deterministic transition matrix
        P_det, _ = build_transition_matrix(F, partition)
        normalize_columns!(P_det)

        # Build noise operator
        N_op = build_grid_aligned_noise_operator(noise, partition, periodic=true)

        # Compose
        P_noisy = N_op * P_det
        normalize_columns!(P_noisy)

        # Compute error bound
        result = rigorous_error_bound(P_noisy, partition, noise,
                                      target_contraction=0.9,
                                      max_power=50,
                                      verbose=false)

        # Should find contraction (simple contracting map + noise)
        @test result.N !== nothing
        @test result.α !== nothing
        @test result.α < 0.9
        @test result.δ > 0
        @test result.var_ρ > 0

        # The bound should be finite if mixing is satisfied
        if result.mixing_satisfied
            @test isfinite(result.error_bound)
            @test result.error_bound > 0
        end
    end

    @testset "spectral_gap estimation" begin
        # Simple mixing matrix
        P = [0.7 0.4;
             0.3 0.6]

        gap_exact = spectral_gap(P, method=:eigen)
        gap_power = spectral_gap(P, method=:power_iteration)

        # Both methods should give similar results
        @test gap_exact > 0
        @test gap_power > 0
        @test abs(gap_exact - gap_power) < 0.1

        # For this matrix, eigenvalues are 1 and 0.3, so gap = 0.7
        @test gap_exact ≈ 0.7 atol=1e-10
    end

    @testset "integration with Ulam method" begin
        # Full workflow test
        F(x, y) = (0.5*x + 0.25, 0.5*y + 0.25)
        domain = IntervalBox(interval(0.0, 1.0), interval(0.0, 1.0))

        # Build adaptive partition
        noise = uniform_noise_box(1/32, 2)
        partition, dyn_graph, topo_graph = build_adaptive_partition(
            F, domain, noise,
            max_diameter=0.1,
            initial_depth=2,
            verbose=false
        )

        # Build operators
        P_det, _ = build_transition_matrix(F, partition,
            transition_graph=dyn_graph)
        normalize_columns!(P_det)

        N_op = build_grid_aligned_noise_operator(noise, partition,
            topological_graph=topo_graph, periodic=true)

        P_noisy = N_op * P_det
        normalize_columns!(P_noisy)

        # Compute stationary density
        stat_result = compute_stationary_density(P_noisy)
        @test stat_result.converged
        @test sum(stat_result.π) ≈ 1.0

        # Find contraction
        contract_result = find_contraction_power(P_noisy,
            target_norm=0.9, max_power=30)

        # For a contracting map with noise, should find contraction
        @test contract_result.N !== nothing
    end

    @testset "sparse matrix support" begin
        # Create sparse stochastic matrix
        n = 100
        I_idx = Int[]
        J_idx = Int[]
        vals = Float64[]

        for j in 1:n
            # Each column has a few non-zeros
            neighbors = [j, mod1(j+1, n), mod1(j-1, n)]
            for i in neighbors
                push!(I_idx, i)
                push!(J_idx, j)
                push!(vals, 1.0/3)
            end
        end

        P_sparse = sparse(I_idx, J_idx, vals, n, n)

        # Should work with sparse matrices
        norm_V = operator_norm_on_V(P_sparse)
        @test norm_V > 0
        @test norm_V <= 1.0

        # Stationary density
        result = compute_stationary_density(P_sparse)
        @test result.converged
        @test sum(result.π) ≈ 1.0
    end

end
