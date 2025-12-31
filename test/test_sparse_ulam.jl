using IntervalArithmetic, IntervalBoxes
using RigorousUlam2D
using SparseArrays
using Test

@testset "Sparse Ulam Methods" begin

    @testset "UlamPartition" begin
        boxes = [
            IntervalBox(interval(0.0, 0.5), interval(0.0, 0.5)),
            IntervalBox(interval(0.5, 1.0), interval(0.0, 0.5)),
            IntervalBox(interval(0.0, 0.5), interval(0.5, 1.0)),
            IntervalBox(interval(0.5, 1.0), interval(0.5, 1.0))
        ]
        domain = IntervalBox(interval(0.0, 1.0), interval(0.0, 1.0))

        partition = UlamPartition(boxes, domain)

        @test partition.n_boxes == 4
        @test partition.domain == domain
        @test length(partition.boxes) == 4
    end

    @testset "partition_index" begin
        boxes = [
            IntervalBox(interval(0.0, 0.5), interval(0.0, 1.0)),
            IntervalBox(interval(0.5, 1.0), interval(0.0, 1.0))
        ]
        domain = IntervalBox(interval(0.0, 1.0), interval(0.0, 1.0))
        partition = UlamPartition(boxes, domain)

        point1 = IntervalBox(interval(0.2, 0.3), interval(0.4, 0.6))
        point2 = IntervalBox(interval(0.7, 0.8), interval(0.4, 0.6))

        @test partition_index(partition, point1) == 1
        @test partition_index(partition, point2) == 2
    end

    @testset "build_transition_matrix identity" begin
        # Identity map on a simple partition
        F_id(x, y) = (x, y)

        boxes = quadrisect(IntervalBox(interval(0.0, 1.0), interval(0.0, 1.0)))
        domain = IntervalBox(interval(0.0, 1.0), interval(0.0, 1.0))
        partition = UlamPartition(boxes, domain)

        P, U = build_transition_matrix(F_id, partition, maxdepth=6, ε=1e-4)

        @test size(P) == (4, 4)
        @test issparse(P)

        # For identity map, diagonal should dominate
        @test all(P[i, i] > 0.5 for i in 1:4)

        # Check column sums (should be ≈ 1 for well-behaved partition)
        col_sums = [sum(P[:, j]) for j in 1:4]
        @test all(0.5 .<= col_sums .<= 1.5)  # Allow some uncertainty
    end

    @testset "normalize_columns!" begin
        # Create a simple sparse matrix
        I_idx = [1, 2, 1, 2]
        J_idx = [1, 1, 2, 2]
        vals = [0.3, 0.6, 0.5, 0.4]
        P = sparse(I_idx, J_idx, vals, 2, 2)

        col_sums = normalize_columns!(P)

        @test col_sums[1] ≈ 0.9
        @test col_sums[2] ≈ 0.9

        # After normalization, columns should sum to 1
        @test sum(P[:, 1]) ≈ 1.0
        @test sum(P[:, 2]) ≈ 1.0
    end

    @testset "estimate_sparsity" begin
        I_idx = [1, 2, 3]
        J_idx = [1, 2, 3]
        vals = [1.0, 1.0, 1.0]
        P = sparse(I_idx, J_idx, vals, 10, 10)

        stats = estimate_sparsity(P)

        @test stats.nnz == 3
        @test stats.total == 100
        @test stats.sparsity ≈ 0.97
        @test stats.avg_per_column ≈ 0.3
    end

    @testset "build_transition_matrix scaling" begin
        # Scaling map: x' = 0.5x, y' = 0.5y (contracting)
        F_contract(x, y) = (0.5 * x, 0.5 * y)

        # Create a finer partition
        domain = IntervalBox(interval(0.0, 1.0), interval(0.0, 1.0))
        boxes = quadrisect(domain)
        for _ in 1:1  # One more level of refinement
            new_boxes = eltype(boxes)[]
            for box in boxes
                append!(new_boxes, quadrisect(box))
            end
            boxes = new_boxes
        end

        partition = UlamPartition(boxes, domain)

        P, U = build_transition_matrix(F_contract, partition, maxdepth=5, ε=1e-3)

        @test size(P) == (16, 16)
        @test issparse(P)

        # Contracting map should have non-zero entries concentrated
        # in the lower-left region (maps to smaller values)
        stats = estimate_sparsity(P)
        @test stats.sparsity > 0.5  # Should be quite sparse
    end
end
