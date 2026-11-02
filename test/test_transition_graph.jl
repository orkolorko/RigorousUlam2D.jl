using IntervalArithmetic, IntervalBoxes
using RigorousUlam2D
using SparseArrays, LinearAlgebra
using Test

# The Ulam matrix of (x, y) ↦ (x + c, y) on [0,1]² with n×n cells, by hand in exact rationals:
# periodic, the image is taken modulo 1; reflecting, the part of the image in [1, 2] is
# reflected about 1 (for 0 < c < 1 the image lies in [0, 2]).
function shift_matrix(c::Rational, n, bc)
    L = zeros(Rational{BigInt}, n * n, n * n)
    for iy in 1:n, ix in 1:n
        j = (iy - 1) * n + ix
        a, b = (ix - 1) // n + c, ix // n + c
        for k in 0:n-1
            lo, hi = k // n, (k + 1) // n
            pieces = bc === :periodic ? [(lo + t, hi + t) for t in -2:2] :
                                        [(lo, hi), (2 - hi, 2 - lo)]
            for (p, q) in pieces
                L[(iy - 1) * n + k + 1, j] += max(0 // 1, min(b, q) - max(a, p)) * n
            end
        end
    end
    return L
end

# grid id (iy-1)n+ix → box index of partition
function boxperm(partition, n)
    perm = zeros(Int, n * n)
    d = partition.domain
    for (k, b) in enumerate(partition.boxes)
        ix = round(Int, (inf(b[1]) - inf(d[1])) / diam(d[1]) * n) + 1
        iy = round(Int, (inf(b[2]) - inf(d[2])) / diam(d[2]) * n) + 1
        perm[(iy - 1) * n + ix] = k
    end
    return perm
end

@testset "Transition graph and Ulam hierarchy" begin
    dom = IntervalBox(interval(0.0, 1.0), interval(0.0, 1.0))

    @testset "doubling in x, periodic: exact at every depth, no uncertainty" begin
        doubling_x = (x, y) -> (2x, y)
        h = ulam_hierarchy(doubling_x, dom, 1, 3; leaf_depth = 1)
        @test [r.depth for r in h] == [1, 2, 3]
        for r in h
            n = 2^r.depth
            perm = boxperm(r.partition, n)
            @test nnz(r.U) == 0
            Lt = zeros(n * n, n * n)
            for iy in 1:n, ix in 1:n
                j = (iy - 1) * n + ix
                Lt[(iy - 1) * n + mod1(2ix - 1, n), j] += 0.5
                Lt[(iy - 1) * n + mod1(2ix, n), j] += 0.5
            end
            @test Matrix(r.L_lo)[perm, perm] == Lt
            @test Matrix(r.graph)[perm, perm] == (Lt .> 0)
        end
        @test transition_graph(doubling_x, dom, 2) == h[2].graph
    end

    @testset "translation by 1/3: two-sided enclosure, periodic and reflecting" begin
        c = 1 / 3
        shift_third = (x, y) -> (x + c, y)
        for bc in (:periodic, :reflecting)
            h = ulam_hierarchy(shift_third, dom, 2, 3; leaf_depth = 6, bc)
            for r in h
                n = 2^r.depth
                perm = boxperm(r.partition, n)
                Lt = Float64.(shift_matrix(Rational{BigInt}(c), n, bc))
                Llo = Matrix(r.L_lo)[perm, perm]; U = Matrix(r.U)[perm, perm]
                @test all(Llo .<= Lt)
                @test all(Lt .<= Llo .+ U .+ 1e-15)
                # the uncertain leaves lie in a strip of one leaf width around the two preimages
                # of grid lines in each cell (three, where the fold adds one)
                @test maximum(sum(U; dims = 1)) <= 3 * 2.0^(-(3 + 6 - r.depth)) + 1e-12
                @test all(sum(Llo; dims = 1) .<= 1) && all(sum(Llo .+ U; dims = 1) .>= 1)
            end
        end
    end

    @testset "the tree read level by level: enclosures tighten, never fail" begin
        c = 1 / 3
        shift_third = (x, y) -> (x + c, y)
        tree = UlamTree(shift_third, dom, 2, 3)
        @test tree.level == 2
        @test_throws ArgumentError ulam_matrices(tree, 3)
        truth = Dict(d => Float64.(shift_matrix(Rational{BigInt}(c), 2^d, :periodic)) for d in 2:3)
        width = Dict(2 => Inf, 3 => Inf)
        for ℓ in 2:9
            ℓ > 2 && refine!(tree)
            @test tree.level == ℓ
            for d in 2:min(ℓ, 3)
                r = ulam_matrices(tree, d)
                perm = boxperm(r.partition, 2^d)
                Llo = Matrix(r.L_lo)[perm, perm]; U = Matrix(r.U)[perm, perm]
                @test all(Llo .<= truth[d]) && all(truth[d] .<= Llo .+ U .+ 1e-15)
                w = maximum(sum(U; dims = 1))
                @test w <= width[d]
                width[d] = w
            end
        end
        @test width[2] <= 3 * 2.0^-7 && width[3] <= 3 * 2.0^-6
        one_shot = ulam_hierarchy(shift_third, dom, 2, 3; leaf_depth = 6)
        @test one_shot[2].L_lo == ulam_matrices(tree, 3).L_lo && one_shot[2].U == ulam_matrices(tree, 3).U
    end

    @testset "reflecting: folded images, resolved without subdivision" begin
        shift_half = (x, y) -> (x + 0.5, y)     # cell [1/2,1] → [1,3/2] folds onto [1/2,1]
        r = ulam_hierarchy(shift_half, dom, 1, 1; leaf_depth = 0, bc = :reflecting)[1]
        perm = boxperm(r.partition, 2)
        @test nnz(r.U) == 0
        @test Matrix(r.L_lo)[perm, perm] == [0 0 0 0; 1 1 0 0; 0 0 0 0; 0 0 1 1]
        shift_quarter = (x, y) -> (x + 0.25, y) # cell [1/2,1] → [3/4,5/4] folds into itself
        r2 = ulam_hierarchy(shift_quarter, dom, 1, 1; leaf_depth = 1, bc = :reflecting)[1]
        perm2 = boxperm(r2.partition, 2)
        @test nnz(r2.U) == 0
        @test Matrix(r2.L_lo)[perm2, perm2] == [0.5 0 0 0; 0.5 1 0 0; 0 0 0.5 0; 0 0 0.5 1]
        flip_shift = (x, y) -> (1.5 - x, y)     # [0,1/2] → [1,3/2] folds onto [1/2,1]
        r3 = ulam_hierarchy(flip_shift, dom, 1, 1; leaf_depth = 0, bc = :reflecting)[1]
        perm3 = boxperm(r3.partition, 2)
        @test nnz(r3.U) == 0
        @test Matrix(r3.L_lo)[perm3, perm3] == [0 0 0 0; 1 1 0 0; 0 0 0 0; 0 0 1 1]
    end

    @testset "Hénon, periodic: consistency between depths" begin
        henon = (x, y) -> ((1 - 1.4 * (1.5x)^2 + 0.5y) / 1.5, (0.3 * 1.5x) / 0.5)
        domh = IntervalBox(interval(-1.0, 1.0), interval(-1.0, 1.0))
        h = ulam_hierarchy(henon, domh, 3, 4; leaf_depth = 3)
        r3, r4 = h
        for r in h
            cs = vec(sum(r.L_lo; dims = 1)); csu = vec(sum(r.L_lo .+ r.U; dims = 1))
            @test all(cs .<= 1) && all(csu .>= 1)
            @test r.graph == (r.L_lo .+ r.U .> 0)
        end
        # aggregate the depth-4 enclosure to depth 3: lower bounds can only improve, upper bounds too
        p3 = boxperm(r3.partition, 8); p4 = boxperm(r4.partition, 16)
        A3 = Matrix(r3.L_lo)[p3, p3]; B3 = Matrix(r3.L_lo .+ r3.U)[p3, p3]
        A4 = Matrix(r4.L_lo)[p4, p4]; B4 = Matrix(r4.L_lo .+ r4.U)[p4, p4]
        agg(M) = begin                            # (1/4) Σ_{children} M
            S = zeros(64, 64)
            for jy in 1:16, jx in 1:16, iy in 1:16, ix in 1:16
                S[((iy + 1) ÷ 2 - 1) * 8 + (ix + 1) ÷ 2, ((jy + 1) ÷ 2 - 1) * 8 + (jx + 1) ÷ 2] +=
                    M[(iy - 1) * 16 + ix, (jy - 1) * 16 + jx] / 4
            end
            S
        end
        @test all(A3 .>= agg(A4) .- 1e-15)
        @test all(B3 .<= agg(B4) .+ 1e-15)
        # the one-level graph contains the refined one
        @test all(transition_graph(henon, domh, 4)[r4.graph] .== true)
    end
end
