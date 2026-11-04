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

    @testset "the store: resumed from disk, and rows refined on demand" begin
        c = 1 / 3
        shift_third = (x, y) -> (x + c, y)
        store = mktempdir()
        tree = UlamTree(shift_third, dom, 2, 3; store)
        for _ in 1:3; refine!(tree); end
        @test tree.level == 5 && tree.maxlevel == 5
        tree2 = UlamTree(shift_third, dom, 2, 3; store)      # resumed: same level, same matrices
        @test tree2.level == 5 && tree2.nactive == tree.nactive && tree2.nresolved == tree.nresolved
        r1, r2 = ulam_matrices(tree, 2:3), ulam_matrices(tree2, 2:3)
        @test all(a.L_lo == b.L_lo && a.U == b.U for (a, b) in zip(r1, r2))
        @test r1[2].L_lo == ulam_matrices(tree, 3).L_lo
        @test_throws ArgumentError UlamTree(shift_third, dom, 2, 4; store)
        refine!(tree2); refine!(tree2)
        fresh = ulam_hierarchy(shift_third, dom, 2, 3; leaf_depth = 4)
        rr = ulam_matrices(tree2, 2:3)
        @test all(a.L_lo == b.L_lo && a.U == b.U for (a, b) in zip(fresh, rr))
        # refine only the cells feeding the rows of the lower half of the domain at depth 3:
        # those rows halve, the others do not move, and the enclosure still holds
        truth = Float64.(shift_matrix(Rational{BigInt}(c), 8, :periodic))
        rows0 = vec(sum(Matrix(rr[2].U); dims = 2))
        heavy = [sup(b[2]) <= 0.5 for b in rr[2].partition.boxes]
        @test 0 < count(heavy) < length(heavy)
        refine!(tree2; targets = heavy)
        @test tree2.level == 7 && tree2.maxlevel == 8
        r = ulam_matrices(tree2, 3)
        perm = boxperm(r.partition, 8)
        rows1 = vec(sum(Matrix(r.U); dims = 2))
        @test all(rows1[heavy] .<= rows0[heavy] / 2 .+ 1e-15)
        @test rows1[.!heavy] == rows0[.!heavy]
        Llo = Matrix(r.L_lo)[perm, perm]; U = Matrix(r.U)[perm, perm]
        @test all(Llo .<= truth) && all(truth .<= Llo .+ U .+ 1e-15)
    end

    @testset "the continuation in memory: the matrices of a deeper store, nothing written" begin
        c = 1 / 3
        shift_third = (x, y) -> (x + c, y)
        tree = UlamTree(shift_third, dom, 2, 3)
        @test_throws ArgumentError ulam_matrices(tree, 3)
        r = ulam_matrices(tree, 2:3; level = 3)             # the level-3 matrices from the level-2 store
        one_shot = ulam_hierarchy(shift_third, dom, 2, 3; leaf_depth = 0)
        @test all(a.L_lo == b.L_lo && a.U == b.U for (a, b) in zip(r, one_shot))
        @test tree.level == 2 && tree.nactive == 16
        for _ in 1:3; refine!(tree); end
        r = ulam_matrices(tree, 2:3; level = 9)
        @test tree.level == 5 && tree.maxlevel == 5
        deep = ulam_hierarchy(shift_third, dom, 2, 3; leaf_depth = 6)
        @test all(a.L_lo == b.L_lo && a.U == b.U for (a, b) in zip(r, deep))
        @test ulam_matrices(tree, 3; level = 4).U == ulam_matrices(tree, 3).U   # level ≤ tree.level: as read
        # targeted: the same rows as a store refined twice with the same targets
        heavy = [sup(b[2]) <= 0.5 for b in r[2].partition.boxes]
        store = mktempdir()
        tree2 = UlamTree(shift_third, dom, 2, 3; store)
        for _ in 1:3; refine!(tree2); end
        refine!(tree2; targets = heavy); refine!(tree2; targets = heavy)
        @test tree2.level == 5 && tree2.maxlevel == 7
        rt = ulam_matrices(tree, 2:3; level = ifelse.(heavy, 7, 5))
        rs = ulam_matrices(tree2, 2:3)
        @test all(a.L_lo == b.L_lo && a.U == b.U for (a, b) in zip(rt, rs))
        @test ulam_matrices(tree2, 3; level = 9).U == ulam_matrices(tree, 3; level = 9).U
        # raising the levels of the heavy rows only: the others keep their width
        rows7 = vec(sum(rt[2].U; dims = 2))
        r8 = ulam_matrices(tree, 3; level = ifelse.(heavy, 8, 5))
        rows8 = vec(sum(r8.U; dims = 2))
        @test all(rows8[heavy] .<= rows7[heavy] / 2 .+ 1e-15) && rows8[.!heavy] == rows7[.!heavy]
        @test_throws ArgumentError ulam_matrices(tree, 3; level = [5, 6])
    end

    @testset "fibred_area: the inverse of the fibre integrated in closed form" begin
        # (x, y) ↦ (x, x + y): the fraction of [0,1]² with x + y ∈ [1/2, 3/2] is 3/4
        area = fibred_area(1, x -> x, (x, w) -> w - x; Yof = (x, w) -> w * x - x^2 / 2)
        box = IntervalBox(interval(0.0, 1.0), interval(0.0, 1.0))
        A = area(box, IntervalBox(interval(0.0, 1.0), interval(0.5, 1.5)))
        @test inf(A) <= 0.75 <= sup(A) && diam(A) <= 1e-12   # the clamps change side at a bisection point
        @test sup(area(box, IntervalBox(interval(2.0, 3.0), interval(0.5, 1.5)))) == 0
        A2 = area(box, IntervalBox(interval(0.25, 0.5), interval(0.0, 2.0)))   # 1/4, by bisection in x
        @test inf(A2) <= 0.25 <= sup(A2) && diam(A2) <= 2.0^-10
        riemann = fibred_area(1, x -> x, (x, w) -> w - x; bisect = 6)         # no antiderivative
        A3 = riemann(box, IntervalBox(interval(0.0, 1.0), interval(0.5, 1.5)))
        @test inf(A3) <= 0.75 <= sup(A3) && diam(A3) <= 2.0^-5
        @test_throws ArgumentError fibred_area(3, x -> x, (x, w) -> w)
    end

    @testset "areas at the leaves: the translation exact, Hénon tightened and consistent" begin
        c = 1 / 3
        shift_third = (x, y) -> (x + c, y)
        area = fibred_area(1, x -> x + c, (x, w) -> w; Yof = (x, w) -> w * x, bisect = 30)
        for bc in (:periodic, :reflecting)
            r = ulam_hierarchy(shift_third, dom, 2, 3; leaf_depth = 0, bc, area)[2]   # no leaf refined
            n = 8
            perm = boxperm(r.partition, n)
            Lt = Float64.(shift_matrix(Rational{BigInt}(c), n, bc))
            Llo = Matrix(r.L_lo)[perm, perm]; U = Matrix(r.U)[perm, perm]
            @test all(Llo .<= Lt) && all(Lt .<= Llo .+ U)
            @test maximum(U) <= 1e-8                      # the ends x = g - 1/3 are bisected 30 times
            @test all(sum(Llo; dims = 1) .<= 1) && all(sum(Llo .+ U; dims = 1) .>= 1)
        end
        # Hénon: with areas the enclosure lies inside the one without, column sums still bracket 1,
        # and the depth-3 enclosure is consistent with the aggregated depth-4 one
        henon = (x, y) -> ((1 - 1.4 * (1.5x)^2 + 0.5y) / 1.5, (0.3 * 1.5x) / 0.5)
        harea = fibred_area(2, x -> (0.3 * 1.5x) / 0.5, (x, w) -> 3w - 2 + 2 * 1.4 * (1.5x)^2;
                            Yof = (x, w) -> (3w - 2) * x + 2 * 1.4 * 1.5^2 * x^3 / 3)
        domh = IntervalBox(interval(-1.0, 1.0), interval(-1.0, 1.0))
        h0 = ulam_hierarchy(henon, domh, 3, 4; leaf_depth = 3)
        ha = ulam_hierarchy(henon, domh, 3, 4; leaf_depth = 3, area = harea)
        for (r0, r) in zip(h0, ha)
            @test all(r.L_lo .>= r0.L_lo .- 1e-15) && all(r.L_lo .+ r.U .<= r0.L_lo .+ r0.U .+ 1e-15)
            @test maximum(sum(r.U; dims = 1)) <= maximum(sum(r0.U; dims = 1)) / 50
            @test all(sum(r.L_lo; dims = 1) .<= 1) && all(sum(r.L_lo .+ r.U; dims = 1) .>= 1)
            @test r.graph == (r.L_lo .+ r.U .> 0)
        end
        p3 = boxperm(ha[1].partition, 8); p4 = boxperm(ha[2].partition, 16)
        A3 = Matrix(ha[1].L_lo)[p3, p3]; B3 = Matrix(ha[1].L_lo .+ ha[1].U)[p3, p3]
        A4 = Matrix(ha[2].L_lo)[p4, p4]; B4 = Matrix(ha[2].L_lo .+ ha[2].U)[p4, p4]
        agg(M) = begin
            S = zeros(64, 64)
            for jy in 1:16, jx in 1:16, iy in 1:16, ix in 1:16
                S[((iy + 1) ÷ 2 - 1) * 8 + (ix + 1) ÷ 2, ((jy + 1) ÷ 2 - 1) * 8 + (jx + 1) ÷ 2] +=
                    M[(iy - 1) * 16 + ix, (jy - 1) * 16 + jx] / 4
            end
            S
        end
        @test all(A3 .>= agg(A4) .- 1e-12) && all(B3 .<= agg(B4) .+ 1e-12)
        # the same areas through the in-memory descent of a stored tree
        tree = UlamTree(henon, domh, 3, 4; area = harea)
        rt = ulam_matrices(tree, 3:4; level = 7)
        @test all(all(abs.(a.L_lo .- b.L_lo) .<= 4e-15) && all(abs.(a.U .- b.U) .<= 4e-15) for (a, b) in zip(rt, ha))
    end
end
