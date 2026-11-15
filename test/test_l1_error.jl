using Test, RigorousUlam2D, IntervalArithmetic, IntervalBoxes, LinearAlgebra, SparseArrays, Random
using RigorousInvariantMeasures: RigorousInvariantMeasures
const RIMl = RigorousInvariantMeasures

@testset "the L¹ error scheme" begin
    @testset "grid_variation against the edge sum" begin
        Random.seed!(2)
        k, h = 5, 0.25
        m = rand(k * k)
        M = reshape(m, k, k)
        s = 0.0
        for ix in 1:k, iy in 1:k
            s += abs(M[iy, ix] - M[mod1(iy + 1, k), ix]) + abs(M[iy, ix] - M[iy, mod1(ix + 1, k)])
        end
        v = grid_variation(m, k, h)
        @test v >= s / h
        @test v ≈ s / h rtol = 1e-12
        # a single bump of mass 1 in a cell of side h: four jumps of height 1/h², edges of length h
        e = zeros(k * k); e[7] = 1.0
        @test grid_variation(e, k, h) >= 4 / h
        @test grid_variation(e, k, h) ≈ 4 / h rtol = 1e-15
        @test grid_variation(ones(k * k), k, h) == 0.0
        @test_throws DimensionMismatch grid_variation(ones(k * k + 1), k, h)
    end

    @testset "the constants against BigFloat" begin
        setprecision(BigFloat, 256) do
            CM = 1 / (2 * sqrt(big(2)))
            @test RigorousUlam2D.C_M_UP >= CM && RigorousUlam2D.C_M_UP - CM < 1e-16
            δ, varρ, varf = sqrt(2) / 64, 2 / 0.3, 3.7
            @test defect_apriori(δ, varρ, varf) >= 1.5 * big(δ) * varρ + 0.5 * big(δ) * varf
            s = sigma_from_circle(4.0, varρ, 9.2)
            @test s.grid == s.V && s.V >= 1 + 2 * CM * big(varρ) * 9.2
            # sigma_from_gap: the block sum at its k, with C_0 = C_1 = 1 and C_i ≤ min(1, K rⁱ)
            K, r = 2330.0, 0.6342
            g = sigma_from_gap(K, r)
            α = big(K) * big(r)^g.k
            S = (2 + sum(min(1, big(K) * big(r)^i) for i in 2:g.k-1)) / (1 - α)
            @test α < 1 && g.grid >= S && g.grid - S < 1e-10 * S && g.V >= 1 + S
            @test g.α >= α
            # no k does better by more than rounding
            for k in 2:400
                αk = big(K) * big(r)^k
                αk < 1 || continue
                Sk = (2 + sum(min(1, big(K) * big(r)^i) for i in 2:k-1; init = big(0))) / (1 - αk)
                @test g.grid <= Sk * (1 + 1e-12)
            end
            @test_throws ArgumentError sigma_from_gap(K, 1.0)
        end
        @test fixed_point_distance(10.0, 1e-14, 1e-16) >= 10 * 1e-14 + 1e-16
        @test_throws ArgumentError fixed_point_distance(10.0, 1e-14, 1.0)
    end

    @testset "the computed density of the depth-3 Hénon operator" begin
        dom = IntervalBox(interval(-1.0, 1.0), interval(-1.0, 1.0))
        d = 3; k = 2^d; n = k * k; h = 2.0 / k
        henon = (x, y) -> ((1 - 1.4 * (1.5x)^2 + 0.5y) / 1.5, (0.3 * 1.5x) / 0.5)
        hier = ulam_hierarchy(henon, dom, d, d; leaf_depth = 4, bc = :periodic,
            enclosure_method = zeroth_order_enclosure)[1]
        perm = grid_order(dom, d)
        L_lo, U = hier.L_lo[perm, perm], hier.U[perm, perm]
        N = SquareNoise(dom, d, 0.5; slots = 2)
        op, info = noisy_operator(L_lo, U, N)
        @test info.colL >= opnorm(Matrix(info.Lhat), 1) && info.colE >= opnorm(Matrix(L_lo + U - info.Lhat), 1)
        @test info.η1 >= info.colE && info.eta1A >= info.etaA
        # the ℓ¹ rounding model against the product in 256-bit arithmetic
        B = RIMl.Ulam(k)
        Kb = Matrix{BigFloat}(undef, k, k)
        for j in 1:k
            e = zeros(BigFloat, k); e[j] = 1
            RIMl.mul!(RIMl.UlamProjectedUniformKernel(Val(:periodic), BigFloat, B, 0.5), e)
            Kb[:, j] = e
        end
        Nb = kron(Kb, Kb); Lb = Matrix{BigFloat}(info.Lhat)
        Random.seed!(3)
        for _ in 1:5
            x = rand(n)
            Ax = info.Amul(x)
            @test norm(Float64.(BigFloat.(Ax) - Nb * (Lb * BigFloat.(x))), 1) <= (info.η1 - info.colE) * norm(x, 1) + info.eta1A
        end
        # the fixed point, its residual and its distance from the exact one
        fp = ulam_fixed_point(info.Amul, n; tol = 1e-14)
        @test fp.converged && all(>=(0), fp.v) && sum(fp.v) ≈ 1
        res = fixed_point_residual(info.Amul, fp.v, info.η1, info.eta1A)
        @test res.ε1 >= norm(info.Amul(fp.v) - fp.v, 1) + info.colE * norm(fp.v, 1)
        # the rest is the computed residual (the columns of L̂ = L_lo + U/2 do not sum to 1 when the enclosure is loose, as here) and rounding
        @test res.ε1 - info.colE * norm(fp.v, 1) <= norm(info.Amul(fp.v) - fp.v, 1) + 1e-12
        @test res.ε2 >= abs(sum(fp.v) - 1) && res.ε2 < 1e-14
        # the exact fixed point of the midpoint operator, and the resolvent on V_h, densely
        Ad = hcat([info.Amul(Matrix{Float64}(I, n, n)[:, j]) for j in 1:n]...)
        F = eigen(Ad)
        i1 = argmin(abs.(F.values .- 1))
        fh = real(F.vectors[:, i1]); fh ./= sum(fh)
        Pv = I - ones(n) * ones(n)' / n              # onto zero-sum vectors
        Rinv = pinv(Matrix(I - Ad)) * Pv
        Σexact = opnorm(Rinv, 1)
        # the gap route with K, r from the dense data: r just above |λ₂|, R_Γ the resolvent on |z| = r
        λs = sort(F.values; by = abs, rev = true)
        r = 1.02 * abs(λs[2])
        RΓ = maximum(opnorm(inv(Ad - r * cis(θ) * I)) for θ in range(0, 2π; length = 721))
        K = 2 * RigorousUlam2D.C_M_UP * (2 / 0.5) * RΓ
        g = sigma_from_gap(K, r)
        @test g.grid >= Σexact
        # the circle route: the resolvent on |z − 1| = 0.25
        R1 = maximum(opnorm(inv(Ad - (1 + 0.25 * cis(θ)) * I)) for θ in range(0, 2π; length = 721))
        c = sigma_from_circle(4.0, 2 / 0.5, R1)
        @test c.grid >= Σexact
        @test norm(fp.v - fh, 1) <= fixed_point_distance(g.grid, res.ε1, res.ε2)
    end
end
