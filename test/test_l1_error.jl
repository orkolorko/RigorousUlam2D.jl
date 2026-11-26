using Test, RigorousUlam2D, IntervalArithmetic, IntervalBoxes, LinearAlgebra, SparseArrays, Random
using RigorousInvariantMeasures: RigorousInvariantMeasures
using BallArithmetic: mul_up, sqrt_up, div_up
const RIMl = RigorousInvariantMeasures

@testset "the L¹ error scheme" begin
    @testset "coordinate_variations against the edge sums" begin
        Random.seed!(2)
        k, h = 5, 0.25
        m = rand(k * k)
        M = reshape(m, k, k)                         # M[iy, ix]
        s1 = sum(abs(M[iy, ix] - M[iy, mod1(ix + 1, k)]) for ix in 1:k, iy in 1:k)   # edges normal to e₁
        s2 = sum(abs(M[iy, ix] - M[mod1(iy + 1, k), ix]) for ix in 1:k, iy in 1:k)   # edges normal to e₂
        v = coordinate_variations(m, k, h)
        @test v.var1 >= s1 / h && v.var2 >= s2 / h
        @test isapprox(v.var1, s1 / h; rtol = 1e-12) && isapprox(v.var2, s2 / h; rtol = 1e-12)
        # a function of x alone (constant along e₂) has var₂ = 0, and the reverse
        mx = vec(repeat(rand(1, k), k, 1))           # M[iy, ix] depends on ix only
        @test coordinate_variations(mx, k, h).var2 == 0 && coordinate_variations(mx, k, h).var1 > 0
        my = vec(repeat(rand(k, 1), 1, k))
        @test coordinate_variations(my, k, h).var1 == 0 && coordinate_variations(my, k, h).var2 > 0
        # a single bump of mass 1 in a cell of side h: two jumps of 1/h² across edges of length h each way
        e = zeros(k * k); e[7] = 1.0
        @test coordinate_variations(e, k, h).var1 >= 2 / h && coordinate_variations(e, k, h).var2 >= 2 / h
        @test coordinate_variations(ones(k * k), k, h) == (; var1 = 0.0, var2 = 0.0)
        @test_throws DimensionMismatch coordinate_variations(ones(k * k + 1), k, h)
    end

    @testset "the constants against BigFloat" begin
        setprecision(BigFloat, 256) do
            h, var12ρ = 2 / 64, 2 / 0.3
            @test defect_apriori(h, var12ρ) >= 1.5 * big(h) * big(var12ρ)
            @test defect_apriori(h, var12ρ) <= 1.5 * big(h) * big(var12ρ) * (1 + 1e-15)
        end
        @test fixed_point_distance(10.0, 1e-14, 1e-16) >= 10 * big(1e-14) / (1 - big(1e-16)) + big(1e-16)
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
        # the column sweep bounds the norms of the powers of the midpoint operator on V_h
        C = norms_of_powers_columns(info.Bmul!, n, 20; η1 = info.η1, eta1A = info.eta1A,
                                    normP = info.colL + info.colE + 1e-15, u = fp.v, block = 4)   # ≥ ‖L_lo + U‖₁
        for j in 1:20
            @test opnorm(Ad^j * Pv, 1) / 2 <= C[j]      # (e_i − e_j)/2 span the unit ball of V_h
        end
        @test norm(fp.v - fh, 1) <= fixed_point_distance(Σexact * (1 + 1e-12), res.ε1, res.ε2)
    end

    @testset "coarse_fine_norms bounds the norms of a finer operator (Lorenz, depths 6 and 7)" begin
        sys = lorenz_system(bisect = 12)
        mats = ulam_matrices(UlamTree(sys.F, sys.domain, 2, 7; enclosure_method = zeroth_order_enclosure,
                                      area = sys.area), 6:7; level = 9)
        a = 3 // 8                                   # ξ = 3/4: ξ k exact at both depths
        op(d) = (m = mats[d - 5]; perm = grid_order(sys.domain, d);
                 noisy_operator(m.L_lo[perm, perm], m.U[perm, perm], SquareNoise(sys.domain, d, Float64(a); slots = 2))[2])
        ic, if_ = op(6), op(7)
        nc = 4^6; M = 30
        u = ulam_fixed_point(ic.Amul, nc).v
        C = norms_of_powers_columns(ic.Bmul!, nc, M; η1 = ic.η1, eta1A = ic.eta1A, u, block = 64)
        h = 1 / 64; var12ρ = div_up(16.0, 3.0)                       # 2/a
        F = coarse_fine_norms(refine_norms_of_powers(C, M), h, var12ρ)
        @test length(F) == M + 1 && all(<=(1), F) && any(<(1), F)
        # a lower bound on ‖P_f^k|_V‖₁: half the largest distance between the images of 48
        # cells, as (e_i − e_j)/2 lies in the unit ball of V; the fine operator is the computed one
        nf = 4^7
        Random.seed!(5)
        cols = sort(rand(1:nf, 48))
        X = zeros(nf, length(cols)); for (c, i) in enumerate(cols); X[i, c] = 1.0; end
        Y = similar(X)
        for k in 1:M + 1
            if_.Bmul!(Y, X); X, Y = Y, X
            low = maximum(norm(X[:, p] - X[:, q], 1) / 2 for p in axes(X, 2) for q in p+1:size(X, 2))
            @test low <= F[k] + 1e-9
        end
    end

    @testset "swapped_error against BigFloat" begin
        Random.seed!(11)
        for _ in 1:200
            Σ, h, var12ρ = 1 + 50rand(), rand() / 100, 1 + 20rand()
            ε1, ε2 = rand() * 1e-10, rand() * 1e-10
            b = swapped_error(Σ, h, var12ρ, ε1, ε2)
            exact = setprecision(BigFloat, 256) do
                B = big
                B(Σ) * (B(h) * B(var12ρ) + B(ε1) / (1 - B(ε2))) + B(ε2)
            end
            @test b >= exact && b <= exact * (1 + 1e-14)
            @test b < mul_up(Σ, defect_apriori(h, var12ρ))   # below the direct route when the residuals are small
        end
        @test_throws ArgumentError swapped_error(1.0, 0.1, 1.0, 0.0, 1.0)
    end

    @testset "rounding: sigma_from_norms, coarse_fine_norms, coordinate_variations against BigFloat" begin
        Random.seed!(17)
        setprecision(BigFloat, 256) do
            for _ in 1:50
                M = 40
                C = sort(rand(M) .* 1.2; rev = true)          # bounds, some below 1
                any(<(1), C) || continue
                g = sigma_from_norms(C)
                exact = sum(min(1, big(C[i])) for i in 1:g.k-1; init = big(1)) / (1 - big(C[g.k]))
                @test g.grid >= exact && g.V >= 1 + exact
                # a bound given in BigFloat is rounded up, not to nearest
                Cb = big.(C) .+ big(2)^-80
                @test sigma_from_norms(Cb).grid >= sum(min(1, Cb[i]) for i in 1:sigma_from_norms(Cb).k-1; init = big(1)) /
                                                     (1 - Cb[sigma_from_norms(Cb).k])
                # the recursion of coarse_fine_norms, in BigFloat with the same constant
                h, var12ρ = 2.0^-9, 2 / (51 / 512)
                F = coarse_fine_norms(C, h, var12ρ)
                c = big(h) / 2 * big(var12ρ)
                Cx(k) = k == 0 ? big(1) : big(C[k])
                Fx = Vector{BigFloat}(undef, M + 1); Fx[1] = 1
                F0(k) = k == 0 ? big(1) : Fx[k]
                for n in 1:M
                    t = c * F0(n) + sum(Cx(i) * (c * F0(n - 1 - i) + c * F0(n - i)) for i in 0:n-1)
                    Fx[n+1] = min(Cx(n) * Fx[1] + t, Fx[n] * Fx[1])
                end
                for k in 2:M+1                                 # submultiplicative extension, exactly
                    Fx[k] = min(Fx[k], minimum(Fx[i] * Fx[k-i] for i in 1:k-1))
                end
                @test all(F .>= Fx)
            end
            # coordinate_variations: the edge sums |m_i − m_j|/h, upward
            k = 8; h = 2.0 / k
            m = rand(k * k) ./ 64
            Mx = reshape(big.(m), k, k)
            e1 = sum(abs(Mx[iy, ix] - Mx[iy, mod1(ix + 1, k)]) for ix in 1:k, iy in 1:k) / big(h)
            e2 = sum(abs(Mx[iy, ix] - Mx[mod1(iy + 1, k), ix]) for ix in 1:k, iy in 1:k) / big(h)
            cv = coordinate_variations(m, k, h)
            @test cv.var1 >= e1 && cv.var2 >= e2
        end
    end
end

@testset "x_marginal" begin
    k, h = 4, 0.25
    M = reshape(collect(1.0:16.0), k, k) ./ 136                 # M[iy, ix]
    xm = x_marginal(vec(M), k, h)
    @test length(xm) == k
    @test xm ≈ [sum(M[:, ix]) / h for ix in 1:k]
    @test sum(xm) * h ≈ 1                                      # a probability density in x
    @test_throws DimensionMismatch x_marginal(ones(15), k, h)
end
