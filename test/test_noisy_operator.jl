using Test, RigorousUlam2D, IntervalArithmetic, IntervalBoxes, LinearAlgebra, SparseArrays, Random
using RigorousInvariantMeasures: RigorousInvariantMeasures
const RIMt = RigorousInvariantMeasures

# the dense matrix of a one-dimensional kernel, by columns
function dense1d(K, k, T = Float64)
    M = zeros(T, k, k)
    for j in 1:k
        e = zeros(T, k); e[j] = 1
        RIMt.mul!(K, e)
        M[:, j] = e
    end
    return M
end
dense2d(N, n) = hcat([N * Matrix{Float64}(I, n, n)[:, j] for j in 1:n]...)

@testset "the noisy operator on the square" begin
    dom = IntervalBox(interval(-1.0, 1.0), interval(-1.0, 1.0))
    d = 3; k = 2^d; n = k * k
    B = RIMt.Ulam(k)

    @testset "SquareNoise is the product of the two kernels, $bc" for bc in (:periodic, :reflecting)
        for a in (0.5, 0.3125)                      # s = ξ k = 4 and 2.5 cells
            N = SquareNoise(dom, d, a; bc, slots = 2)
            ξ = a                                   # 2a of a side of length 2
            @test N.ξ == (ξ, ξ)
            K = RIMt.UlamProjectedUniformKernel(Val(bc), Float64, B, ξ)
            Kd = dense1d(K, k)
            Nd = dense2d(N, n)
            @test Nd ≈ kron(Kd, Kd) atol = 1e-15   # v[(ix-1)k + iy]: Kx on the index ix, Ky on iy
            @test Nd ≈ Nd' atol = 1e-15
            @test all(>=(0), Nd)
            @test all(x -> abs(x - 1) < 1e-14, sum(Nd; dims = 1))
            @test all(x -> abs(x - 1) < 1e-14, sum(Nd; dims = 2))
        end
        Na = SquareNoise(dom, d, (0.5, 0.3125); bc, slots = 1)
        Kx = dense1d(RIMt.UlamProjectedUniformKernel(Val(bc), Float64, B, 0.5), k)
        Ky = dense1d(RIMt.UlamProjectedUniformKernel(Val(bc), Float64, B, 0.3125), k)
        @test dense2d(Na, n) ≈ kron(Kx, Ky) atol = 1e-15
        # the pool: as many applications at once as slots, every one correct
        Np = SquareNoise(dom, d, 0.5; bc, slots = 3)
        vs = [rand(n) for _ in 1:12]
        outs = Vector{Vector{Float64}}(undef, 12)
        Threads.@threads for i in 1:12
            outs[i] = Np * vs[i]
        end
        @test all(outs[i] ≈ kron(Kx, Kx) * vs[i] for i in 1:12)
        @test_throws DimensionMismatch Np * rand(n + 1)
    end

    @testset "grid_order" begin
        perm = grid_order(dom, d)
        @test sort(perm) == 1:n
        P = build_uniform_partition(dom, d)
        for ix in 1:k, iy in 1:k
            box = P.boxes[perm[(ix - 1) * k + iy]]
            @test inf(box[1]) == -1 + (ix - 1) * 2 / k && inf(box[2]) == -1 + (iy - 1) * 2 / k
        end
    end

    @testset "noisy_operator and its ℓ¹ constants" begin
        henon = (x, y) -> ((1 - 1.4 * (1.5x)^2 + 0.5y) / 1.5, (0.3 * 1.5x) / 0.5)
        h = ulam_hierarchy(henon, dom, d, d; leaf_depth = 4, bc = :periodic,
            enclosure_method = zeroth_order_enclosure)[1]
        perm = grid_order(dom, d)
        L_lo, U = h.L_lo[perm, perm], h.U[perm, perm]
        @test any(>(0), U)                          # the enclosure is not trivial at this depth
        N = SquareNoise(dom, d, 0.5; slots = 2)
        info = noisy_operator(L_lo, U, N)
        @test info.Lhat ≈ L_lo + U / 2
        # colE bounds ‖L̂ − L‖₁ at the two ends of the enclosure, colL bounds ‖L̂‖₁
        @test opnorm(Matrix(info.Lhat - L_lo), 1) <= info.colE
        @test opnorm(Matrix(L_lo + U - info.Lhat), 1) <= info.colE
        @test opnorm(Matrix(info.Lhat), 1) <= info.colL
        @test info.w == max(maximum(diff(info.Lhat.colptr)), maximum(count(!iszero, Matrix(info.Lhat); dims = 2)))
        # the products, against the dense operator, and Bmul! column by column
        Nd = dense2d(N, n)
        Ad = Nd * Matrix(info.Lhat)
        Random.seed!(1)
        X = randn(n, 5); Y = zeros(n, 5)
        info.Bmul!(Y, X)
        for c in 1:5
            @test info.Amul(X[:, c]) ≈ Ad * X[:, c]
            @test Y[:, c] == info.Amul(X[:, c])
        end
        # the rounding part of η₁, η₁ − ‖E‖₁ with L = L̂: against the product in 256-bit arithmetic
        K = RIMt.UlamProjectedUniformKernel(Val(:periodic), BigFloat, B, 0.5)
        Kb = dense1d(K, k, BigFloat)
        Nb = kron(Kb, Kb)
        Lb = Matrix{BigFloat}(info.Lhat)
        for _ in 1:5
            x = randn(n)
            exact = Nb * (Lb * BigFloat.(x))
            @test Float64(sum(abs, BigFloat.(info.Amul(x)) - exact), RoundUp) <=
                  (info.η1 - info.colE) * sum(abs, x) + info.eta1A
        end
    end
end
