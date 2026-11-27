using Test, RigorousUlam2D, IntervalArithmetic, IntervalBoxes, LinearAlgebra, SparseArrays, Random

# ‖P^k|_{V_h}‖₁ exactly: the extreme points of the unit ball of V_h in ℓ¹ are (eᵢ − eⱼ)/2
function exact_norm_on_V(P::Matrix{Float64}, k::Int)
    Q = Rational{BigInt}.(P)^k
    n = size(P, 1)
    return maximum(sum(abs, Q[:, i] - Q[:, j]) / 2 for i in 1:n for j in i+1:n)
end

@testset "norms of powers on V_h by columns" begin
    @testset "against the exact norm, random Markov matrices" begin
        Random.seed!(7)
        for n in (3, 6, 9)
            P = rand(n, n); P ./= sum(P; dims = 1)
            C = norms_of_powers_sparse(P, 6; block = 4)
            for k in 1:6
                @test C[k] >= exact_norm_on_V(P, k)
            end
            @test C[6] < 2 * exact_norm_on_V(P, 6) + 1e-12
        end
    end

    @testset "the pair that e₁ − eⱼ halved misses" begin
        # columns a = (b + c)/2, b, c: the norm on V_h is ‖b − c‖₁/2 = 1, while
        # ‖P(e₁ − eⱼ)/2‖₁ = 1/2 for j = 2, 3
        P = [0.5 1.0 0.0; 0.5 0.0 1.0; 0.0 0.0 0.0]
        @test exact_norm_on_V(P, 1) == 1
        @test norms_of_powers_sparse(P, 1; u = zeros(3))[1] >= 1
        @test norms_of_powers_sparse(P, 1; u = [0.5, 0.5, 0.0])[1] >= 1
    end

    @testset "sparse and blocks agree, against the exact norm" begin
        Random.seed!(3)
        n = 20
        P = sprand(n, n, 0.3) + sparse(I, n, n); P = P ./ sum(P; dims = 1)
        P = sparse(P)
        u = fill(1 / n, n)
        Cs = norms_of_powers_sparse(P, 5; u, block = 7)
        Cs1 = norms_of_powers_sparse(P, 5; u, block = 1)
        @test Cs == Cs1
        @test all(Cs .>= [exact_norm_on_V(Matrix(P), k) for k in 1:5])
    end

    @testset "the rounding term is carried" begin
        P = [0.9 0.2; 0.1 0.8]
        C0 = norms_of_powers_sparse(P, 3; rigorous = false, u = [2 / 3, 1 / 3])
        C1 = norms_of_powers_sparse(P, 3; rigorous = true, u = [2 / 3, 1 / 3])
        @test all(C1 .> C0)
        @test_throws DimensionMismatch norms_of_powers_columns((Y, X) -> mul!(Y, P, X), 2, 3; η1 = 0.0, u = [1.0])
    end

    @testset "refine_norms_of_powers rounds upward" begin
        a = 0.1
        r = refine_norms_of_powers([a, 1.0], 2)
        @test r[2] >= big(a) * big(a)
    end

    @testset "the noisy operator: blocks of any size give the same bounds" begin
        dom = IntervalBox(interval(-1.0, 1.0), interval(-1.0, 1.0))
        d = 3; k = 2^d; n = k * k
        Random.seed!(11)
        L = sprand(n, n, 0.08) + sparse(I, n, n); L = sparse(L ./ sum(L; dims = 1))
        U = sparse(abs.(sprand(n, n, 0.02)) .* 1e-9)
        N = SquareNoise(dom, d, 0.25; bc = :periodic, slots = Threads.nthreads())
        info = noisy_operator(L, U, N)
        u = fill(1 / n, n)
        Cb = norms_of_powers_columns(info.Bmul!, n, 4; η1 = info.η1, eta1A = info.eta1A, u, block = 9)
        C1 = norms_of_powers_columns(info.Bmul!, n, 4; η1 = info.η1, eta1A = info.eta1A, u, block = 1)
        @test Cb == C1
        # split into two column ranges, the entrywise maximum is the whole run
        Ca = norms_of_powers_columns(info.Bmul!, n, 4; η1 = info.η1, eta1A = info.eta1A, u, block = 9, cols = 1:23)
        Cc = norms_of_powers_columns(info.Bmul!, n, 4; η1 = info.η1, eta1A = info.eta1A, u, block = 9, cols = 24:n)
        @test max.(Ca, Cc) == Cb
    end
end
