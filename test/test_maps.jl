using IntervalArithmetic, IntervalBoxes
using RigorousUlam2D
using SparseArrays
using Test

@testset "maps of the examples" begin
    # entries of two enclosures L_lo ≤ L ≤ L_lo + U of the same matrix must meet
    meet(a, b) = all(a.L_lo .<= b.L_lo .+ b.U) && all(b.L_lo .<= a.L_lo .+ a.U)

    @testset "$(name): the map at a point" for (name, sys, x, y, ref) in (
            ("henon", henon_system(), 0.1, 0.2, ((1 - 1.4 * 0.15^2 + 0.1) / 1.5, 0.09)),
            ("lorenz +", lorenz_system(), 0.09, 0.2, (0.5 - 1.3 * sqrt(0.09), 0.2 * 0.09^3 - 0.25)),
            ("lorenz −", lorenz_system(), -0.09, 0.2, (1.3 * sqrt(0.09) - 0.5, -0.2 * 0.09^3 + 0.25)),
            ("contracting +", contracting_lorenz_system(), 0.3, 0.2, (0.5 - 4 * 0.09, 0.2 * 0.027 - 0.25)),
            ("contracting −", contracting_lorenz_system(ρ = 18 // 5), -0.3, 0.2, (3.6 * 0.09 - 0.5, -0.2 * 0.027 + 0.25)),
            ("metzger +", metzger_lorenz_system(), 0.3, 0.2, (3.6 * 0.09 - 0.5, 0.2 * 0.3^6 - 0.25)),
            ("metzger −", metzger_lorenz_system(), -0.3, 0.2, (0.5 - 3.6 * 0.09, -0.2 * 0.3^6 + 0.25)))
        P = sys.F(interval(x), interval(y))
        @test all(abs(mid(P[i]) - ref[i]) <= 1e-14 && diam(P[i]) <= 1e-14 for i in 1:2)
    end

    @testset "$(sys.name): column sums bracket 1, routes agree" for sys in (henon_system(bisect = 16),
                                                                          lorenz_system(bisect = 16),
                                                                          contracting_lorenz_system(bisect = 16),
                                                                          contracting_lorenz_system(ρ = 18 // 5, bisect = 16),
                                                                          metzger_lorenz_system(bisect = 16))
        plain = ulam_system(sys.name; bisect = 16, xinv = false)
        mats(area) = ulam_matrices(UlamTree(sys.F, sys.domain, 2, 5; enclosure_method = zeroth_order_enclosure,
                                            area), 2:4; level = 7)
        cut, bis = mats(sys.area), mats(plain.area)
        none = ulam_matrices(UlamTree(sys.F, sys.domain, 2, 5; enclosure_method = zeroth_order_enclosure), 2:4; level = 7)
        for (c, b, z) in zip(cut, bis, none)
            lo, hi = vec(sum(c.L_lo; dims = 1)), vec(sum(c.L_lo .+ c.U; dims = 1))
            @test all(lo .<= 1 + 1e-12) && all(hi .>= 1 - 1e-12)
            @test meet(c, b) && meet(c, z)
            @test sum(c.U) <= sum(z.U)              # the areas tighten the counting enclosure
        end
    end
    @test ulam_system("clorenz_rho18-5").name == "clorenz_rho18-5"
    @test ulam_system("mlorenz_rho18-5").name == "mlorenz_rho18-5"
    # the integral of the fibre inverse against a fine midpoint sum, on one piece of each sign
    sys = metzger_lorenz_system()
    for (x0, w) in ((0.11, 0.1), (-0.13, -0.2))
        δ = 1e-3; n = 20_000
        num = sum(mid(sys.area.yof(interval(x0 + (i - 0.5) * δ / n), interval(w))) for i in 1:n) * δ / n
        Y = sys.area.yint(interval(x0), interval(δ), interval(w))
        @test abs(mid(Y) - num) <= 1e-6 * abs(num) && diam(Y) <= 1e-9 * abs(num)
    end
    # ℓ increases on each branch
    @test inf(metzger_lorenz_system().F(interval(0.2), interval(0.0))[1]) > sup(metzger_lorenz_system().F(interval(0.1), interval(0.0))[1])
    @test_throws ArgumentError contracting_lorenz_system(ρ = 5 // 1)
end
