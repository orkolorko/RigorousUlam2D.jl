using IntervalArithmetic
using RigorousUlam2D
using Test

@testset "zeroth order enclosure" begin
    F(x, y) = x^2 + y
    box = IntervalBox(Interval(0.0, 1.0), Interval(-1.0, 2.0))
    @test zeroth_order_enclosure(F, box) == F(box...)
end

@testset "first order enclosure" begin
    F(x, y) = x^2 + y^2
    box = IntervalBox(Interval(-0.2, 0.4), Interval(1.0, 1.5))
    first_order = first_order_enclosure(F, box)
    zeroth_order = zeroth_order_enclosure(F, box)
    @test subset(zeroth_order, first_order)

    G(x, y) = (x + y, x - y)
    zeroth_tuple = zeroth_order_enclosure(G, box)
    first_tuple = first_order_enclosure(G, box)
    @test all(subset(zeroth_tuple[i], first_tuple[i]) for i in eachindex(zeroth_tuple))
end
