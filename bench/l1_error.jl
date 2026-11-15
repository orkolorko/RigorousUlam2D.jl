# The table of the L¹ error bounds: one row per stationary_density record of this host (and
# per certified_gap record without one), read from ~/ResearchData/RigorousUlam2D_Data/data,
# written to data/l1_error.<host>.tsv and .jls.  Columns: map, depth, a, n, h, var(ρ_ξ), the
# a priori defect 2δ var(ρ_ξ), Σ from the gap (radius, k, value), Σ from the circle (ρ, R₁,
# value), the error ‖f_ξ − f_h‖₁ ≤ Σ_V · defect, and from the density record the residual ε₁,
# var(f̃), ‖f̃ − f_h‖₁ and ‖f_ξ − f̃‖₁.  Everything is read off the records; nothing is
# recomputed except the Σ of a certified_gap record that has no density record yet, by the same
# functions of l1_error.jl as bench/stationary_density.jl.
#
#   julia --project=. bench/l1_error.jl

using RigorousUlam2D, Printf, Dates
using BallArithmetic: div_up, mul_up, sqrt_up, add_up
import Serialization

const HOST = chomp(read(`hostname`, String))
const SCRIPT = splitext(basename(@__FILE__))[1]
const DATADIR = expanduser(get(ENV, "DATADIR", "~/ResearchData/RigorousUlam2D_Data/data"))
gitrev(dir) = try chomp(read(`git -C $dir rev-parse --short HEAD`, String)) catch; "?" end

files = sort(filter(f -> occursin(r"^(stationary_density|certified_gap)\..*\.jls$", f) && endswith(f, ".$HOST.jls"), readdir(DATADIR)))
dens = Dict{String,Any}(); gaps = Dict{String,Any}()
for f in files
    r = Serialization.deserialize(joinpath(DATADIR, f))
    key = "$(r.map)_d$(r.depth)_a$(numerator(r.a))-$(denominator(r.a))"
    (startswith(f, "stationary") ? dens : gaps)[key] = r
end

rows = NamedTuple[]
for key in sort(union(keys(dens), keys(gaps)))
    d = get(dens, key, nothing); g = get(gaps, key, nothing)
    r = something(d, g)
    side = r.map == "henon" ? 2.0 : 1.0; area = side^2
    h = side / 2^r.depth; δ = mul_up(sqrt_up(2.0), h)
    varρ = div_up(2.0 * denominator(r.a), Float64(numerator(r.a)))
    defect = defect_apriori(δ, varρ, varρ)
    sg = sc = nothing
    if d !== nothing && d.sigmas !== nothing
        sg = isempty(d.sigmas.gap) ? nothing : d.sigmas.gap[argmin([x.grid for x in d.sigmas.gap])]
        sc = isempty(d.sigmas.circles) ? nothing : d.sigmas.circles[argmin([x.grid for x in d.sigmas.circles])]
    elseif g !== nothing
        cand = [(; R = x.R, k = s.k, grid = s.grid, V = s.V) for x in g.radii if x.ok && isfinite(x.floor) && x.floor > 0 && x.R < 1
                for s in (sigma_from_gap(mul_up(mul_up(mul_up(sqrt_up(area), RigorousUlam2D.C_M_UP), varρ), div_up(1.0, x.floor)), x.R),)]
        sg = isempty(cand) ? nothing : cand[argmin([x.grid for x in cand])]
        candc = [(; ρ = c.ρ, R1 = c.R1, grid = sigma_from_circle(area, varρ, c.R1).grid) for c in g.circles if c.ok && c.proj_ok && c.proj_count == 1 && isfinite(c.R1)]
        sc = isempty(candc) ? nothing : candc[argmin([x.grid for x in candc])]
    end
    ΣV = minimum(filter(isfinite, [sg === nothing ? Inf : sg.V, sc === nothing ? Inf : sc.grid]); init = Inf)
    push!(rows, (; map = r.map, depth = r.depth, a = r.a, n = r.n, h, varρ, defect,
        gapR = sg === nothing ? NaN : sg.R, gapk = sg === nothing ? 0 : sg.k, Σgap = sg === nothing ? NaN : sg.V,
        ρ = sc === nothing ? NaN : sc.ρ, R1 = sc === nothing ? NaN : sc.R1, Σcircle = sc === nothing ? NaN : sc.grid,
        err_apriori = isfinite(ΣV) ? mul_up(ΣV, defect) : NaN,
        ε1 = d === nothing ? NaN : d.ε1, varf = d === nothing ? NaN : d.varf,
        dist = d === nothing || d.sigmas === nothing ? NaN : d.sigmas.dist,
        err = d === nothing || d.sigmas === nothing ? NaN : d.sigmas.err))
end

tsv = joinpath(DATADIR, "$SCRIPT.$HOST.tsv")
open(tsv, "w") do io
    println(io, "# $SCRIPT.jl of RigorousUlam2D.jl $(gitrev(dirname(dirname(pathof(RigorousUlam2D))))) on $HOST, $(Dates.now()); bounds are upper bounds, see src/l1_error.jl")
    println(io, join(["map", "depth", "a", "n", "h", "var_rho", "defect", "gap_r", "gap_k", "Sigma_gap", "circle_rho", "R1", "Sigma_circle", "err_apriori", "eps1", "var_f", "dist_fh", "err"], "\t"))
    for x in rows
        println(io, join([x.map, x.depth, "$(numerator(x.a))/$(denominator(x.a))", x.n, @sprintf("%.6g", x.h), @sprintf("%.6g", x.varρ), @sprintf("%.6g", x.defect),
            @sprintf("%.6g", x.gapR), x.gapk, @sprintf("%.6g", x.Σgap), @sprintf("%.3g", x.ρ), @sprintf("%.6g", x.R1), @sprintf("%.6g", x.Σcircle),
            @sprintf("%.6g", x.err_apriori), @sprintf("%.3e", x.ε1), @sprintf("%.6g", x.varf), @sprintf("%.3e", x.dist), @sprintf("%.6g", x.err)], "\t"))
    end
end
Serialization.serialize(joinpath(DATADIR, "$SCRIPT.$HOST.jls"), (; rows, host = HOST, script = SCRIPT, time = Dates.now()))
print(read(tsv, String))
