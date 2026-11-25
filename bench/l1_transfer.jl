# The L¹ error of the stationary density on finer grids, from the column sweep of a coarse one.
#
# For one (MAP, A) and a coarse depth DEPTH, the refined bounds C_k ≥ ‖P_h^k|_{V_h}‖₁ of the
# l1_mixing record go through `coarse_fine_norms` (the recursion of math/local_ly_2d.tex, after
# Galatolo, Monge, Nisoli 2020, Lemma 29, with the two square-cell constants (h/2) var₁₂(ρ_ξ)) to
# F_k ≥ ‖Λ^k|_V‖₁ for Λ the noisy Ulam operator P_{h'} of any finer dyadic grid, or L_ξ;
# `sigma_from_norms(F).grid` bounds Σ_{j ≥ 0} ‖Λ^j|_V‖₁ ≥ ‖(I − Λ)⁻¹|_V‖₁, the F_k bounding Λ on all
# of V.  For each finer depth in FINE with a row in the stationary_density record (its ε₁, ε₂, the
# residuals of the computed density f̃; the defect is recomputed here, `defect_apriori(h', var₁₂(ρ_ξ))`):
#     ‖f_ξ − f_{h'}‖₁ ≤ Σ · (3h'/2) var₁₂(ρ_ξ),   ‖f̃ − f_{h'}‖₁ ≤ fixed_point_distance(Σ, ε₁, ε₂),
# and their sum bounds ‖f_ξ − f̃‖₁; and `swapped_error`, the same Σ as a bound on the resolvent of
# L_ξ against the defect of L_ξ on f̃, constant on the cells: h' var₁₂(ρ_ξ) + ε₁/(1 − ε₂).  Every
# operation rounds upward; the inputs are read back exactly from the latest records of this host.
#
# Env: MAP, DEPTH (8), A (rational such as 77//256), FINE (comma-separated depths, "10,12").
# Record (CertifiedRecords): ~/ResearchData/RigorousUlam2D_Data/data/l1_transfer.<commit>.<host>.tsv,
# one row per (map, coarse depth, a, fine depth); a row already there is not computed again.
# Every row names, in its column `environment`, the copy of the Manifest of the run written
# beside the record by `save_environment`, with the commits of the dependencies given by path.
#
#   julia --project=. bench/l1_transfer.jl

using RigorousUlam2D, Printf
using BallArithmetic: add_up, mul_up
using CertifiedRecords

const SCRIPT = splitext(basename(@__FILE__))[1]
const DATADIR = expanduser(get(ENV, "DATADIR", "~/ResearchData/RigorousUlam2D_Data/data"))
const MAP = get(ENV, "MAP", "lorenz")
const DEPTH = parse(Int, get(ENV, "DEPTH", "8"))
const A = let s = get(ENV, "A", "77//256"); p, q = parse.(Int, split(s, "//")); p // q end
const FINE = parse.(Int, split(get(ENV, "FINE", "10,12"), ","))

rd(s) = Float64(read_endpoint(s, 53), RoundUp)        # exact for a double written by write_endpoint; upward otherwise
record = record_file(DATADIR, SCRIPT, RigorousUlam2D)
done = recorded_keys(record, "map", "coarse_depth", "a", "fine_depth")
environment = save_environment(record)
powers = latest_record(DATADIR, "l1_mixing.$(MAP)_d$(DEPTH)_a$(numerator(A))-$(denominator(A))")
Cr = [rd(r["C_refined"]) for r in sort(last(read_record(powers)); by = r -> parse(Int, r["j"]))]
densrec = latest_record(DATADIR, "stationary_density")
dens = last(read_record(densrec))
g = grid_constants(ulam_system(MAP), DEPTH, A)

print(run_header(RigorousUlam2D, "bench/$SCRIPT.jl"; map = MAP, depth = DEPTH, a = A, fine = FINE,
    powers = basename(powers), densities = basename(densrec)))
@printf("h = %.6g, var₁₂(ρ_ξ) = %.6g, (h/2) var₁₂(ρ_ξ) = %.6g, M = %d\n", g.h, g.var12ρ, mul_up(0.5 * g.h, g.var12ρ), length(Cr))
F = coarse_fine_norms(Cr, g.h, g.var12ρ)
s = any(<(1), F) ? sigma_from_norms(F) : nothing
s === nothing ? println("no F_k below 1 up to k = $(length(F))") :
    @printf("F_k < 1 first at k = %d; Σ ≤ %.6g at k = %d (α = %.6g)\n", findfirst(<(1), F), s.grid, s.k, s.α)

open_record(record, run_header(RigorousUlam2D, "bench/$SCRIPT.jl"),
    ["map", "coarse_depth", "a", "fine_depth", "Sigma", "k", "defect", "err_apriori", "dist", "err", "err_swapped", "powers_file", "densities_file", "environment"])
for d in FINE
    (MAP, string(DEPTH), string(A), string(d)) in done && continue
    i = findfirst(r -> r["map"] == MAP && r["depth"] == string(d) && r["a"] == string(A), dens)
    i === nothing && (println("depth $d: no row in $(basename(densrec))"); continue)
    g′ = grid_constants(ulam_system(MAP), d, A)
    defect, ε1, ε2 = defect_apriori(g′.h, g′.var12ρ), rd(dens[i]["eps1"]), rd(dens[i]["eps2"])
    if s === nothing
        append_record(record, [MAP, DEPTH, A, d, "Inf", 0, write_endpoint(defect)[1], "Inf", "Inf", "Inf", "Inf", basename(powers), basename(densrec), environment])
        continue
    end
    err_apriori = mul_up(s.grid, defect)
    dist = fixed_point_distance(s.grid, ε1, ε2)
    err = add_up(err_apriori, dist)
    err_swapped = swapped_error(s.grid, g′.h, g′.var12ρ, ε1, ε2)
    @printf("depth %2d: (3h'/2) var₁₂(ρ_ξ) = %.6g, ‖f_ξ − f_h'‖₁ ≤ %.6g, ‖f̃ − f_h'‖₁ ≤ %.3e, ‖f_ξ − f̃‖₁ ≤ %.6g, swapped ≤ %.6g\n",
        d, defect, err_apriori, dist, err, err_swapped)
    append_record(record, [MAP, DEPTH, A, d, write_endpoint(s.grid)[1], s.k, write_endpoint(defect)[1],
        write_endpoint(err_apriori)[1], write_endpoint(dist)[1], write_endpoint(err)[1], write_endpoint(err_swapped)[1], basename(powers), basename(densrec), environment])
end
println("recorded in $(basename(record))")
