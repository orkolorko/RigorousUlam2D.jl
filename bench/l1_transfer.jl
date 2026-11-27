# The L¹ error of the stationary density on finer grids, from the column sweep of a coarse one.
#
# For one (MAP, A) and a coarse depth DEPTH, the refined bounds C_k ≥ ‖P_h^k|_{V_h}‖₁ of the
# l1_mixing record go through `coarse_fine_norms` (the recursion eq:coarsefine of the paper, after
# Galatolo, Monge, Nisoli 2020, Lemma 29, with θ₁ = θ₂ = θ_h = (h/2) var₁₂(ρ_ξ) of Theorem C) to
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
# one row per (map, coarse depth, a, fine depth); a row already there is not computed again.  The
# column `seconds` is the wall-clock time from the start of the run to the row.
# Every row names, in its column `environment`, the copy of the Manifest of the run written
# beside the record by `save_environment`, with the commits of the dependencies given by path.
#
#   julia --project=. bench/l1_transfer.jl

using RigorousUlam2D

env(k, d) = get(ENV, k, d)
rational(s) = (p = parse.(Int, split(s, "//")); p[1] // p[2])
run_transfer(env("MAP", "lorenz"), parse(Int, env("DEPTH", "8")), rational(env("A", "77//256")),
    parse.(Int, split(env("FINE", "10,12"), ","));
    datadir = expanduser(env("DATADIR", "~/ResearchData/RigorousUlam2D_Data/data")), script = "bench/l1_transfer.jl")
