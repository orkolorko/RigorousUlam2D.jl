# The computed stationary density of the noisy Ulam operator P_h = π N_ξ π L π of one map at one
# depth and one noise amplitude (`noisy_ulam_operator` of the block of bench/tree_deepen.jl on this
# host), with its a posteriori quantities: the density f̃ is the power iteration `ulam_fixed_point`
# on the masses; ε₁ ≥ ‖P_h f̃ − f̃‖₁ and ε₂ ≥ |Σ f̃ − 1| (`fixed_point_residual`, with the ℓ¹ rounding
# model η₁ of `noisy_operator`, which covers the rounding of the products and the enclosure of L);
# var₁(f̃), var₂(f̃), the coordinate variations on the torus of the piecewise-constant density
# (`coordinate_variations`); and the
# a priori defect (3h/2) var₁₂(ρ_ξ) ≥ ‖(L_ξ − P_h) f_ξ‖₁ of src/l1_error.jl.
#
# Env: MAP, DEPTH, A (rational such as 51//512), TOL (1e-14), MAXITER (20000).  Output, in
# ~/ResearchData/RigorousUlam2D_Data/data:
#     stationary_density.<map>_d<depth>_a<p>-<q>.<commit>.<host>.jls   the density (cell masses, grid order)
#     stationary_density.<commit>.<host>.tsv   (CertifiedRecords) one row per (map, depth, a), the
#                                              bounds as exact doubles
# A configuration already in the record is not run again.  The column `seconds` is the wall-clock
# time of the run, from reading the block to the record.
# Every row names, in its column `environment`, the copy of the Manifest of the run written
# beside the record by `save_environment`, with the commits of the dependencies given by path.
#
#   julia --project=. -t 16 bench/stationary_density.jl

using RigorousUlam2D

env(k, d) = get(ENV, k, d)
rational(s) = (p = parse.(Int, split(s, "//")); p[1] // p[2])
run_stationary_density(env("MAP", "henon"), parse(Int, env("DEPTH", "8")), rational(env("A", "51//512"));
    datadir = expanduser(env("DATADIR", "~/ResearchData/RigorousUlam2D_Data/data")),
    tol = parse(Float64, env("TOL", "1e-14")), maxiter = parse(Int, env("MAXITER", "20000")),
    script = "bench/stationary_density.jl")
