# The ℓ¹ norms of the powers of the noisy Ulam operator P_h = π N_ξ π L π on the zero-sum vectors
# V_h, by columns (`norms_of_powers_columns`): C_j ≥ ‖P_h^j|_{V_h}‖₁, the maximum over the cells i
# of ‖P_h^j e_i − u‖₁ plus the propagated rounding, every column iterated in floating point with
# the ℓ¹ rounding model η₁ of `noisy_operator` and ‖P_h‖₁ ≤ ‖L_lo + U‖₁.  The operator is
# `noisy_ulam_operator` of the block of bench/tree_deepen.jl at this depth on this host.  u, a free
# choice in the bound, is the density of the stationary_density record of the same (map, depth, a)
# when there is one, else the power iteration.  The C_j are refined by submultiplicativity
# (`refine_norms_of_powers`); `sigma_from_norms` gives Σ_grid ≥ ‖(I − P_h)⁻¹|_{V_h}‖₁ and
# Σ_V = 1 + Σ_grid, and with the a priori defect (3h/2) var₁₂(ρ_ξ), ‖f_ξ − f_h‖₁ ≤ Σ_V · (3h/2) var₁₂(ρ_ξ)
# (src/l1_error.jl).
#
# Env: MAP, DEPTH, A (rational such as 51//512), M (largest power, 150), BLOCK (columns per
# block, 8: the time per column does not depend on it, the memory is 2 n BLOCK doubles a thread).
# Records (CertifiedRecords), in ~/ResearchData/RigorousUlam2D_Data/data:
#     l1_mixing.<map>_d<depth>_a<p>-<q>.<commit>.<host>.tsv  j, C_j and refined C_j as exact doubles
#     l1_mixing.<commit>.<host>.tsv                         one row per (map, depth, a): Σ and the bound
# A configuration already in the second record is not run again.
#
# With COLS (a range such as 1:524288) only those columns are iterated, and the record is
#     l1_mixing.<map>_d<depth>_a<p>-<q>_cols<first>-<last>.<commit>.<host>.tsv   j, max over COLS
# with the SHA-256 of the block in its header and no summary row; bench/l1_mixing_merge.jl takes
# the maximum of parts covering all the columns, which is the C of the whole run bit for bit.
# Every row names, in its column `environment`, the copy of the Manifest of the run written
# beside the record by `save_environment`, with the commits of the dependencies given by path.
#
#   julia --project=. -t 16 bench/l1_mixing.jl

using RigorousUlam2D

env(k, d) = get(ENV, k, d)
rational(s) = (p = parse.(Int, split(s, "//")); p[1] // p[2])
run_column_sweep(env("MAP", "henon"), parse(Int, env("DEPTH", "8")), rational(env("A", "51//512"));
    datadir = expanduser(env("DATADIR", "~/ResearchData/RigorousUlam2D_Data/data")),
    M = parse(Int, env("M", "150")), block = parse(Int, env("BLOCK", "8")),
    cols = let s = env("COLS", ""); isempty(s) ? nothing : (r = parse.(Int, split(s, ":")); r[1]:r[2]) end,
    script = "bench/l1_mixing.jl")
