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

using RigorousUlam2D, SparseArrays, Printf
using BallArithmetic: mul_up
using CertifiedRecords
import Serialization

const SCRIPT = splitext(basename(@__FILE__))[1]
const DATADIR = expanduser(get(ENV, "DATADIR", "~/ResearchData/RigorousUlam2D_Data/data"))
const MAP = get(ENV, "MAP", "henon")
const DEPTH = parse(Int, get(ENV, "DEPTH", "8"))
const A = let s = get(ENV, "A", "51//512"); p, q = parse.(Int, split(s, "//")); p // q end
const M = parse(Int, get(ENV, "M", "150"))
const BLOCK = parse(Int, get(ENV, "BLOCK", "8"))
const COLS = let s = get(ENV, "COLS", ""); isempty(s) ? nothing : (r = parse.(Int, split(s, ":")); r[1]:r[2]) end

amp = "a$(numerator(A))-$(denominator(A))"
summary = record_file(DATADIR, SCRIPT, RigorousUlam2D)
(MAP, string(DEPTH), string(A)) in recorded_keys(summary, "map", "depth", "a") &&
    (println("$MAP depth $DEPTH a = $A is in $(basename(summary))"); exit())
part = COLS === nothing ? nothing : record_file(DATADIR, "$SCRIPT.$(MAP)_d$(DEPTH)_$(amp)_cols$(first(COLS))-$(last(COLS))", RigorousUlam2D)
part !== nothing && isfile(part) && (println("$(basename(part)) exists"); exit())
powers = record_file(DATADIR, "$SCRIPT.$(MAP)_d$(DEPTH)_$amp", RigorousUlam2D)
COLS === nothing && isfile(powers) && error("$powers exists without its summary row; remove it to run again")
environment = save_environment(part === nothing ? summary : part)

sys = ulam_system(MAP)
blkfile = block_file(DATADIR, MAP, DEPTH)
t0 = time()
P = noisy_ulam_operator(sys, Serialization.deserialize(blkfile), DEPTH, A)
densrec = try latest_record(DATADIR, "stationary_density") catch; nothing end
densrow = densrec === nothing ? nothing :
    findfirst(r -> r["map"] == MAP && r["depth"] == string(DEPTH) && r["a"] == string(A), last(read_record(densrec)))
densfile = densrow === nothing ? nothing : last(read_record(densrec))[densrow]["density_file"]
u = densfile === nothing ? ulam_fixed_point(P.info.Amul, P.n).v :
    Serialization.deserialize(joinpath(DATADIR, densfile)).density
length(u) == P.n || error("the density has length $(length(u)), the operator n = $(P.n)")
cols = COLS === nothing ? (1:P.n) : COLS

header = run_header(RigorousUlam2D, "bench/$SCRIPT.jl"; map = MAP, depth = DEPTH, a = A, M, block = BLOCK,
    cols = "$(first(cols)):$(last(cols))", n = P.n, blockfile = basename(blkfile), blocksha = file_sha256(blkfile),
    u = densfile === nothing ? "power iteration" : densfile, η1 = P.info.η1, eta1A = P.info.eta1A, normP = P.normP)
print(header)
@printf("%s depth %d n = %d, a = %s, h = %.6g, var₁₂(ρ_ξ) = %.6g, columns %s (%.1f s)\n", MAP, DEPTH, P.n, A, P.h, P.var12ρ, cols, time() - t0)
flush(stdout)

tc = @elapsed C = norms_of_powers_columns(P.info.Bmul!, P.n, M; η1 = P.info.η1, eta1A = P.info.eta1A,
    normP = P.normP, u, block = BLOCK, verbose = true, cols)
@printf("columns: %d × %d products in %.1f s\n", length(cols), M, tc)
if part !== nothing
    open_record(part, header * "# environment = $environment\n# seconds = $(round(tc; digits = 1))\n", ["j", "C", "bits"])
    for j in 1:M
        append_record(part, [j, write_endpoint(C[j])[1], 53])
    end
    println("recorded in $(basename(part))")
else
    s = record_column_sweep(DATADIR, SCRIPT, MAP, DEPTH, A, C, header, environment, tc; P.h, P.var12ρ, P.n)
    s === nothing ? println("no C_j below 1 up to j = $M") :
        @printf("Σ_grid ≤ %.6g at k = %d (α = %.6g), Σ_V ≤ %.6g\n", s.grid, s.k, s.α, s.V)
    println("recorded in $(basename(powers)) and $(basename(summary))")
end
