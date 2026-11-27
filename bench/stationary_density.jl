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

using RigorousUlam2D, Printf
using CertifiedRecords
import Serialization

const SCRIPT = splitext(basename(@__FILE__))[1]
const DATADIR = expanduser(get(ENV, "DATADIR", "~/ResearchData/RigorousUlam2D_Data/data"))
const MAP = get(ENV, "MAP", "henon")
const DEPTH = parse(Int, get(ENV, "DEPTH", "8"))
const A = let s = get(ENV, "A", "51//512"); p, q = parse.(Int, split(s, "//")); p // q end
const TOL = parse(Float64, get(ENV, "TOL", "1e-14"))
const MAXITER = parse(Int, get(ENV, "MAXITER", "20000"))

record = record_file(DATADIR, SCRIPT, RigorousUlam2D)
(MAP, string(DEPTH), string(A)) in recorded_keys(record, "map", "depth", "a") &&
    (println("$MAP depth $DEPTH a = $A is in $(basename(record))"); exit())
densfile = replace(record_file(DATADIR, "$SCRIPT.$(MAP)_d$(DEPTH)_a$(numerator(A))-$(denominator(A))", RigorousUlam2D),
                   r"\.tsv$" => ".jls")

blkfile = block_file(DATADIR, MAP, DEPTH)
t0 = time()
P = noisy_ulam_operator(ulam_system(MAP), Serialization.deserialize(blkfile), DEPTH, A)
header = run_header(RigorousUlam2D, "bench/$SCRIPT.jl")
environment = save_environment(record)
print(run_header(RigorousUlam2D, "bench/$SCRIPT.jl"; map = MAP, depth = DEPTH, a = A, tol = TOL, blockfile = basename(blkfile)))
@printf("%s depth %d n = %d, a = %s, h = %.6g, var₁₂(ρ_ξ) = %.6g; η₁ = %.3e, η₁ᴬ = %.3e (%.1f s)\n",
    MAP, DEPTH, P.n, A, P.h, P.var12ρ, P.info.η1, P.info.eta1A, time() - t0)

tp = @elapsed fp = ulam_fixed_point(P.info.Amul, P.n; tol = TOL, maxiter = MAXITER)
res = fixed_point_residual(P.info.Amul, fp.v, P.info.η1, P.info.eta1A)
cv = coordinate_variations(fp.v, P.k, P.h)
defect = defect_apriori(P.h, P.var12ρ)
@printf("power iteration: %d iterations, last ℓ¹ change %.3e, converged %s (%.1f s)\n", fp.iterations, fp.diff, fp.converged, tp)
@printf("ε₁ ≤ %.3e, ε₂ ≤ %.3e, var₁(f̃) ≤ %.6g, var₂(f̃) ≤ %.6g, (3h/2) var₁₂(ρ_ξ) = %.6g\n", res.ε1, res.ε2, cv.var1, cv.var2, defect)

Serialization.serialize(densfile, (; map = MAP, depth = DEPTH, a = A, density = fp.v, header, environment))
open_record(record, header, ["map", "depth", "a", "n", "iterations", "converged", "eps1", "eps2", "var1f", "var2f", "defect",
    "blockfile", "density_file", "seconds", "environment"])
append_record(record, [MAP, DEPTH, A, P.n, fp.iterations, fp.converged, write_endpoint(res.ε1)[1],
    write_endpoint(res.ε2)[1], write_endpoint(cv.var1)[1], write_endpoint(cv.var2)[1], write_endpoint(defect)[1], basename(blkfile), basename(densfile),
    round(time() - t0; digits = 1), environment])
println("recorded in $(basename(record)), density in $(basename(densfile)) ($(round(time() - t0; digits = 1)) s)")
