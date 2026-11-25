# Two enclosures L_lo ≤ L ≤ L_lo + U of the same Ulam matrix, compared entry by entry: the blocks
# of bench/tree_deepen.jl for one map and depth built on two hosts (or by two drivers).  Both
# enclose the same L, so every entry's two intervals must meet; the driver counts those that do
# not, and reports the largest differences of the lower ends and of the widths, and the ℓ¹ column
# norms of U.  Nothing is certified by this comparison; a disjoint entry would show that one of
# the enclosures is wrong.
#
# Env: MAP, DEPTH, HOSTS (two host names, "nighthawk,ibis"; the blocks are block_file(...; host)).
# Record (CertifiedRecords): ~/ResearchData/RigorousUlam2D_Data/data/compare_blocks.<commit>.<host>.tsv.
#
#   julia --project=. bench/compare_blocks.jl

using RigorousUlam2D, SparseArrays, Printf
using CertifiedRecords
import Serialization

const SCRIPT = splitext(basename(@__FILE__))[1]
const DATADIR = expanduser(get(ENV, "DATADIR", "~/ResearchData/RigorousUlam2D_Data/data"))
const MAP = get(ENV, "MAP", "henon")
const DEPTH = parse(Int, get(ENV, "DEPTH", "8"))
const HOSTS = split(get(ENV, "HOSTS", "nighthawk,ibis"), ",")

record = record_file(DATADIR, SCRIPT, RigorousUlam2D)
header = run_header(RigorousUlam2D, "bench/$SCRIPT.jl")
files = [block_file(DATADIR, MAP, DEPTH; host = h) for h in HOSTS]
A, B = (Serialization.deserialize(f) for f in files)
size(A.L_lo) == size(B.L_lo) || error("the blocks have sizes $(size(A.L_lo)) and $(size(B.L_lo))")
loA, hiA = A.L_lo, A.L_lo + A.U
loB, hiB = B.L_lo, B.L_lo + B.U
disjoint = nnz(dropzeros!(sparse(loA .> hiB))) + nnz(dropzeros!(sparse(loB .> hiA)))
dlo = maximum(abs.(nonzeros(loA - loB)); init = 0.0)
dU = maximum(abs.(nonzeros(A.U - B.U)); init = 0.0)
colU(U) = maximum(vec(sum(U; dims = 1)); init = 0.0)
@printf("%s depth %d, %s against %s: %d disjoint entries, max |ΔL_lo| %.3e, max |ΔU| %.3e, ‖U‖₁ %.3e and %.3e\n",
    MAP, DEPTH, basename(files[1]), basename(files[2]), disjoint, dlo, dU, colU(A.U), colU(B.U))
open_record(record, header, ["map", "depth", "file1", "file2", "disjoint", "max_dLlo_fl", "max_dU_fl", "U1_1_fl", "U1_2_fl"])
append_record(record, [MAP, DEPTH, basename(files[1]), basename(files[2]), disjoint, dlo, dU, colU(A.U), colU(B.U)])
