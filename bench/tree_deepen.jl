# The enclosures L_lo ≤ L ≤ L_lo + U of the Ulam matrices of one map of the paper's examples
# (`henon_system` or `lorenz_system`, src/maps.jl) on the uniform grids of depths D0..D1, read off
# an `UlamTree`: the tree is refined on disk to the depth STORE (its subtrees in the store
# directory, resumed if the run is repeated), then read at the depth LEVEL by `ulam_matrices`,
# which quadrisects the uncertain cells in memory and integrates the uncertain leaves through the
# map's `fibred_area` (BISECT bisections, the cut at the inverse of the x-only component unless
# XINV=0). Images are enclosed by the natural interval extension (`zeroth_order_enclosure`).
#
# Env: MAP (henon|lorenz), D0 (4), D1 (12), STORE (12), LEVEL (14), BISECT (30), XINV (1),
# DEPTHS (D0:D1, e.g. "6:10").  Output, one file per depth, an existing one not rewritten:
#     ~/ResearchData/RigorousUlam2D_Data/data/tree_deepen.<map>_d<d>_level<L>_area<B>[_xinv]_t<D1>.<host>.jls
# a named tuple (map, depth, store_level, store_maxlevel, tag, levels, d0, d1, graph, L_lo, U, host,
# script, commits, julia, threads, time) with the matrices indexed as build_uniform_partition(domain, d);
# the store in ~/ResearchData/RigorousUlam2D_Data/bulk/tree_deepen.<map>_d<D0>-<D1>.store.<host>/;
# and a record (CertifiedRecords) data/tree_deepen.<commit>.<host>.tsv, one row per file, with the
# counts and the widths of U as floating-point diagnostics.
# One log line per depth on stdout: counts, column sums, the widths of U.
# Every row names, in its column `environment`, the copy of the Manifest of the run written
# beside the record by `save_environment`, with the commits of the dependencies given by path.
#
#   julia --project=. -t 32 bench/tree_deepen.jl

using RigorousUlam2D, IntervalArithmetic, IntervalBoxes
using SparseArrays, Printf, Dates
using CertifiedRecords
import Serialization

const HOST = chomp(read(`hostname`, String))
const SCRIPT = splitext(basename(@__FILE__))[1]
const ROOT = expanduser(get(ENV, "DATAROOT", "~/ResearchData/RigorousUlam2D_Data"))
const MAP = get(ENV, "MAP", "henon")
const D0 = parse(Int, get(ENV, "D0", "4"))
const D1 = parse(Int, get(ENV, "D1", "12"))
const STORE = parse(Int, get(ENV, "STORE", "12"))
const LEVEL = parse(Int, get(ENV, "LEVEL", "14"))
const BISECT = parse(Int, get(ENV, "BISECT", "30"))
const XINV = get(ENV, "XINV", "1") == "1"
const DEPTHS = let s = get(ENV, "DEPTHS", "$D0:$D1"); a, b = parse.(Int, split(s, ":")); a:b end

gitrev(dir) = try chomp(read(`git -C $dir rev-parse --short HEAD`, String)) catch; "?" end
gitdirty(dir) = try !isempty(read(`git -C $dir status --porcelain --untracked-files=no -- src bench Project.toml Manifest.toml`, String)) catch; true end
const PKG = dirname(dirname(pathof(RigorousUlam2D)))
const COMMITS = (; RigorousUlam2D = gitrev(PKG), dirty = gitdirty(PKG))
COMMITS.dirty && error("$PKG has uncommitted changes in src, bench or the project files; commit before running")

sys = ulam_system(MAP; bisect = BISECT, xinv = XINV)
tag = "level$(LEVEL)_area$(BISECT)$(XINV ? "_xinv" : "")_t$(D1)"
outfile(d) = block_file(joinpath(ROOT, "data"), MAP, d; level = LEVEL, bisect = BISECT, xinv = XINV, d1 = D1, host = HOST)
record = record_file(joinpath(ROOT, "data"), SCRIPT, RigorousUlam2D; host = HOST)
header = run_header(RigorousUlam2D, "bench/$SCRIPT.jl")
environment = save_environment(record)
todo = filter(d -> !isfile(outfile(d)), collect(DEPTHS))
store = joinpath(ROOT, "bulk", "$SCRIPT.$(MAP)_d$(D0)-$(D1).store.$HOST")

println("$SCRIPT on $HOST, $(Dates.now()), julia $VERSION, threads $(Threads.nthreads()), commits $COMMITS")
println("map $MAP, domain $(sys.domain), enclosure zeroth_order_enclosure, bisect $BISECT, xinv $XINV, ",
        "d0 $D0, d1 $D1, store to $STORE, read at $LEVEL, depths $DEPTHS (to do: $todo), store $store")
isempty(todo) && (println("nothing to do"); exit())
flush(stdout)

mkpath(joinpath(ROOT, "data")); mkpath(dirname(store))
t0 = time()
tree = UlamTree(sys.F, sys.domain, D0, D1; enclosure_method = zeroth_order_enclosure, area = sys.area, store)
@printf("store at level %d-%d, %d active, %d resolved (%.1f s)\n", tree.level, tree.maxlevel, tree.nactive, tree.nresolved, time() - t0)
while tree.level < STORE
    t = @elapsed refine!(tree)
    @printf("store refined to level %d-%d in %.1f s, %d active, %d resolved\n", tree.level, tree.maxlevel, t, tree.nactive, tree.nresolved)
    flush(stdout)
end

tr = @elapsed mats = ulam_matrices(tree, minimum(todo):maximum(todo); level = LEVEL)
colmax(A) = maximum(vec(sum(A; dims = 1)); init = 0.0)
rowmax(A) = maximum(vec(sum(A; dims = 2)); init = 0.0)
for m in mats
    m.depth in todo || continue
    lo, hi = extrema(vec(sum(m.L_lo; dims = 1))), extrema(vec(sum(m.L_lo .+ m.U; dims = 1)))
    @printf("%s %s depth %d  read %.1fs  nnz(L_lo) %d  nnz(U) %d  colsums L_lo %.12f..%.12f, L_lo+U %.12f..%.12f  ‖U‖₁ %.3e  ‖U‖_∞ %.3e\n",
        MAP, tag, m.depth, tr, nnz(m.L_lo), nnz(m.U), lo..., hi..., colmax(m.U), rowmax(m.U))
    Serialization.serialize(outfile(m.depth), (; map = MAP, depth = m.depth, store_level = tree.level,
        store_maxlevel = tree.maxlevel, tag, levels = LEVEL, d0 = D0, d1 = D1, graph = m.graph, L_lo = m.L_lo,
        U = m.U, host = HOST, script = SCRIPT, commits = COMMITS, header, environment, julia = string(VERSION),
        threads = Threads.nthreads(), time = tr))
    # diagnostics, summed in floating point: not bounds
    open_record(record, header, ["map", "depth", "level", "bisect", "xinv", "d1", "nnz_L_lo", "nnz_U",
        "colsum_L_lo_min_fl", "colsum_L_hi_max_fl", "U_1_fl", "U_inf_fl", "read_seconds", "file", "environment"])
    append_record(record, [MAP, m.depth, LEVEL, BISECT, XINV, D1, nnz(m.L_lo), nnz(m.U), lo[1], hi[2],
        colmax(m.U), rowmax(m.U), round(tr; digits = 1), basename(outfile(m.depth)), environment])
    flush(stdout)
end
@printf("total %.1f s\n", time() - t0)
