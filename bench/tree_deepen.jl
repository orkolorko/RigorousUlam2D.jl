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
# counts and the widths of U as floating-point diagnostics, `read_seconds` (the reading at LEVEL)
# and `seconds` (the wall clock from the start of the run, store refinement included).
# One log line per depth on stdout: counts, column sums, the widths of U.
# Every row names, in its column `environment`, the copy of the Manifest of the run written
# beside the record by `save_environment`, with the commits of the dependencies given by path.
#
#   julia --project=. -t 32 bench/tree_deepen.jl

using RigorousUlam2D

env(k, d) = get(ENV, k, d)
const D0 = parse(Int, env("D0", "4"))
const D1 = parse(Int, env("D1", "12"))
run_blocks(env("MAP", "henon"); root = expanduser(env("DATAROOT", "~/ResearchData/RigorousUlam2D_Data")),
    d0 = D0, d1 = D1, store_level = parse(Int, env("STORE", "12")), level = parse(Int, env("LEVEL", "14")),
    bisect = parse(Int, env("BISECT", "30")), xinv = env("XINV", "1") == "1",
    depths = let (a, b) = parse.(Int, split(env("DEPTHS", "$D0:$D1"), ":")); a:b end,
    script = "bench/tree_deepen.jl")
