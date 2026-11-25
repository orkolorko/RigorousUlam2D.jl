# The column sweep of bench/l1_mixing.jl assembled from parts run on several machines with COLS.
#
# Each part record holds, for j = 1, …, M, the maximum over its columns of ‖P_h^j e_i − u‖₁ plus the
# propagated rounding; the parts must have read the same block (the SHA-256 in their headers), the
# same density u and the same M, and their column ranges must cover 1:n without overlap.  The
# entrywise maximum is then the C_j of the whole run, bit for bit, and it is recorded as
# bench/l1_mixing.jl records a whole run (`record_column_sweep`): the powers record and a row of the
# l1_mixing summary, under the name of this script's own commit and host.
#
# Env: MAP, DEPTH, A, PARTS (comma-separated file names of part records in the data directory).
# Every row names, in its column `environment`, the copy of the Manifest of the run written
# beside the record by `save_environment`, with the commits of the dependencies given by path.
#
#   julia --project=. bench/l1_mixing_merge.jl

using RigorousUlam2D, Printf
using CertifiedRecords

const SCRIPT = "l1_mixing"                       # the records it writes are those of l1_mixing
const DATADIR = expanduser(get(ENV, "DATADIR", "~/ResearchData/RigorousUlam2D_Data/data"))
const MAP = get(ENV, "MAP", "henon")
const DEPTH = parse(Int, get(ENV, "DEPTH", "10"))
const A = let s = get(ENV, "A", "51//256"); p, q = parse.(Int, split(s, "//")); p // q end
const PARTS = split(get(ENV, "PARTS", ""), ",")

param(hdr, key) = (m = match(Regex("(?:^|, )$key = ([^,]+)"), join(hdr, ", ")); m === nothing ? error("no $key in the header") : strip(m.captures[1]))
summary = record_file(DATADIR, SCRIPT, RigorousUlam2D)
(MAP, string(DEPTH), string(A)) in recorded_keys(summary, "map", "depth", "a") &&
    (println("$MAP depth $DEPTH a = $A is in $(basename(summary))"); exit())

parts = map(PARTS) do f
    hdr, rows = read_record(joinpath(DATADIR, f))
    params = [lstrip(l, ['#', ' ']) for l in hdr if occursin(" = ", l)]
    (param(params, "map") == MAP && param(params, "depth") == string(DEPTH) && param(params, "a") == string(A)) ||
        error("$f is for another configuration")
    r = parse.(Int, split(param(params, "cols"), ":"))
    (; file = f, cols = r[1]:r[2], n = parse(Int, param(params, "n")), M = parse(Int, param(params, "M")),
       sha = param(params, "blocksha"), u = param(params, "u"), seconds = parse(Float64, param(params, "seconds")),
       C = [Float64(read_endpoint(x["C"], x["bits"]), RoundUp) for x in sort(rows; by = x -> parse(Int, x["j"]))])
end
n, M = parts[1].n, parts[1].M
all(p -> p.n == n && p.M == M && length(p.C) == M, parts) || error("the parts differ in n or M")
all(p -> p.sha == parts[1].sha, parts) || error("the parts read different blocks")
all(p -> p.u == parts[1].u, parts) || error("the parts used different densities u")
order = sortperm([first(p.cols) for p in parts])
first(parts[order[1]].cols) == 1 && last(parts[order[end]].cols) == n &&
    all(i -> first(parts[order[i+1]].cols) == last(parts[order[i]].cols) + 1, 1:length(parts)-1) ||
    error("the column ranges $([p.cols for p in parts]) do not partition 1:$n")
C = reduce((x, y) -> max.(x, y), [p.C for p in parts])

environment = save_environment(summary)
g = grid_constants(ulam_system(MAP), DEPTH, A)
header = run_header(RigorousUlam2D, "bench/l1_mixing_merge.jl"; map = MAP, depth = DEPTH, a = A, M, n,
    parts = join(PARTS, " "), blocksha = parts[1].sha, u = parts[1].u)
print(header)
s = record_column_sweep(DATADIR, SCRIPT, MAP, DEPTH, A, C, header, environment, sum(p.seconds for p in parts);
                        g.h, g.var12ρ, n)
s === nothing ? println("no C_j below 1 up to j = $M") :
    @printf("Σ_grid ≤ %.6g at k = %d (α = %.6g), Σ_V ≤ %.6g\n", s.grid, s.k, s.α, s.V)
