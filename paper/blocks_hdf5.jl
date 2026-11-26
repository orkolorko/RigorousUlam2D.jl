# Archival HDF5 copies of the deterministic blocks the densities of Theorem E of RigorousUlam2D.tex were
# computed from, for the data deposit (paper/dataverse.jl).
#
# Reads the sources file of paper/figures.jl named by SOURCES, finds with `deposit_plan` the block of
# bench/tree_deepen.jl that each density row was computed from, and writes into
# ~/ResearchData/RigorousUlam2D_Data/data, for each block and each of its matrices L_lo and U:
#
#   blocks_hdf5.<block>.<L_lo|U>.<commit>.<host>.h5
#       the matrix in grid order, compressed sparse columns with HDF5's shuffle and deflate filters
#       (`write_block_hdf5`), with the block's metadata as attributes
#
# A copy already in data (any commit) is not written again.  Each copy is read back and compared with
# the block, entry by entry, before the next one is written.
#
# Env: SOURCES (path of the sources file; required), DEFLATE (6).
#
#   julia --project=paper paper/blocks_hdf5.jl

using RigorousUlam2D, HDF5, Printf
using SparseArrays: SparseMatrixCSC
import Serialization

const SCRIPT = splitext(basename(@__FILE__))[1]
@assert SCRIPT == BLOCK_HDF5_SCRIPT
const ROOT = expanduser(get(ENV, "DATAROOT", "~/ResearchData/RigorousUlam2D_Data"))
const DATA = joinpath(ROOT, "data")
const DEFLATE = parse(Int, get(ENV, "DEFLATE", "6"))
sources = ENV["SOURCES"]

commit = try readchomp(`git -C $(pkgdir(RigorousUlam2D)) rev-parse --short HEAD`) catch; "unknown" end
host = gethostname()
println("# paper/$SCRIPT.jl, RigorousUlam2D $commit on $host; sources $(basename(sources)), deflate $DEFLATE")

plan = deposit_plan(sources; data = DATA, images = joinpath(ROOT, "images"), blocks = :none)
for path in plan.fine
    stem = replace(basename(path), r"\.jls$" => "")
    todo = filter(field -> !any(f -> startswith(f, "$SCRIPT.$stem.$field.") && endswith(f, ".h5"), readdir(DATA)),
                  ("L_lo", "U"))
    isempty(todo) && (println("$stem: both copies exist"); continue)
    t0 = time()
    blk = Serialization.deserialize(path)
    perm = grid_order(ulam_system(String(blk.map)).domain, blk.depth)
    @printf("%s: read in %.1f s, %d cells\n", stem, time() - t0, length(perm))
    for field in todo
        out = joinpath(DATA, "$SCRIPT.$stem.$field.$commit.$host.h5")
        t1 = time()
        write_block_hdf5(out, blk, Symbol(field); perm, source = basename(path), deflate = DEFLATE)
        A = getfield(blk, Symbol(field))[perm, perm]
        B = h5open(out) do f
            SparseMatrixCSC(size(A)..., read(f["colptr"]), read(f["rowval"]), read(f["nzval"]))
        end
        B == A || error("$out does not read back as $field of $(basename(path))")
        @printf("%s: %s, %d bytes (%.3f of the limit %d), %.1f s, read back equal\n", basename(out), field,
                filesize(out), filesize(out) / DATAVERSE_FILE_LIMIT, DATAVERSE_FILE_LIMIT, time() - t1)
    end
end
