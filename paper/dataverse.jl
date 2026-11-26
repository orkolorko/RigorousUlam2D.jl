# The deposit on Harvard Dataverse of the computed densities of Theorem E of RigorousUlam2D.tex and of
# the records its table was read from.
#
# Reads one sources file of paper/figures.jl (default: the newest figures.sources.*.<host>.txt in
# ~/ResearchData/RigorousUlam2D_Data/images) and writes into
# ~/ResearchData/RigorousUlam2D_Data/dataverse.<commit>.<host>/, <commit> that of this package:
#
#   files.tsv         one row per file of the deposit: directory in the dataset, name, bytes, SHA-256,
#                     description, local path (`deposit_plan`); the files left out, with the reason,
#                     in excluded.tsv
#   MANIFEST.sha256   the SHA-256 of every file by its path in the dataset (`write_manifest`)
#   README.md         paper/dataverse_README.md, with the rows of the table, the files and the
#                     header of every record appended
#   metadata.json     the citation metadata of the dataset, in the JSON of the Dataverse native API
#                     (POST /api/dataverses/<alias>/datasets)
#
# The deposit is the files of files.tsv, README.md and MANIFEST.sha256.  With DOI set (a draft
# dataset, doi:10.7910/DVN/XXXXXX), each of them is added to the draft by CertifiedRecords.dataverse_add,
# which compares the checksum Dataverse reports with the local one; afterwards the listing of the
# draft is compared with the manifest.  Nothing is published.
#
# Env: SOURCES (path of the sources file), BLOCKS (none | coarse | all: the deterministic blocks the
# records name; default coarse), DOI (unset: no upload), CONTACT (contact e-mail written into
# metadata.json; default `git config user.email`).
#
#   julia --project=paper paper/dataverse.jl

using RigorousUlam2D, CertifiedRecords, Printf
using CertifiedRecords: read_record

const SCRIPT = splitext(basename(@__FILE__))[1]
const ROOT = expanduser(get(ENV, "DATAROOT", "~/ResearchData/RigorousUlam2D_Data"))
const DATA = joinpath(ROOT, "data")
const IMAGES = joinpath(ROOT, "images")
const BLOCKS = Symbol(get(ENV, "BLOCKS", "coarse"))
const DOI = get(ENV, "DOI", "")

commit = try readchomp(`git -C $(pkgdir(RigorousUlam2D)) rev-parse --short HEAD`) catch; "unknown" end
host = gethostname()
sources = get(ENV, "SOURCES") do
    fs = filter(f -> startswith(f, "figures.sources.") && endswith(f, ".$host.txt"), readdir(IMAGES))
    isempty(fs) && error("no figures.sources.*.$host.txt in $IMAGES")
    joinpath(IMAGES, fs[argmax([mtime(joinpath(IMAGES, f)) for f in fs])])
end
out = mkpath(joinpath(ROOT, "$SCRIPT.$commit.$host"))
println("# paper/$SCRIPT.jl, RigorousUlam2D $commit on $host; sources $(basename(sources)), blocks $BLOCKS")

plan = deposit_plan(sources; data = DATA, images = IMAGES, blocks = BLOCKS)
for e in plan.excluded
    println("left out: $(e.directory)/$(basename(e.path)), $(e.reason)")
end

# the manifest covers the files of the plan; README.md, written below, is the one file it cannot hold
sha = write_manifest(joinpath(out, "MANIFEST.sha256"), plan.files)
key(f) = "$(f.directory)/$(basename(f.path))"
open(joinpath(out, "files.tsv"), "w") do io
    println(io, "# paper/$SCRIPT.jl, RigorousUlam2D $commit on $host, from $(basename(sources))")
    println(io, join(["directory", "name", "bytes", "sha256", "description", "path"], '\t'))
    for f in sort(plan.files; by = key)
        println(io, join([f.directory, basename(f.path), filesize(f.path), sha[key(f)], f.description, f.path], '\t'))
    end
end
open(joinpath(out, "excluded.tsv"), "w") do io
    println(io, join(["directory", "name", "reason", "path"], '\t'))
    foreach(e -> println(io, join([e.directory, basename(e.path), e.reason, e.path], '\t')), plan.excluded)
end

# README: the committed text, then what this run read
name(m) = m == "henon" ? "Hénon" : m == "lorenz" ? "Lorenz (s = 1/2)" :
    (r = match(r"^mlorenz_rho([0-9]+)-([0-9]+)$", m)) !== nothing ? "Lorenz (s = 2, ρ = $(r[1])/$(r[2]))" : m
gen = IOBuffer()
println(gen, "## The rows of Table 1\n")
println(gen, "| map | a | density (HDF5) | density (Julia) | norms of powers |")
println(gen, "|---|---|---|---|---|")
for r in plan.rows
    println(gen, "| $(name(r.map)) | $(replace(r.a, "//" => "/")) | `$(r.hdf5)` | `$(r.density)` | `$(r.powers)` |")
end
println(gen, "\n## Files\n")
println(gen, "| path | bytes | SHA-256 |")
println(gen, "|---|---:|---|")
for f in sort(plan.files; by = key)
    println(gen, "| `$(key(f))` | $(filesize(f.path)) | `$(sha[key(f)])` |")
end
if !isempty(plan.excluded)
    println(gen, "\nNamed by the records but not in the dataset; the blocks are written by `bench/tree_deepen.jl` " *
                 "of RigorousUlam2D.jl (the upload limit is $DATAVERSE_FILE_LIMIT bytes):\n")
    println(gen, "| file | bytes | SHA-256 | why |")
    println(gen, "|---|---:|---|---|")
    for e in plan.excluded
        println(gen, "| `$(basename(e.path))` | $(filesize(e.path)) | `$(file_sha256(e.path))` | $(e.reason) |")
    end
end
println(gen, "\n## The header of every record\n")
for f in sort(filter(f -> f.directory == "records", plan.files); by = key)
    println(gen, "`$(key(f))`:\n\n```")
    foreach(l -> println(gen, l), first(read_record(f.path)))
    println(gen, "```\n")
end
println(gen, "This README and MANIFEST.sha256 were written by paper/$SCRIPT.jl of RigorousUlam2D.jl at $commit on $host from $(basename(sources)).")
matrices = BLOCKS == :none ? "" :
    "`matrices/` holds the deterministic blocks the records name: Julia `Serialization` files of a named " *
    "tuple whose fields `L_lo` and `U` are sparse matrices with `L_lo ≤ L ≤ L_lo + U` entrywise, `L` the Ulam " *
    "matrix of the map on the grid of the depth in the file name, indexed as `build_uniform_partition` of " *
    "RigorousUlam2D.jl orders the cells.\n"
readme = read(joinpath(@__DIR__, "dataverse_README.md"), String)
readme = replace(readme, "<!-- MATRICES -->\n" => matrices, "<!-- GENERATED -->\n" => String(take!(gen)))
write(joinpath(out, "README.md"), readme)

# metadata.json, the citation block of the Dataverse native API
contact = get(ENV, "CONTACT") do
    try readchomp(`git config user.email`) catch; "" end
end
prim(t, v) = Dict("typeName" => t, "typeClass" => "primitive", "multiple" => false, "value" => v)
comp(t, vs) = Dict("typeName" => t, "typeClass" => "compound", "multiple" => true, "value" => vs)
authors = ["Pacifico, Maria José", "Bioni Liberalquino, Ricardo", "Nisoli, Isaia"]
description = "The computed stationary densities of the random Hénon map and of two random " *
    "Lorenz maps with uniform noise on a square, on a grid of 4096 × 4096 cells, for the " *
    "$(length(plan.rows)) pairs of map and noise half-width of Theorem E of the related publication, " *
    "each with a certified bound on its L¹ distance from the stationary density; the densities in HDF5 " *
    "and as Julia records, the records of the bounds, the run environments, the table and the figures. " *
    "Produced by RigorousUlam2D.jl (https://github.com/orkolorko/RigorousUlam2D.jl)."
fields = Any[
    prim("title", "Certified stationary densities of random Hénon and contracting Lorenz maps"),
    comp("author", [Dict("authorName" => prim("authorName", a)) for a in authors]),
    comp("datasetContact", [Dict("datasetContactName" => prim("datasetContactName", "Nisoli, Isaia"),
                                 "datasetContactEmail" => prim("datasetContactEmail", contact))]),
    comp("dsDescription", [Dict("dsDescriptionValue" => prim("dsDescriptionValue", description))]),
    Dict("typeName" => "subject", "typeClass" => "controlledVocabulary", "multiple" => true,
         "value" => ["Mathematical Sciences"]),
    comp("keyword", [Dict("keywordValue" => prim("keywordValue", k)) for k in
        ["random dynamical systems", "stationary density", "Ulam method", "transfer operator",
         "computer-assisted proof", "Hénon map", "contracting Lorenz map", "interval arithmetic"]]),
    comp("publication", [Dict("publicationCitation" => prim("publicationCitation",
        "Pacifico, M. J., Bioni Liberalquino, R., Nisoli, I.: On the existence, regularity and rigorous " *
        "computation of stationary densities of systems with additive BV noise (in preparation)"))]),
]
meta = Dict("datasetVersion" => Dict(
    "license" => Dict("name" => "CC0 1.0", "uri" => "http://creativecommons.org/publicdomain/zero/1.0"),
    "metadataBlocks" => Dict("citation" => Dict("displayName" => "Citation Metadata", "fields" => fields))))
write(joinpath(out, "metadata.json"), CertifiedRecords.JSON.json(meta))
@printf("%d files, %.1f MB, %d left out; wrote files.tsv, MANIFEST.sha256, README.md, metadata.json in %s\n",
        length(plan.files), sum(filesize(f.path) for f in plan.files) / 1e6, length(plan.excluded), out)

isempty(DOI) && exit()
up = vcat([(f.path, f.directory, f.description) for f in sort(plan.files; by = key)],
          [(joinpath(out, n), "", d) for (n, d) in (("README.md", "how the files are read"),
                                                    ("MANIFEST.sha256", "SHA-256 of every file"))])
for (path, dir, d) in up
    f = dataverse_add(DOI, path; directory = dir, description = d)
    println("added $(isempty(dir) ? "" : dir * "/")$(f.label), $(f.size) bytes, $(f.checksum_type) $(f.checksum)")
end
listed = Dict((isempty(f.directory) ? "" : f.directory * "/") * f.label => f for f in
              dataverse_files(DOI; version = ":draft", token_file = default_token_file()))
for (path, dir, _) in up
    k = (isempty(dir) ? "" : dir * "/") * basename(path)
    haskey(listed, k) || error("$k is not in the draft of $DOI")
    g = listed[k]
    (g.size == filesize(path) && g.checksum == file_checksum(path, g.checksum_type)) ||
        error("$k: the draft has $(g.size) bytes and $(g.checksum_type) $(g.checksum), the local file differs")
end
println("the draft of $DOI lists every file, with the size and checksum of the local one")
