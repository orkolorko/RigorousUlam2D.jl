"""
The files of a data deposit for the computed results of `RigorousUlam2D.tex`, collected from the
sources file that `paper/figures.jl` writes beside the table and the figures, and the SHA-256
manifest of them.  The deposit itself is made by `paper/dataverse.jl`.
"""

export DATAVERSE_FILE_LIMIT, deposit_plan, write_manifest

using CertifiedRecords: read_record

"The largest file, in bytes, that Harvard Dataverse accepts by upload: 2.5 GiB."
const DATAVERSE_FILE_LIMIT = 2_684_354_560

const _SOURCES_NAME = r"^(?<script>[^.]+)\.sources\.(?<commit>[0-9a-z]+)\.(?<host>[^.]+)\.txt$"
const _SOURCES_ROW = r"^(?<map>\w+) a = (?<a>\S+): density (?<density>\S+), powers (?<powers>\S+)$"
const _SOURCES_SHA = r"^\s+sha256 (?<sha>[0-9a-f]{64})  (?<file>\S+)$"

"""
    deposit_plan(sources; data, images, blocks = :none, maxsize = DATAVERSE_FILE_LIMIT)
        -> (; commit, host, rows, files, excluded)

The files a deposit of the results named by `sources` holds, where `sources` is the path of a
`figures.sources.<commit>.<host>.txt` written by `paper/figures.jl`, `data` the directory of the
records and `images` that of the figures.  Each entry of `files` is `(; path, directory,
description)`, `directory` the folder of the deposit it goes in:

- `densities`: for every row of the sources file (one row of the table of bounds), the density
  record `stationary_density.*.jls` and its HDF5 copy, the files the sources file gives a SHA-256 for;
- `records`: every record the sources file lists, the record of norms of powers (`powers`) of
  every row, and the record each of these names in a column or header field `u`;
- `environments`: every `Manifest.toml` copy that a record names in its column `environment`;
- `figures`: the sources file and every file of `images` written by the same script at the same
  commit and host (the table and the density figures);
- `matrices`, only when `blocks` is `:coarse` or `:all`: the deterministic block a record of norms
  of powers names in its header (`blockfile = `), and with `:all` also the block a density row of
  the stationary-density record names, the one the density of the table was computed from.

A file larger than `maxsize` bytes goes to `excluded`, as `(; path, directory, reason)`, instead of
`files`, and so does a block that `blocks` leaves out: the deposit can then name it.  An error if a file named is missing, or if a file the sources file gives a SHA-256 for
does not have it.  `rows` holds, for each row, the map, the noise half-width and the file names.
"""
function deposit_plan(sources::AbstractString; data::AbstractString, images::AbstractString,
                      blocks::Symbol = :none, maxsize::Integer = DATAVERSE_FILE_LIMIT)
    blocks in (:none, :coarse, :all) || throw(ArgumentError("blocks must be :none, :coarse or :all"))
    m = match(_SOURCES_NAME, basename(sources))
    m === nothing && throw(ArgumentError("$(basename(sources)) is not named <script>.sources.<commit>.<host>.txt"))
    script, commit, host = m[:script], m[:commit], m[:host]
    recs, rows, sha = String[], NamedTuple[], Dict{String,String}()
    for line in eachline(sources)
        if startswith(line, "== ")
            push!(recs, line[4:end])
        elseif (r = match(_SOURCES_ROW, line)) !== nothing
            push!(rows, (; map = String(r[:map]), a = String(r[:a]), density = String(r[:density]),
                           powers = String(r[:powers]), hdf5 = ""))
        elseif (s = match(_SOURCES_SHA, line)) !== nothing
            isempty(rows) && error("$(basename(sources)): a sha256 line before any row")
            f = String(s[:file])
            sha[f] = String(s[:sha])
            endswith(f, ".h5") && (rows[end] = merge(rows[end], (; hdf5 = f)))
        end
    end
    isempty(rows) && error("$(basename(sources)) names no density")

    files, excluded, seen = NamedTuple[], NamedTuple[], Set{String}()
    function add!(dir, name, dirlabel, description; omit = "")
        name in seen && return
        push!(seen, name)
        path = joinpath(dir, name)
        isfile(path) || error("$name, named by $(basename(sources)), is not in $dir")
        if haskey(sha, name) && file_sha256(path) != sha[name]
            error("$name does not have the SHA-256 $(sha[name]) that $(basename(sources)) gives it")
        end
        if !isempty(omit)
            push!(excluded, (; path, directory = dirlabel, reason = omit))
        elseif filesize(path) > maxsize
            push!(excluded, (; path, directory = dirlabel,
                               reason = "$(filesize(path)) bytes, above the limit of $maxsize"))
        else
            push!(files, (; path, directory = dirlabel, description))
        end
    end

    for r in rows
        what = "$(r.map), noise half-width $(replace(r.a, "//" => "/"))"
        add!(data, r.density, "densities", "computed stationary density, $what (Julia Serialization)")
        isempty(r.hdf5) || add!(data, r.hdf5, "densities", "computed stationary density, $what (HDF5)")
    end

    # the records, and what each names: other records, environments, blocks
    queue = vcat(recs, [r.powers for r in rows])
    i = 0
    while i < length(queue)
        name = queue[i += 1]
        name in seen && continue
        add!(data, name, "records", "CertifiedRecords record")
        header, rs = read_record(joinpath(data, name))
        params = join(header, ", ")
        u = match(r"\bu = ([^,\s]+)", params)
        u === nothing || push!(queue, u.captures[1])
        for env in unique(get(x, "environment", "") for x in rs)
            isempty(env) || add!(data, env, "environments", "Manifest.toml of the run that wrote $name")
        end
        b = match(r"\bblockfile = ([^,\s]+)", params)
        b === nothing || add!(data, b.captures[1], "matrices",
                              "deterministic block L_lo <= L <= L_lo + U read by $name";
                              omit = blocks == :none ? "not deposited (blocks = :none)" : "")
        for x in rs, r in rows
            get(x, "density_file", "") == r.density && haskey(x, "blockfile") &&
                add!(data, x["blockfile"], "matrices",
                     "deterministic block L_lo <= L <= L_lo + U the density $(r.density) was computed from";
                     omit = blocks == :all ? "" : "not deposited (blocks = :$blocks)")
        end
    end

    add!(dirname(sources), basename(sources), "figures", "the records each number and figure was read from")
    for f in sort(readdir(images))
        startswith(f, "$script.") && (endswith(f, ".$commit.$host.pdf") || endswith(f, ".$commit.$host.tex")) || continue
        add!(images, f, "figures", endswith(f, ".tex") ? "table of bounds (LaTeX)" : "figure of a computed density")
    end
    return (; commit, host, rows, files, excluded)
end

"""
    write_manifest(path, files) -> Dict{String,String}

Write to `path` the SHA-256 of every entry of `files` (as `deposit_plan` returns them), one line
`<sha256>  <directory>/<name>` each, sorted by that name, in the format `sha256sum -c` reads from
the root of the deposit; returns the hashes by `<directory>/<name>`.
"""
function write_manifest(path::AbstractString, files)
    h = Dict("$(f.directory)/$(basename(f.path))" => file_sha256(f.path) for f in files)
    open(path, "w") do io
        for k in sort(collect(keys(h)))
            println(io, h[k], "  ", k)
        end
    end
    return h
end
