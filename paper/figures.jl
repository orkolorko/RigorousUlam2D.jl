# The certified numbers on record for RigorousUlam2D.tex, and figures of the computed densities.
#
# Reads, from ~/ResearchData/RigorousUlam2D_Data/data on this host, every record (all commits, a later
# row replacing an earlier one of the same configuration) of
# bench/stationary_density.jl, bench/l1_mixing.jl and bench/l1_transfer.jl (CertifiedRecords), and
# writes into ~/ResearchData/RigorousUlam2D_Data/images:
#
#   figures.density_<map>_a<p>-<q>_d<depth>.<commit>.<host>.pdf
#       the computed stationary density f̃ at depth DEPTH (12) of each (map, a) whose L¹ bound at
#       that depth is finite, its cell masses summed over blocks of 2^(DEPTH − SHOW) × 2^(DEPTH − SHOW)
#       cells for display (SHOW = 9: 512 × 512 pixels) and divided by the block area; the title
#       gives the bound ‖f_ξ − f̃‖₁ ≤ b, b the smallest err_swapped on record for that depth over
#       the coarse depths, written upward at 3 digits (fmt_upper);
#   figures.marginal_<map>_d<depth>.<commit>.<host>.pdf
#       for each Lorenz map, the density of the x-marginal of f̃ (x_marginal) at every noise size
#       with a finite bound, the bound of the density in the legend: integrating out y is an L¹
#       contraction, so it bounds the marginal too;
#   figures.marginal_<map>_a<p>-<q>_d<depth>.<commit>.<host>.pdf
#       for each Lorenz map, the same marginal at the smallest noise size alone, the size of a
#       density panel;
#   figures.table_bounds.<commit>.<host>.tex
#       a LaTeX tabular, one row per (map, a): the coarse depth of the best bound, Σ_V of that
#       sweep, the transfer's Σ, ε₁ and ε₂ of f̃ at DEPTH, and the bounds at DEPTH by the direct
#       and the swapped route; every number an upper bound written upward at 3 digits;
#   figures.sources.<commit>.<host>.txt
#       the record files every number and figure was read from, with their headers, and for
#       each (map, a) the density record and the HDF5 copy below with their SHA-256;
#
# and into ~/ResearchData/RigorousUlam2D_Data/data:
#
#   figures.density_<map>_a<p>-<q>_d<depth>.<commit>.<host>.h5
#       the cell masses of f̃, bit for bit those of the stationary_density record, with the
#       metadata needed to read them without this package (`write_density_hdf5`): the archival
#       copy for the data repository.  A copy already in data/ for the same record, with the same
#       masses, is reused rather than written again under this commit's name.
#
# The density in a figure is the computed f̃ (cell masses / cell area), not the stationary density
# itself; the bound in the title is the certified distance between the two.  Rendering is headless.
#
# Env: DEPTH (12), SHOW (9), HDF5_FROM (the commit of the sources file whose HDF5 copies are
# preferred; default 54dff0a, the copies of the Dataverse dataset doi:10.7910/DVN/2OTJ8O), HOSTS (hosts whose records are read, "ibis,albatross"), MAPS
# ("lorenz,mlorenz_rho18-5,henon").
#
#   julia --project=paper paper/figures.jl

ENV["GKSwstype"] = "100"
using RigorousUlam2D, CertifiedRecords, CairoMakie, Printf, HDF5
using IntervalArithmetic: inf, sup
import Serialization

const SCRIPT = splitext(basename(@__FILE__))[1]
const ROOT = expanduser(get(ENV, "DATAROOT", "~/ResearchData/RigorousUlam2D_Data"))
const DATA = joinpath(ROOT, "data")
const IMAGES = joinpath(ROOT, "images")
const DEPTH = parse(Int, get(ENV, "DEPTH", "12"))
const SHOW = parse(Int, get(ENV, "SHOW", "9"))
const HOSTS = split(get(ENV, "HOSTS", "ibis,albatross"), ",")       # the records of these hosts, in the annex here
const MAPS = split(get(ENV, "MAPS", "lorenz,mlorenz_rho18-5,henon"), ",")

commit = try readchomp(`git -C $(pkgdir(RigorousUlam2D)) rev-parse --short HEAD`) catch; "unknown" end
host = gethostname()
out(name, ext) = joinpath(IMAGES, "$SCRIPT.$name.$commit.$host.$ext")
up3(x) = fmt_upper(Upper(x); digits = 3)
# A decimal string `d.ddde±x` written for LaTeX: plainly when -2 ≤ x ≤ 3 (the decimal point is
# moved, so the value, and with it the upward rounding, is unchanged), else as d.dd × 10^x.
# `unicode = true` gives the same for plain text (figure titles): d.dd×10⁻ˣ.
const _SUP = Dict(zip("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))
function tex(s; unicode::Bool = false)
    m = match(r"^(-?)([0-9])(?:\.([0-9]*))?e([+-][0-9]+)$", s)
    m === nothing && return s
    sg, d1, rest, e = m.captures[1], m.captures[2], something(m.captures[3], ""), parse(Int, m.captures[4])
    mant = isempty(rest) ? d1 : "$d1.$rest"
    -2 <= e <= 3 || return unicode ? "$sg$mant×10" * map(c -> _SUP[c], string(e)) : "$sg$mant\\times10^{$e}"
    digs = d1 * rest
    p = 1 + e                                         # digits before the decimal point
    str = p <= 0 ? "0." * "0"^(-p) * digs :
          p >= length(digs) ? digs * "0"^(p - length(digs)) : digs[1:p] * "." * digs[p+1:end]
    return sg * str
end
rd(s) = s == "Inf" ? Inf : Float64(read_endpoint(s, 53), RoundUp)

# the rows of every record of each driver, oldest first, a later row replacing an earlier one
# with the same key; each row remembers the record it came from
function rows_of(prefix, key)
    byk = Dict{Any,Dict{String,String}}()
    for f in sort(reduce(vcat, [records(DATA, prefix; host) for host in HOSTS]); by = mtime)
        for r in last(read_record(f))
            r["record"] = basename(f)
            byk[Tuple(r[c] for c in key)] = r
        end
    end
    return collect(values(byk))
end
dens = rows_of("stationary_density", ("map", "depth", "a"))
mix = rows_of("l1_mixing", ("map", "depth", "a"))
tr = rows_of("l1_transfer", ("map", "coarse_depth", "a", "fine_depth"))
sources = sort(unique(vcat([r["record"] for r in dens], [r["record"] for r in mix], [r["record"] for r in tr])))
mkpath(IMAGES)

rows = []
for m in MAPS, a in ("77//256", "51//256", "51//512", "13//256")
    cand = filter(r -> r["map"] == m && r["a"] == a && r["fine_depth"] == string(DEPTH) && r["err_swapped"] != "Inf", tr)
    isempty(cand) && continue
    best = cand[argmin([rd(r["err_swapped"]) for r in cand])]
    d = only(filter(r -> r["map"] == m && r["a"] == a && r["depth"] == string(DEPTH), dens))
    sw = only(filter(r -> r["map"] == m && r["a"] == a && r["depth"] == best["coarse_depth"], mix))
    push!(rows, (; map = m, a, coarse = best["coarse_depth"], SigmaV = rd(sw["Sigma_V"]), Sigma = rd(best["Sigma"]),
                 eps1 = rd(d["eps1"]), eps2 = rd(d["eps2"]), err = rd(best["err"]), errs = rd(best["err_swapped"]),
                 density = d["density_file"], powers = sw["powers_file"]))
end

name(m) = m == "henon" ? "Hénon" : m == "lorenz" ? "Lorenz-like" : startswith(m, "mlorenz") ? "Contracting Lorenz" : m
hdf5 = Dict{Any,String}()
marg = Dict{Any,Vector{Float64}}()

"The side ξ = 2a of the noise square, as \"p/q\", from the half-width a of the records (\"p//q\")."
xi(a) = (q = 2 * parse(Rational{Int}, replace(a, "//" => "/")); "$(numerator(q))/$(denominator(q))")

# An HDF5 copy already on record for this density: a file `<stem>*.h5` in DATA whose attribute
# `source` names the same density record and whose masses equal the record's, bit for bit; the
# oldest such file.  Reusing it keeps the names (and hashes) cited by the paper and the
# deposit stable across reruns, which would otherwise write a copy named after each commit.
# Copies named in an earlier sources file are preferred, so that the names cited elsewhere stay
# put: those of the file figures.sources.<HDF5_FROM>.*.txt (HDF5_FROM = 54dff0a by default, the
# copies deposited in Dataverse and cited by the paper; the newest sources file by mtime is not
# reliable, since the annex keeps the mtimes of its objects); among the other matches, the oldest
# file.
function cited_hdf5()
    from = get(ENV, "HDF5_FROM", "54dff0a")
    files = filter(n -> startswith(n, "$SCRIPT.sources.") && endswith(n, ".txt") &&
                        (isempty(from) || startswith(n, "$SCRIPT.sources.$from.")), readdir(IMAGES))
    isempty(files) && return Set{String}()
    newest = argmax(n -> mtime(joinpath(IMAGES, n)), files)
    return Set(m.captures[1] for m in eachmatch(r"sha256 \w+  (\S+\.h5)", read(joinpath(IMAGES, newest), String)))
end
const CITED_HDF5 = cited_hdf5()

function existing_hdf5(stem, source, masses)
    paths = [joinpath(DATA, n) for n in readdir(DATA) if startswith(n, stem) && endswith(n, ".h5")]
    for path in sort(paths; by = p -> (basename(p) in CITED_HDF5 ? 0 : 1, mtime(p)))
        same = h5open(path, "r") do io
            read(HDF5.attributes(io)["source"]) == source && read(io["masses"]) == masses
        end
        same && return path
    end
    return nothing
end
for r in rows
    rec = Serialization.deserialize(joinpath(DATA, r.density))
    f = rec.density
    k = 2^DEPTH; s = 2^(DEPTH - SHOW); sys = ulam_system(r.map)
    stem = "$SCRIPT.density_$(r.map)_a$(replace(r.a, "//" => "-"))_d$DEPTH."
    h5 = existing_hdf5(stem, r.density, f)
    if h5 === nothing
        h5 = joinpath(DATA, "$stem$commit.$host.h5")
        write_density_hdf5(h5, rec; source = r.density,
            domain = [inf(sys.domain[1]), sup(sys.domain[1]), inf(sys.domain[2]), sup(sys.domain[2])])
    end
    hdf5[r] = h5
    M = reshape(f, k, k)                                  # M[iy, ix], grid order
    marg[r] = x_marginal(f, k, (sup(sys.domain[1]) - inf(sys.domain[1])) / k)
    B = [sum(@view M[(j-1)*s+1:j*s, (i-1)*s+1:i*s]) for j in 1:2^SHOW, i in 1:2^SHOW]
    side = sup(sys.domain[1]) - inf(sys.domain[1])
    B ./= (s * side / k)^2                                # mass over block area: the density
    xs = range(inf(sys.domain[1]), sup(sys.domain[1]); length = 2^SHOW)
    ys = range(inf(sys.domain[2]), sup(sys.domain[2]); length = 2^SHOW)
    fig = Figure(size = (520, 460))
    ax = Axis(fig[1, 1]; aspect = DataAspect(), xlabel = "x", ylabel = "y",
              title = "$(name(r.map)), ξ = $(xi(r.a)), depth $DEPTH: ‖f_ξ − f̃‖₁ ≤ $(tex(up3(r.errs); unicode = true))")
    hm = heatmap!(ax, xs, ys, permutedims(B); colormap = :viridis, rasterize = 2)
    Colorbar(fig[1, 2], hm)
    save(out("density_$(r.map)_a$(replace(r.a, "//" => "-"))_d$DEPTH", "pdf"), fig)
end

# The x-marginals of the Lorenz maps, one figure per map with a curve per noise size; integrating
# out y is an L¹ contraction, so the bound of each density bounds its marginal too.
for m in unique(r.map for r in rows if r.map != "henon")
    rs = filter(r -> r.map == m, rows)
    sys = ulam_system(m)
    k = 2^DEPTH
    xs = range(inf(sys.domain[1]), sup(sys.domain[1]); length = k + 1)
    xc = (xs[1:end-1] .+ xs[2:end]) ./ 2
    fig = Figure(size = (860, 360))
    ax = Axis(fig[1, 1]; xlabel = "x", ylabel = "density of the x-marginal", title = "$(name(m)), depth $DEPTH")
    for r in rs
        lines!(ax, xc, marg[r]; label = "ξ = $(xi(r.a)): ‖·‖₁ error ≤ $(tex(up3(r.errs); unicode = true))")
    end
    Legend(fig[1, 2], ax; labelsize = 10, framevisible = false)
    save(out("marginal_$(m)_d$DEPTH", "pdf"), fig)
    # the marginal at the smallest noise size alone, the size of a density panel, for the
    # overview figure of the introduction that puts it beside that density
    r = argmin(r -> parse(Rational{Int}, replace(r.a, "//" => "/")), rs)
    fig = Figure(size = (520, 460))
    ax = Axis(fig[1, 1]; xlabel = "x", ylabel = "density of the x-marginal",
              title = "$(name(m)), ξ = $(xi(r.a)): ‖·‖₁ error ≤ $(tex(up3(r.errs); unicode = true))")
    lines!(ax, xc, marg[r]; color = :black)
    save(out("marginal_$(m)_a$(replace(r.a, "//" => "-"))_d$DEPTH", "pdf"), fig)
end

open(out("table_bounds", "tex"), "w") do io
    println(io, "% $SCRIPT.jl of RigorousUlam2D.jl at $commit on $host; every number an upper bound written upward")
    println(io, "% (CertifiedRecords.fmt_upper, 3 digits); sources in $(basename(out("sources", "txt")))")
    println(io, "\\begin{tabular}{llrrrrrrr}")
    println(io, "map & \$\\xi\$ & coarse depth & \$\\Sigma_V\$ & \$\\Sigma\$ & \$\\varepsilon_1\$ & \$\\varepsilon_2\$ & direct & swapped \\\\ \\hline")
    for r in rows
        @printf(io, "%s & \$%s\$ & %s & \$%s\$ & \$%s\$ & \$%s\$ & \$%s\$ & \$%s\$ & \$%s\$ \\\\\n", name(r.map),
                xi(r.a), r.coarse, tex(up3(r.SigmaV)), tex(up3(r.Sigma)), tex(up3(r.eps1)),
                tex(up3(r.eps2)), tex(up3(r.err)), tex(up3(r.errs)))
    end
    println(io, "\\end{tabular}")
end

open(out("sources", "txt"), "w") do io
    for f in sources
        println(io, "== ", f); foreach(l -> println(io, l), first(read_record(joinpath(DATA, f))))
    end
    for r in rows
        println(io, "$(r.map) a = $(r.a): density $(r.density), powers $(r.powers)")
        println(io, "    sha256 $(file_sha256(joinpath(DATA, r.density)))  $(r.density)")
        println(io, "    sha256 $(file_sha256(hdf5[r]))  $(basename(hdf5[r]))")
    end
end
println("wrote $(length(rows)) figures, the table and the sources in $IMAGES")
