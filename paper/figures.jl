# The certified numbers on record for RigorousUlam2D.tex, and figures of the computed densities.
#
# Reads, from ~/ResearchData/RigorousUlam2D_Data/data on this host, the latest records of
# bench/stationary_density.jl, bench/l1_mixing.jl and bench/l1_transfer.jl (CertifiedRecords), and
# writes into ~/ResearchData/RigorousUlam2D_Data/images:
#
#   figures.density_<map>_a<p>-<q>_d<depth>.<commit>.<host>.pdf
#       the computed stationary density f̃ at depth DEPTH (12) of each (map, a) whose L¹ bound at
#       that depth is finite, its cell masses summed over blocks of 2^(DEPTH − SHOW) × 2^(DEPTH − SHOW)
#       cells for display (SHOW = 9: 512 × 512 pixels) and divided by the block area; the title
#       gives the bound ‖f_ξ − f̃‖₁ ≤ b, b the smallest err_swapped on record for that depth over
#       the coarse depths, written upward at 3 digits (fmt_upper);
#   figures.table_bounds.<commit>.<host>.tex
#       a LaTeX tabular, one row per (map, a): the coarse depth of the best bound, Σ_V of that
#       sweep, the transfer's Σ, ε₁ and ε₂ of f̃ at DEPTH, and the bounds at DEPTH by the direct
#       and the swapped route; every number an upper bound written upward at 3 digits;
#   figures.sources.<commit>.<host>.txt
#       the record files every number and figure was read from, with their headers.
#
# The density in a figure is the computed f̃ (cell masses / cell area), not the stationary density
# itself; the bound in the title is the certified distance between the two.  Rendering is headless.
#
# Env: DEPTH (12), SHOW (9).
#
#   julia --project=paper paper/figures.jl

ENV["GKSwstype"] = "100"
using RigorousUlam2D, CertifiedRecords, CairoMakie, Printf
import Serialization

const SCRIPT = splitext(basename(@__FILE__))[1]
const ROOT = expanduser(get(ENV, "DATAROOT", "~/ResearchData/RigorousUlam2D_Data"))
const DATA = joinpath(ROOT, "data")
const IMAGES = joinpath(ROOT, "images")
const DEPTH = parse(Int, get(ENV, "DEPTH", "12"))
const SHOW = parse(Int, get(ENV, "SHOW", "9"))

commit = try readchomp(`git -C $(pkgdir(RigorousUlam2D)) rev-parse --short HEAD`) catch; "unknown" end
host = gethostname()
out(name, ext) = joinpath(IMAGES, "$SCRIPT.$name.$commit.$host.$ext")
up3(x) = fmt_upper(Upper(x); digits = 3)
tex(s) = (m = match(r"^(-?[0-9.]+)e([+-][0-9]+)$", s); m === nothing ? s :
          (e = parse(Int, m.captures[2]); e == 0 ? m.captures[1] : "$(m.captures[1])\\times10^{$e}"))
rd(s) = s == "Inf" ? Inf : Float64(read_endpoint(s, 53), RoundUp)

densrec = latest_record(DATA, "stationary_density")
mixrec = latest_record(DATA, "l1_mixing")
trrec = latest_record(DATA, "l1_transfer")
dens, mix, tr = last(read_record(densrec)), last(read_record(mixrec)), last(read_record(trrec))
mkpath(IMAGES)

rows = []
for m in ("lorenz", "henon"), a in ("77//256", "51//256", "51//512", "13//256")
    cand = filter(r -> r["map"] == m && r["a"] == a && r["fine_depth"] == string(DEPTH) && r["err_swapped"] != "Inf", tr)
    isempty(cand) && continue
    best = cand[argmin([rd(r["err_swapped"]) for r in cand])]
    d = only(filter(r -> r["map"] == m && r["a"] == a && r["depth"] == string(DEPTH), dens))
    sw = only(filter(r -> r["map"] == m && r["a"] == a && r["depth"] == best["coarse_depth"], mix))
    push!(rows, (; map = m, a, coarse = best["coarse_depth"], SigmaV = rd(sw["Sigma_V"]), Sigma = rd(best["Sigma"]),
                 eps1 = rd(d["eps1"]), eps2 = rd(d["eps2"]), err = rd(best["err"]), errs = rd(best["err_swapped"]),
                 density = d["density_file"], powers = sw["powers_file"]))
end

name(m) = m == "henon" ? "Hénon" : "Lorenz"
for r in rows
    f = Serialization.deserialize(joinpath(DATA, r.density)).density
    k = 2^DEPTH; s = 2^(DEPTH - SHOW); sys = ulam_system(r.map)
    M = reshape(f, k, k)                                  # M[iy, ix], grid order
    B = [sum(@view M[(j-1)*s+1:j*s, (i-1)*s+1:i*s]) for j in 1:2^SHOW, i in 1:2^SHOW]
    side = sup(sys.domain[1]) - inf(sys.domain[1])
    B ./= (s * side / k)^2                                # mass over block area: the density
    xs = range(inf(sys.domain[1]), sup(sys.domain[1]); length = 2^SHOW)
    ys = range(inf(sys.domain[2]), sup(sys.domain[2]); length = 2^SHOW)
    fig = Figure(size = (520, 460))
    ax = Axis(fig[1, 1]; aspect = DataAspect(), xlabel = "x", ylabel = "y",
              title = "$(name(r.map)), a = $(replace(r.a, "//" => "/")), depth $DEPTH: ‖f_ξ − f̃‖₁ ≤ $(up3(r.errs))")
    hm = heatmap!(ax, xs, ys, permutedims(B); colormap = :viridis)
    Colorbar(fig[1, 2], hm)
    save(out("density_$(r.map)_a$(replace(r.a, "//" => "-"))_d$DEPTH", "pdf"), fig)
end

open(out("table_bounds", "tex"), "w") do io
    println(io, "% $SCRIPT.jl of RigorousUlam2D.jl at $commit on $host; every number an upper bound written upward")
    println(io, "% (CertifiedRecords.fmt_upper, 3 digits); sources in $(basename(out("sources", "txt")))")
    println(io, "\\begin{tabular}{llrrrrrrr}")
    println(io, "map & \$a\$ & coarse depth & \$\\Sigma_V\$ & \$\\Sigma\$ & \$\\varepsilon_1\$ & \$\\varepsilon_2\$ & direct & swapped \\\\ \\hline")
    for r in rows
        @printf(io, "%s & \$%s\$ & %s & \$%s\$ & \$%s\$ & \$%s\$ & \$%s\$ & \$%s\$ & \$%s\$ \\\\\n", name(r.map),
                replace(r.a, "//" => "/"), r.coarse, tex(up3(r.SigmaV)), tex(up3(r.Sigma)), tex(up3(r.eps1)),
                tex(up3(r.eps2)), tex(up3(r.err)), tex(up3(r.errs)))
    end
    println(io, "\\end{tabular}")
end

open(out("sources", "txt"), "w") do io
    for f in (densrec, mixrec, trrec)
        println(io, "== ", basename(f)); foreach(l -> println(io, l), first(read_record(f)))
    end
    for r in rows
        println(io, "$(r.map) a = $(r.a): density $(r.density), powers $(r.powers)")
    end
end
println("wrote $(length(rows)) figures, the table and the sources in $IMAGES")
