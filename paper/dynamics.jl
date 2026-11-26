# Pictures of the deterministic maps of the paper's examples (Section 6 of RigorousUlam2D.tex).
# Illustrations only: no number in them is certified, and none is quoted in the paper.
#
# Writes into ~/ResearchData/RigorousUlam2D_Data/images:
#
#   dynamics.henon_attractor.<commit>.<host>.pdf
#       N (200000) points of an orbit of the Hénon map of henon_system, after a transient of
#       1000 iterates, on the torus [-1, 1]²;
#   dynamics.<map>_map.<commit>.<host>.pdf, <map> in MAPS ("lorenz,mlorenz_rho18-5")
#       left, the graph of the first component T(x) on [-1/2, 1/2]; right, the image F(Q) of the
#       torus Q = [-1/2, 1/2]², drawn as the images of the horizontal segments y = const for
#       S (41) values of y, each sampled at 4000 points of x.
#
# The maps are those of ulam_system, evaluated at degenerate intervals and read at the midpoint,
# and wrapped to the torus with wrap_to_torus.  Rendering is headless.
#
# Env: N, S, MAPS.
#
#   julia --project=paper paper/dynamics.jl

ENV["GKSwstype"] = "100"
using RigorousUlam2D, CairoMakie
using IntervalArithmetic: interval, inf, sup, mid

const SCRIPT = splitext(basename(@__FILE__))[1]
const ROOT = expanduser(get(ENV, "DATAROOT", "~/ResearchData/RigorousUlam2D_Data"))
const IMAGES = joinpath(ROOT, "images")
const N = parse(Int, get(ENV, "N", "200000"))
const S = parse(Int, get(ENV, "S", "41"))
const MAPS = split(get(ENV, "MAPS", "lorenz,mlorenz_rho18-5"), ",")

commit = try readchomp(`git -C $(pkgdir(RigorousUlam2D)) rev-parse --short HEAD`) catch; "unknown" end
host = gethostname()
out(name) = joinpath(IMAGES, "$SCRIPT.$name.$commit.$host.pdf")
mkpath(IMAGES)

# checked: GCSineMapsExperiments.step! -- iterates a random sine-unit model; here the map of
# ulam_system at a point, read at the midpoint and wrapped to its torus with wrap_to_torus.
"The map of `sys` at the point (x, y), at the midpoint, wrapped to the torus `sys.domain`."
function image_point(sys, x, y)
    u, v = sys.F(interval(x), interval(y))
    return (wrap_to_torus(mid(u), inf(sys.domain[1]), sup(sys.domain[1])),
            wrap_to_torus(mid(v), inf(sys.domain[2]), sup(sys.domain[2])))
end

# Hénon attractor
let sys = ulam_system("henon"), p = (0.1, 0.1)
    for _ in 1:1000
        p = image_point(sys, p...)
    end
    xs, ys = Vector{Float64}(undef, N), Vector{Float64}(undef, N)
    for i in 1:N
        p = image_point(sys, p...); xs[i], ys[i] = p
    end
    fig = Figure(size = (480, 460))
    ax = Axis(fig[1, 1]; aspect = DataAspect(), xlabel = "x", ylabel = "y", limits = (-1, 1, -1, 1))
    scatter!(ax, xs, ys; markersize = 0.6, color = :black, rasterize = 3)
    save(out("henon_attractor"), fig)
end

# Lorenz maps: the one-dimensional map and the image of the square
for m in MAPS
    sys = ulam_system(String(m))
    lo, hi = inf(sys.domain[1]), sup(sys.domain[1])
    xs = [x for x in range(lo, hi; length = 4001) if x != 0]   # the discontinuity
    fig = Figure(size = (900, 430))
    ax1 = Axis(fig[1, 1]; aspect = DataAspect(), xlabel = "x", ylabel = "T(x)", limits = (lo, hi, lo, hi))
    lines!(ax1, [lo, hi], [lo, hi]; color = :gray, linestyle = :dash)
    for branch in (filter(<(0), xs), filter(>(0), xs))
        lines!(ax1, branch, [image_point(sys, x, 0.0)[1] for x in branch]; color = :black)
    end
    ax2 = Axis(fig[1, 2]; aspect = DataAspect(), xlabel = "x", ylabel = "y", limits = (lo, hi, lo, hi))
    for y in range(inf(sys.domain[2]), sup(sys.domain[2]); length = S)
        P = [image_point(sys, x, y) for x in xs]
        scatter!(ax2, first.(P), last.(P); markersize = 0.8, color = :black, rasterize = 3)
    end
    save(out("$(m)_map"), fig)
end
println("wrote the pictures of the dynamics in $IMAGES")
