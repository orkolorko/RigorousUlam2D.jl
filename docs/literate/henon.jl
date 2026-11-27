# # The Hénon map
#
# This example regenerates the rows of Table 1 of the paper for the random perturbation of the
# Hénon map of Section 6.1,
#
# ```math
# H(X, Y) = (1 - AX^2 + Y,\; BX), \qquad A = 7/5,\ B = 3/10,
# ```
#
# in the coordinates ``X = 3x/2``, ``Y = y/2``, which send ``[-1, 1]^2`` to a box containing the
# attractor: ``F(x, y) = (\tfrac23(1 - A(\tfrac32 x)^2 + \tfrac12 y),\ \tfrac9{10} x)`` on the torus
# ``[-1, 1]^2``, with additive uniform noise on a square of side ``ξ``.  The first component is
# increasing in ``y`` and the second depends on ``x`` alone, which is what `fibred_area` uses
# (`henon_system`).  The table has two rows for this map, ``ξ = 77/128`` from the coarse depth 9 and
# ``ξ = 51/128`` from depth 8; at ``ξ = 51/256`` and ``ξ = 13/128`` the recursion did not reach
# ``F_k < 1`` from the coarse grids that were run.
#
# Run as `MODE=check julia --project=paper docs/literate/henon.jl` (the default: small depths, then the
# table from the records where they are on this machine), `MODE=records` (the table from the
# records only) or `MODE=full` (the paper's depths, on ibis or albatross); see `example_settings`.

using RigorousUlam2D
include(joinpath(@__DIR__, "settings.jl"))
using CertifiedRecords: latest_record, read_record, read_endpoint, write_endpoint, fmt_upper, Upper
using CairoMakie
import Serialization

const SCRIPT = "henon"
const MAP = "henon"
S = example_settings(SCRIPT)
print_settings(S)
commit = try readchomp(`git -C $(pkgdir(RigorousUlam2D)) rev-parse --short HEAD`) catch; "unknown" end
script = "docs/literate/$SCRIPT.jl"
rows = filter(e -> e.map == MAP, PAPER_RECORDS)
for e in rows
    println("  ξ = $(xi_string(e.a)), coarse depth $(S.coarse(e))")
end

# ## The deterministic block
#
# The Ulam matrix of the map on the grid of depth ``d``, `L[i, j] = |I_j ∩ F⁻¹(I_i)| / |I_j|` for the
# ``4^d`` cells of side ``δ = h``, is enclosed entrywise, ``L_{lo} ≤ L ≤ L_{lo} + U``, by refining a
# tree of cells: a cell whose image enclosure meets one target cell only sends all its mass there,
# exactly; the cells that meet several are subdivided, and at the last level the fraction of each
# going to each target is enclosed by integrating the inverse of the fibre map (`fibred_area`).
# `run_blocks` builds the tree from depth `d0` to `d1`, refines it on disk to depth `store` and
# reads the blocks of every depth the run needs at level `level`.

blk(d) = block_file(S.datadir, MAP, d; level = S.level, bisect = S.bisect, d1 = S.d1)
depths = minimum(S.coarse(e) for e in rows):S.fine
if S.mode != "records"
    run_blocks(MAP; root = S.root, d0 = S.d0, d1 = S.d1, store_level = S.store, level = S.level,
               bisect = S.bisect, depths, script)
end

# ## The noisy operator, the computed density and its residuals
#
# The Ulam discretization of the annealed operator is ``P_δ = π_δ N_ξ π_δ L π_δ``, with ``N_ξ`` the
# convolution with the uniform density ``ρ_ξ`` on the square ``[-ξ/2, ξ/2]^2``; it is applied
# matrix-free as ``\mathrm{fl}(N\,\mathrm{fl}(\hat L y))`` with ``\hat L = L_{lo} + U/2``, and
# eq:rounding bounds ``\|\mathrm{fl}(N\,\mathrm{fl}(\hat L y)) - P_δ y\|_1`` by ``η_1\|y\|_1 + η_A``
# for every ``L`` of the enclosure.  The density ``\tilde f`` is the power iteration, and
# `run_stationary_density` certifies its residuals ``ε_1 ≥ \|P_δ \tilde f - \tilde f\|_1`` and
# ``ε_2 ≥ |\sum \tilde f - 1|``, together with the coordinate variations ``\mathrm{var}_1(\tilde f)``,
# ``\mathrm{var}_2(\tilde f)``.  It runs on the coarse grid, whose density is the reference vector of
# the column sweep below, and on the fine grid of the bound.

if S.mode != "records"
    for e in rows, d in unique((S.coarse(e), S.fine))
        run_stationary_density(MAP, d, e.a; datadir = S.datadir, script, blockfile = blk(d))
    end
end

# ## The column sweep on the coarse grid
#
# On the coarse grid of depth ``d_c`` every column is iterated: for ``v`` of sum zero,
# ``v = \sum_i v_i (e_i - u)``, so ``C_j = \max_i \|P_{δ_0}^j e_i - u\|_1`` plus the propagated
# rounding bounds ``\|P_{δ_0}^j|_{V}\|_{L^1}``.  The ``C_j`` are refined by submultiplicativity, and
# ``Σ_V`` bounds ``\|(I - P_{δ_0})^{-1}|_V\|_{L^1}`` from them.

if S.mode != "records"
    for e in rows
        run_column_sweep(MAP, S.coarse(e), e.a; datadir = S.datadir, M = S.M, script, blockfile = blk(S.coarse(e)))
    end
end

# ## The coarse-fine recursion and the bounds of Proposition l1error
#
# With ``θ_δ = (δ/2)(\mathrm{var}_1(ρ_ξ) + \mathrm{var}_2(ρ_ξ)) = 2δ/ξ`` (Theorem C), the recursion
# eq:coarsefine with ``θ_1 = θ_2 = θ_{δ_0}`` turns the ``C_j`` of the coarse grid into ``F_k ≥
# \|Λ^k|_V\|_{L^1}`` for the Ulam operator ``Λ`` of every finer grid and for ``L_ξ`` itself; once some
# ``F_k < 1``, ``Σ`` bounds ``\|(I - Λ)^{-1}|_V\|_{L^1}``.  On the fine grid of side ``δ``,
# ``\|f_ξ - \tilde f\|_1 ≤ 3Σθ_δ + Σε_1/(1 - ε_2) + ε_2`` (direct) and
# ``Σ(2θ_δ + ε_1/(1 - ε_2)) + ε_2`` (swapped).

if S.mode != "records"
    for e in rows
        run_transfer(MAP, S.coarse(e), e.a, [S.fine]; datadir = S.datadir, script)
    end
end

# ## The table
#
# The rows of this run, read back from the records it wrote; every bound is written upward at
# three digits.

rd(s) = s == "Inf" ? Inf : Float64(read_endpoint(s, 53), RoundUp)
if S.mode != "records"
    mix = last(read_record(latest_record(S.datadir, "l1_mixing")))
    tr = last(read_record(latest_record(S.datadir, "l1_transfer")))
    dens = last(read_record(latest_record(S.datadir, "stationary_density")))
    for e in rows
        c, k = string(S.coarse(e)), string(S.fine)
        m = only(filter(r -> r["map"] == MAP && r["depth"] == c && r["a"] == string(e.a), mix))
        t = only(filter(r -> r["map"] == MAP && r["coarse_depth"] == c && r["a"] == string(e.a) && r["fine_depth"] == k, tr))
        d = only(filter(r -> r["map"] == MAP && r["depth"] == k && r["a"] == string(e.a), dens))
        println(table_line(MAP, e.a, c, rd(m["Sigma_V"]), rd(t["Sigma"]), rd(d["eps1"]), rd(d["eps2"]),
                           rd(t["err"]), rd(t["err_swapped"])))
    end
end

# ## The table from the records of the paper
#
# Where the records the paper read are on this machine (ibis holds them), every row is recomputed
# from them by one evaluation of each step on the grid the record ended on (`recertify_from_records`):
# the refined ``C_j``, the recursion, ``Σ``, and the bounds at depth 12 from the recorded ``ε_1``,
# ``ε_2``; each is compared with the recorded value, bit for bit.

have(f) = isfile(joinpath(S.records, f))
for e in rows
    (have(e.powers_record) && have(e.density_record)) || (println("records of ξ = $(xi_string(e.a)) not on this machine"); continue)
    rc = recertify_from_records(e; datadir = S.records)
    same = all(r -> (r["Sigma"], r["err"], r["err_swapped"]) == (write_endpoint(rc.Σ.grid)[1], write_endpoint(rc.bounds.err)[1],
                     write_endpoint(rc.bounds.err_swapped)[1]), rc.recorded.transfer) &&
           [write_endpoint(x)[1] for x in rc.C_refined] == rc.recorded.C_refined
    m = only(filter(r -> r["powers_file"] == e.powers_record,
                    [r for f in readdir(S.records) if occursin(r"^l1_mixing\.[0-9a-f]+\.\w+\.tsv$", f) && have(f)
                       for r in last(read_record(joinpath(S.records, f)))]))
    println(table_line(MAP, e.a, e.coarse, rd(m["Sigma_V"]), rc.Σ.grid, rc.ε1, rc.ε2, rc.bounds.err, rc.bounds.err_swapped),
            same ? "  (as recorded)" : "  (DIFFERS from the records)")
end

# ## The computed densities
#
# The density ``\tilde f`` of each noise size on the fine grid of this run, the cell masses
# divided by the cell area; the title gives the certified bound on ``\|f_ξ - \tilde f\|_1``.  In
# `records` mode nothing is drawn: the figures of the paper are drawn by `paper/figures.jl` of
# RigorousUlam2D from the depth-12 densities.

if S.mode != "records"
    images = joinpath(S.root, "images"); mkpath(images)
    tr = last(read_record(latest_record(S.datadir, "l1_transfer")))
    dens = last(read_record(latest_record(S.datadir, "stationary_density")))
    for e in rows
        d = only(filter(r -> r["map"] == MAP && r["depth"] == string(S.fine) && r["a"] == string(e.a), dens))
        t = only(filter(r -> r["map"] == MAP && r["coarse_depth"] == string(S.coarse(e)) && r["a"] == string(e.a) &&
                             r["fine_depth"] == string(S.fine), tr))
        g = grid_constants(ulam_system(MAP), S.fine, e.a)
        D = density_matrix(Serialization.deserialize(joinpath(S.datadir, d["density_file"])).density, g.k, g.h)
        x0, x1 = -sqrt(g.area) / 2, sqrt(g.area) / 2
        fig = Figure(size = (520, 480))
        ax = Axis(fig[1, 1]; aspect = DataAspect(), title = "ξ = $(xi_string(e.a)), ‖f_ξ − f̃‖₁ ≤ $(fmt_upper(Upper(rd(t["err_swapped"])); digits = 3))")
        hm = heatmap!(ax, range(x0, x1; length = g.k), range(x0, x1; length = g.k), D; rasterize = true)
        Colorbar(fig[1, 2], hm)
        file = joinpath(images, "$SCRIPT.density_$(MAP)_a$(numerator(e.a))-$(denominator(e.a))_d$(S.fine).$commit.$(S.host).pdf")
        save(file, fig)
        println("figure: $file")
    end
end
