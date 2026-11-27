# What the three worked examples of docs/literate share: the depths of a mode (`example_settings`),
# where its output goes, and the table and the figure of a density.  Included by each example.

using RigorousUlam2D
using CertifiedRecords: fmt_upper, Upper
using LinearAlgebra: BLAS


"""
    example_settings(script; mode = get(ENV, "MODE", "check")) -> NamedTuple

The parameters of a worked example run in one of three modes:

- `"check"`: from scratch at small depths (the tree from depth 2 to 7, read at level 9 with 12
  bisections of the fibre integrals, the column sweep at depth 6 for 40 powers, the bounds at
  depth 7), which runs in minutes on any machine; then the table recomputed from the records, as
  in `"records"`, where the records are on this machine.
- `"records"`: no new run; every row of Table 1 for the map recomputed from the records the paper
  read (`PAPER_RECORDS`), one evaluation of each step on the grid it ended on, and compared with
  the recorded bounds.
- `"full"`: from scratch at the paper's depths (the tree from depth 4 to 12, refined on disk to 12
  and read at level 14 with 30 bisections, the sweeps at the coarse depth of the table, 150
  powers, the bounds at depth 12).  A configuration already on record for this commit and host is
  not run again, so a run that stopped continues where it stopped.  Refused on hosts other than
  ibis and albatross unless `FORCE=1`.

Output goes to `~/ResearchData/RigorousUlam2D_Data` (`data/`, the records and blocks the drivers of
RigorousUlam2D write) in `"full"` mode, and to its `bulk/<script>.check.<host>/` in `"check"` mode,
which is regenerable.  `DATAROOT` overrides the directory.
"""
function example_settings(script::AbstractString; mode::AbstractString = get(ENV, "MODE", "check"))
    mode in ("check", "records", "full") || throw(ArgumentError("MODE must be check, records or full, not $mode"))
    host = gethostname()
    base = expanduser("~/ResearchData/RigorousUlam2D_Data")
    if mode == "full" && !(host in ("ibis", "albatross")) && get(ENV, "FORCE", "0") != "1"
        error("MODE=full runs on ibis or albatross; set FORCE=1 to run on $host")
    end
    root = get(ENV, "DATAROOT", mode == "check" ? joinpath(base, "bulk", "$script.check.$host") : base)
    records = joinpath(base, "data")
    small = mode == "check"
    return (; mode, script, host, root, datadir = joinpath(root, "data"), records,
            d0 = small ? 2 : 4, d1 = small ? 7 : 12, store = small ? 7 : 12, level = small ? 9 : 14,
            bisect = small ? 12 : 30, M = small ? 40 : 150, fine = small ? 7 : 12,
            coarse = small ? (_ -> 6) : (e -> e.coarse))
end

"Print the parameters of `example_settings` and the thread counts, before any result."
function print_settings(S)
    println("$(S.script): mode $(S.mode) on $(S.host), $(Threads.nthreads()) Julia threads, ",
            "$(BLAS.get_num_threads()) BLAS threads")
    println("  output in $(S.datadir); records read from $(S.records)")
    S.mode == "records" ||
        println("  tree depths $(S.d0)..$(S.d1), stored to $(S.store), read at level $(S.level), ",
                "$(S.bisect) bisections; $(S.M) powers; bounds at depth $(S.fine)")
end

"""
    table_line(name, a, coarse, ΣV, Σ, ε1, ε2, err, err_swapped) -> String

One row of Table 1 as plain text: the noise size `ξ = 2a`, the coarse depth, and every bound
written upward at three digits (`fmt_upper`).
"""
function table_line(name, a::Rational, coarse, ΣV, Σ, ε1, ε2, err, err_swapped)
    up(x) = isfinite(x) ? fmt_upper(Upper(x); digits = 3) : "∞"
    return "$name  ξ = $(xi_string(a))  coarse $coarse  Σ_V ≤ $(up(ΣV))  Σ ≤ $(up(Σ))  " *
           "ε₁ ≤ $(up(ε1))  ε₂ ≤ $(up(ε2))  ‖f_ξ − f̃‖₁ ≤ $(up(err))  swapped ≤ $(up(err_swapped))"
end

"The side `ξ = 2a` of the noise square, as `\"p/q\"`, from the half-width `a`."
xi_string(a::Rational) = (ξ = 2a; "$(numerator(ξ))/$(denominator(ξ))")

"""
    density_matrix(masses, k, h) -> Matrix

The computed density `f̃` as a `k × k` matrix `D[ix, iy]` (column `ix` along `x`), the cell masses
in grid order divided by the cell area `h²`, for a heatmap.  In floating point: for display.
"""
density_matrix(m::AbstractVector{Float64}, k::Int, h::Float64) = permutedims(reshape(m, k, k)) ./ (h * h)
