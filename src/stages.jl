"""
The four runs behind the paper's numbers, as functions: each runs one configuration, writes its
CertifiedRecords record into the data directory under the name of the stage (`tree_deepen`,
`stationary_density`, `l1_mixing`, `l1_transfer`), this package's commit and the host, and returns
what it computed, or `nothing` when the configuration is already on record for this commit.
The drivers in `bench/` read their parameters from the environment and call these; a worked example
calls them with its own `script`, which the header of every record names.

Each record's header (`run_header`) names the script, the host, the time, the commit and the Julia
and BLAS thread counts; each row names, in its column `environment`, the copy of the Manifest of
the run (`save_environment`), and carries the wall-clock time of the run in a column `seconds`.
"""

export run_blocks, run_stationary_density, run_column_sweep, run_transfer

using CertifiedRecords: run_header, save_environment, recorded_keys, latest_record, read_record, read_endpoint
using Printf: @printf
using Dates: Dates

_amp(a::Rational) = "a$(numerator(a))-$(denominator(a))"
_gitrev(dir) = try chomp(read(`git -C $dir rev-parse --short HEAD`, String)) catch; "?" end
_gitdirty(dir) = try !isempty(read(`git -C $dir status --porcelain --untracked-files=no -- src bench Project.toml Manifest.toml`, String)) catch; true end
_colmax(A) = maximum(vec(sum(A; dims = 1)); init = 0.0)
_rowmax(A) = maximum(vec(sum(A; dims = 2)); init = 0.0)

"""
    run_blocks(map; root, d0 = 4, d1 = 12, store_level = 12, level = 14, bisect = 30, xinv = true,
               depths = d0:d1, script = "bench/tree_deepen.jl") -> Vector of paths, or nothing

The enclosures `L_lo ≤ L ≤ L_lo + U` of the Ulam matrices of the map `map` (`ulam_system`) on the
uniform grids of the depths `depths`, read off an `UlamTree`: the tree is refined on disk to the
depth `store_level` (its subtrees in `root/bulk/tree_deepen.<map>_d<d0>-<d1>.store.<host>/`,
resumed if the run is repeated), then read at the depth `level` by `ulam_matrices`, which
quadrisects the uncertain cells in memory and integrates the uncertain leaves through the map's
`fibred_area` (`bisect` bisections, the cut at the inverse of the x-only component when `xinv`).
Images are enclosed by the natural interval extension (`zeroth_order_enclosure`).

Writes, for each depth whose block is not already there, the block `block_file(root/data, map, d;
level, bisect, xinv, d1)`, a named tuple `(map, depth, store_level, store_maxlevel, tag, levels, d0,
d1, graph, L_lo, U, host, script, commits, header, environment, julia, threads, time)` with the
matrices indexed as `build_uniform_partition(domain, d)`, and a row of the record
`root/data/tree_deepen.<commit>.<host>.tsv` with the counts and the widths of `U` as floating-point
diagnostics (not bounds), `read_seconds` and the wall clock `seconds`.  Refuses to run when this
package has uncommitted changes in `src`, `bench` or the project files.
"""
function run_blocks(map::AbstractString; root::AbstractString, d0::Int = 4, d1::Int = 12,
        store_level::Int = 12, level::Int = 14, bisect::Int = 30, xinv::Bool = true,
        depths::AbstractUnitRange{Int} = d0:d1, script::AbstractString = "bench/tree_deepen.jl")
    host = gethostname()
    pkg = pkgdir(RigorousUlam2D)
    commits = (; RigorousUlam2D = _gitrev(pkg), dirty = _gitdirty(pkg))
    commits.dirty && error("$pkg has uncommitted changes in src, bench or the project files; commit before running")
    data = joinpath(root, "data")
    sys = ulam_system(map; bisect, xinv)
    tag = "level$(level)_area$(bisect)$(xinv ? "_xinv" : "")_t$(d1)"
    outfile(d) = block_file(data, map, d; level, bisect, xinv, d1, host)
    record = record_file(data, "tree_deepen", RigorousUlam2D; host)
    header = run_header(RigorousUlam2D, script)
    environment = save_environment(record)
    todo = filter(d -> !isfile(outfile(d)), collect(depths))
    store = joinpath(root, "bulk", "tree_deepen.$(map)_d$(d0)-$(d1).store.$host")

    println("tree_deepen on $host, $(Dates.now()), julia $VERSION, threads $(Threads.nthreads()), commits $commits")
    println("map $map, domain $(sys.domain), enclosure zeroth_order_enclosure, bisect $bisect, xinv $xinv, ",
            "d0 $d0, d1 $d1, store to $store_level, read at $level, depths $depths (to do: $todo), store $store")
    isempty(todo) && (println("nothing to do"); return nothing)
    flush(stdout)

    mkpath(data); mkpath(dirname(store))
    t0 = time()
    tree = UlamTree(sys.F, sys.domain, d0, d1; enclosure_method = zeroth_order_enclosure, area = sys.area, store)
    @printf("store at level %d-%d, %d active, %d resolved (%.1f s)\n", tree.level, tree.maxlevel, tree.nactive, tree.nresolved, time() - t0)
    while tree.level < store_level
        t = @elapsed refine!(tree)
        @printf("store refined to level %d-%d in %.1f s, %d active, %d resolved\n", tree.level, tree.maxlevel, t, tree.nactive, tree.nresolved)
        flush(stdout)
    end

    tr = @elapsed mats = ulam_matrices(tree, minimum(todo):maximum(todo); level)
    files = String[]
    for m in mats
        m.depth in todo || continue
        lo, hi = extrema(vec(sum(m.L_lo; dims = 1))), extrema(vec(sum(m.L_lo .+ m.U; dims = 1)))
        @printf("%s %s depth %d  read %.1fs  nnz(L_lo) %d  nnz(U) %d  colsums L_lo %.12f..%.12f, L_lo+U %.12f..%.12f  ‖U‖₁ %.3e  ‖U‖_∞ %.3e\n",
            map, tag, m.depth, tr, nnz(m.L_lo), nnz(m.U), lo..., hi..., _colmax(m.U), _rowmax(m.U))
        Serialization.serialize(outfile(m.depth), (; map, depth = m.depth, store_level = tree.level,
            store_maxlevel = tree.maxlevel, tag, levels = level, d0, d1, graph = m.graph, L_lo = m.L_lo,
            U = m.U, host, script = splitext(basename(script))[1], commits, header, environment,
            julia = string(VERSION), threads = Threads.nthreads(), time = tr))
        # diagnostics, summed in floating point: not bounds
        open_record(record, header, ["map", "depth", "level", "bisect", "xinv", "d1", "nnz_L_lo", "nnz_U",
            "colsum_L_lo_min_fl", "colsum_L_hi_max_fl", "U_1_fl", "U_inf_fl", "read_seconds", "seconds", "file", "environment"])
        append_record(record, [map, m.depth, level, bisect, xinv, d1, nnz(m.L_lo), nnz(m.U), lo[1], hi[2],
            _colmax(m.U), _rowmax(m.U), round(tr; digits = 1), round(time() - t0; digits = 1), basename(outfile(m.depth)), environment])
        push!(files, outfile(m.depth))
        flush(stdout)
    end
    @printf("total %.1f s\n", time() - t0)
    return files
end

"""
    run_stationary_density(map, depth, a; datadir, tol = 1e-14, maxiter = 20_000,
                           script = "bench/stationary_density.jl")
        -> (; P, fp, res, cv, defect, record, density_file), or nothing

The computed stationary density of the noisy Ulam operator `P_h = π N_ξ π L π` of the map `map` at
the depth `depth` and the noise half-width `a` (`noisy_ulam_operator` of the block of `run_blocks`
on this host), with its a posteriori quantities: the density `f̃` is the power iteration
`ulam_fixed_point` on the masses; `ε₁ ≥ ‖P_h f̃ − f̃‖₁` and `ε₂ ≥ |Σ f̃ − 1|` (`fixed_point_residual`,
with the ℓ¹ rounding model `η₁` of `noisy_operator`, which covers the rounding of the products and
the enclosure of `L`); `var₁(f̃)`, `var₂(f̃)` (`coordinate_variations`); and the a priori defect
`3θ_h = (3h/2) var₁₂(ρ_ξ) ≥ ‖(L_ξ − P_h) f_ξ‖₁` (`defect_apriori`).

Writes the density (cell masses, grid order) to `datadir/stationary_density.<map>_d<depth>_a<p>-<q>.<commit>.<host>.jls`
and a row of `datadir/stationary_density.<commit>.<host>.tsv`, the bounds as exact doubles.
"""
function run_stationary_density(map::AbstractString, depth::Int, a::Rational; datadir::AbstractString,
        tol::Float64 = 1e-14, maxiter::Int = 20_000, script::AbstractString = "bench/stationary_density.jl")
    record = record_file(datadir, "stationary_density", RigorousUlam2D)
    (map, string(depth), string(a)) in recorded_keys(record, "map", "depth", "a") &&
        (println("$map depth $depth a = $a is in $(basename(record))"); return nothing)
    densfile = replace(record_file(datadir, "stationary_density.$(map)_d$(depth)_$(_amp(a))", RigorousUlam2D),
                       r"\.tsv$" => ".jls")
    blkfile = block_file(datadir, map, depth)
    t0 = time()
    P = noisy_ulam_operator(ulam_system(map), Serialization.deserialize(blkfile), depth, a)
    header = run_header(RigorousUlam2D, script)
    environment = save_environment(record)
    print(run_header(RigorousUlam2D, script; map, depth, a, tol, blockfile = basename(blkfile)))
    @printf("%s depth %d n = %d, a = %s, h = %.6g, var₁₂(ρ_ξ) = %.6g; η₁ = %.3e, η₁ᴬ = %.3e (%.1f s)\n",
        map, depth, P.n, a, P.h, P.var12ρ, P.info.η1, P.info.eta1A, time() - t0)

    tp = @elapsed fp = ulam_fixed_point(P.info.Amul, P.n; tol, maxiter)
    res = fixed_point_residual(P.info.Amul, fp.v, P.info.η1, P.info.eta1A)
    cv = coordinate_variations(fp.v, P.k, P.h)
    defect = defect_apriori(P.h, P.var12ρ)
    @printf("power iteration: %d iterations, last ℓ¹ change %.3e, converged %s (%.1f s)\n", fp.iterations, fp.diff, fp.converged, tp)
    @printf("ε₁ ≤ %.3e, ε₂ ≤ %.3e, var₁(f̃) ≤ %.6g, var₂(f̃) ≤ %.6g, (3h/2) var₁₂(ρ_ξ) = %.6g\n", res.ε1, res.ε2, cv.var1, cv.var2, defect)

    Serialization.serialize(densfile, (; map, depth, a, density = fp.v, header, environment))
    open_record(record, header, ["map", "depth", "a", "n", "iterations", "converged", "eps1", "eps2", "var1f", "var2f", "defect",
        "blockfile", "density_file", "seconds", "environment"])
    append_record(record, [map, depth, a, P.n, fp.iterations, fp.converged, write_endpoint(res.ε1)[1],
        write_endpoint(res.ε2)[1], write_endpoint(cv.var1)[1], write_endpoint(cv.var2)[1], write_endpoint(defect)[1],
        basename(blkfile), basename(densfile), round(time() - t0; digits = 1), environment])
    println("recorded in $(basename(record)), density in $(basename(densfile)) ($(round(time() - t0; digits = 1)) s)")
    return (; P, fp, res, cv, defect, record, density_file = densfile)
end

"""
    run_column_sweep(map, depth, a; datadir, M = 150, block = 8, cols = nothing,
                     script = "bench/l1_mixing.jl") -> (; C, sigma, n, h, var12ρ, record), or nothing

The ℓ¹ norms of the powers of the noisy Ulam operator `P_h` on the zero-sum vectors `V_h`, by
columns (`norms_of_powers_columns`): `C_j ≥ ‖P_h^j|_{V_h}‖₁` for `j ≤ M`, the maximum over the cells
`i` of `‖P_h^j e_i − u‖₁` plus the propagated rounding, every column iterated in floating point with
the ℓ¹ rounding model `η₁` of `noisy_operator` and `‖P_h‖₁ ≤ ‖L_lo + U‖₁`.  The operator is
`noisy_ulam_operator` of the block of `run_blocks` at this depth on this host.  `u`, a free choice
in the bound, is the density of the `stationary_density` record of the same configuration when there
is one, else the power iteration.  The bounds are refined by submultiplicativity and recorded by
`record_column_sweep`: `datadir/l1_mixing.<map>_d<depth>_a<p>-<q>.<commit>.<host>.tsv` (`j`, `C_j`,
refined `C_j`, as exact doubles) and a row of `datadir/l1_mixing.<commit>.<host>.tsv` (`Σ_grid`,
`Σ_V = 1 + Σ_grid`, and the a priori bound `Σ_V · 3θ_h`).

With `cols` (a range) only those columns are iterated and the record is
`datadir/l1_mixing.<map>_d<depth>_a<p>-<q>_cols<first>-<last>.<commit>.<host>.tsv`, the maximum over
`cols` for each `j`, with the SHA-256 of the block in its header and no summary row;
`bench/l1_mixing_merge.jl` takes the maximum of parts covering all the columns, which is the `C` of
the whole run bit for bit.
"""
function run_column_sweep(map::AbstractString, depth::Int, a::Rational; datadir::AbstractString,
        M::Int = 150, block::Int = 8, cols::Union{Nothing,AbstractUnitRange{Int}} = nothing,
        script::AbstractString = "bench/l1_mixing.jl")
    amp = _amp(a)
    summary = record_file(datadir, "l1_mixing", RigorousUlam2D)
    (map, string(depth), string(a)) in recorded_keys(summary, "map", "depth", "a") &&
        (println("$map depth $depth a = $a is in $(basename(summary))"); return nothing)
    part = cols === nothing ? nothing :
        record_file(datadir, "l1_mixing.$(map)_d$(depth)_$(amp)_cols$(first(cols))-$(last(cols))", RigorousUlam2D)
    part !== nothing && isfile(part) && (println("$(basename(part)) exists"); return nothing)
    powers = record_file(datadir, "l1_mixing.$(map)_d$(depth)_$amp", RigorousUlam2D)
    cols === nothing && isfile(powers) && error("$powers exists without its summary row; remove it to run again")
    environment = save_environment(part === nothing ? summary : part)

    sys = ulam_system(map)
    blkfile = block_file(datadir, map, depth)
    t0 = time()
    P = noisy_ulam_operator(sys, Serialization.deserialize(blkfile), depth, a)
    densrec = try latest_record(datadir, "stationary_density") catch; nothing end
    densrow = densrec === nothing ? nothing :
        findfirst(r -> r["map"] == map && r["depth"] == string(depth) && r["a"] == string(a), last(read_record(densrec)))
    densfile = densrow === nothing ? nothing : last(read_record(densrec))[densrow]["density_file"]
    u = densfile === nothing ? ulam_fixed_point(P.info.Amul, P.n).v :
        Serialization.deserialize(joinpath(datadir, densfile)).density
    length(u) == P.n || error("the density has length $(length(u)), the operator n = $(P.n)")
    cs = cols === nothing ? (1:P.n) : cols

    header = run_header(RigorousUlam2D, script; map, depth, a, M, block,
        cols = "$(first(cs)):$(last(cs))", n = P.n, blockfile = basename(blkfile), blocksha = file_sha256(blkfile),
        u = densfile === nothing ? "power iteration" : densfile, η1 = P.info.η1, eta1A = P.info.eta1A, normP = P.normP)
    print(header)
    @printf("%s depth %d n = %d, a = %s, h = %.6g, var₁₂(ρ_ξ) = %.6g, columns %s (%.1f s)\n", map, depth, P.n, a, P.h, P.var12ρ, cs, time() - t0)
    flush(stdout)

    tc = @elapsed C = norms_of_powers_columns(P.info.Bmul!, P.n, M; η1 = P.info.η1, eta1A = P.info.eta1A,
        normP = P.normP, u, block, verbose = true, cols = cs)
    @printf("columns: %d × %d products in %.1f s\n", length(cs), M, tc)
    if part !== nothing
        open_record(part, header * "# environment = $environment\n# seconds = $(round(tc; digits = 1))\n", ["j", "C", "bits"])
        for j in 1:M
            append_record(part, [j, write_endpoint(C[j])[1], 53])
        end
        println("recorded in $(basename(part))")
        return (; C, sigma = nothing, n = P.n, P.h, P.var12ρ, record = part)
    end
    s = record_column_sweep(powers, summary, map, depth, a, C, header, environment, tc; P.h, P.var12ρ, P.n)
    s === nothing ? println("no C_j below 1 up to j = $M") :
        @printf("Σ_grid ≤ %.6g at k = %d (α = %.6g), Σ_V ≤ %.6g\n", s.grid, s.k, s.α, s.V)
    println("recorded in $(basename(powers)) and $(basename(summary))")
    return (; C, sigma = s, n = P.n, P.h, P.var12ρ, record = powers)
end

"""
    run_transfer(map, depth, a, fine; datadir, script = "bench/l1_transfer.jl") -> Vector of rows

The L¹ error of the stationary density on the finer grids of the depths `fine`, from the column
sweep of the coarse depth `depth`.  The refined bounds `C_k ≥ ‖P_h^k|_{V_h}‖₁` of the latest
l1_mixing record of this host go through `coarse_fine_norms` (eq:coarsefine, with
`θ₁ = θ₂ = θ_h = (h/2) var₁₂(ρ_ξ)` of Theorem C) to `F_k ≥ ‖Λ^k|_V‖₁` for `Λ` the noisy Ulam operator
`P_{h'}` of any finer dyadic grid, or `L_ξ`; `sigma_from_norms(F).grid` bounds `‖(I − Λ)⁻¹|_V‖₁`.  For
each fine depth with a row in the latest stationary_density record (its `ε₁`, `ε₂`; the defect is
recomputed, `defect_apriori(h', var₁₂(ρ_ξ))`), as in prop:l1error:

    ‖f_ξ − f_{h'}‖₁ ≤ Σ · 3θ_{h'},   ‖f̃ − f_{h'}‖₁ ≤ fixed_point_distance(Σ, ε₁, ε₂),

their sum bounds `‖f_ξ − f̃‖₁`, and `swapped_error` gives `Σ (2θ_{h'} + ε₁/(1 − ε₂)) + ε₂`.  Every
operation rounds upward; the inputs are read back exactly.  Each fine depth not already in
`datadir/l1_transfer.<commit>.<host>.tsv` gets a row there; the rows written are returned.
"""
function run_transfer(map::AbstractString, depth::Int, a::Rational, fine::AbstractVector{Int};
        datadir::AbstractString, script::AbstractString = "bench/l1_transfer.jl")
    t0 = time()
    rd(s) = Float64(read_endpoint(s, 53), RoundUp)        # exact for a double written by write_endpoint; upward otherwise
    record = record_file(datadir, "l1_transfer", RigorousUlam2D)
    done = recorded_keys(record, "map", "coarse_depth", "a", "fine_depth")
    environment = save_environment(record)
    powers = latest_record(datadir, "l1_mixing.$(map)_d$(depth)_$(_amp(a))")
    Cr = [rd(r["C_refined"]) for r in sort(last(read_record(powers)); by = r -> parse(Int, r["j"]))]
    densrec = latest_record(datadir, "stationary_density")
    dens = last(read_record(densrec))
    g = grid_constants(ulam_system(map), depth, a)

    print(run_header(RigorousUlam2D, script; map, depth, a, fine,
        powers = basename(powers), densities = basename(densrec)))
    @printf("h = %.6g, var₁₂(ρ_ξ) = %.6g, (h/2) var₁₂(ρ_ξ) = %.6g, M = %d\n", g.h, g.var12ρ, mul_up(0.5 * g.h, g.var12ρ), length(Cr))
    F = coarse_fine_norms(Cr, g.h, g.var12ρ)
    s = any(<(1), F) ? sigma_from_norms(F) : nothing
    s === nothing ? println("no F_k below 1 up to k = $(length(F))") :
        @printf("F_k < 1 first at k = %d; Σ ≤ %.6g at k = %d (α = %.6g)\n", findfirst(<(1), F), s.grid, s.k, s.α)

    open_record(record, run_header(RigorousUlam2D, script),
        ["map", "coarse_depth", "a", "fine_depth", "Sigma", "k", "defect", "err_apriori", "dist", "err", "err_swapped",
         "powers_file", "densities_file", "seconds", "environment"])
    rows = []
    for d in fine
        (map, string(depth), string(a), string(d)) in done && continue
        i = findfirst(r -> r["map"] == map && r["depth"] == string(d) && r["a"] == string(a), dens)
        i === nothing && (println("depth $d: no row in $(basename(densrec))"); continue)
        g′ = grid_constants(ulam_system(map), d, a)
        defect, ε1, ε2 = defect_apriori(g′.h, g′.var12ρ), rd(dens[i]["eps1"]), rd(dens[i]["eps2"])
        if s === nothing
            append_record(record, [map, depth, a, d, "Inf", 0, write_endpoint(defect)[1], "Inf", "Inf", "Inf", "Inf",
                basename(powers), basename(densrec), round(time() - t0; digits = 1), environment])
            push!(rows, (; fine_depth = d, Sigma = Inf, k = 0, defect, err_apriori = Inf, dist = Inf, err = Inf, err_swapped = Inf))
            continue
        end
        err_apriori = mul_up(s.grid, defect)
        dist = fixed_point_distance(s.grid, ε1, ε2)
        err = add_up(err_apriori, dist)
        err_swapped = swapped_error(s.grid, g′.h, g′.var12ρ, ε1, ε2)
        @printf("depth %2d: (3h'/2) var₁₂(ρ_ξ) = %.6g, ‖f_ξ − f_h'‖₁ ≤ %.6g, ‖f̃ − f_h'‖₁ ≤ %.3e, ‖f_ξ − f̃‖₁ ≤ %.6g, swapped ≤ %.6g\n",
            d, defect, err_apriori, dist, err, err_swapped)
        append_record(record, [map, depth, a, d, write_endpoint(s.grid)[1], s.k, write_endpoint(defect)[1],
            write_endpoint(err_apriori)[1], write_endpoint(dist)[1], write_endpoint(err)[1], write_endpoint(err_swapped)[1],
            basename(powers), basename(densrec), round(time() - t0; digits = 1), environment])
        push!(rows, (; fine_depth = d, Sigma = s.grid, k = s.k, defect, err_apriori, dist, err, err_swapped))
    end
    println("recorded in $(basename(record))")
    return rows
end
