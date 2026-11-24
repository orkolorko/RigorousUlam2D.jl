"""
The named stages the drivers in `bench/` are thin over, and the names of the files they write:
the noisy Ulam operator of a map of `maps.jl` at one depth and one noise amplitude, read from a
block of `bench/tree_deepen.jl`; and the file names, which carry the script, the parameters,
the commit (for the records of `CertifiedRecords`) and the host.
"""

export ulam_system, block_file, record_file, grid_constants, noisy_ulam_operator, file_sha256,
    record_column_sweep

using CertifiedRecords: run_header, open_record, append_record, write_endpoint
import SHA

"""
    ulam_system(name; kwargs...) -> (; name, F, domain, area)

`henon_system` or `lorenz_system` by name (`"henon"`, `"lorenz"`), with `kwargs` passed on.
"""
ulam_system(name::AbstractString; kwargs...) =
    name == "henon" ? henon_system(; kwargs...) :
    name == "lorenz" ? lorenz_system(; kwargs...) :
    throw(ArgumentError("unknown map $name: henon or lorenz"))

"""
    block_file(dir, map, depth; level = 14, bisect = 30, xinv = true, d1 = 12, host = gethostname())

The block `L_lo, U` written by `bench/tree_deepen.jl` for `map` at `depth`:
`dir/tree_deepen.<map>_d<depth>_level<level>_area<bisect>[_xinv]_t<d1>.<host>.jls`.
"""
block_file(dir::AbstractString, map::AbstractString, depth::Int; level::Int = 14, bisect::Int = 30,
           xinv::Bool = true, d1::Int = 12, host::AbstractString = gethostname()) =
    joinpath(dir, "tree_deepen.$(map)_d$(depth)_level$(level)_area$(bisect)$(xinv ? "_xinv" : "")_t$(d1).$host.jls")

"""
    record_file(dir, prefix, mod; host = gethostname()) -> path

`dir/<prefix>.<commit>.<host>.tsv`, `<commit>` the short hash of the package `mod`: the name
under which a driver writes a record of `CertifiedRecords` and under which `latest_record`
finds it.
"""
function record_file(dir::AbstractString, prefix::AbstractString, mod::Module; host::AbstractString = gethostname())
    commit = try readchomp(`git -C $(pkgdir(mod)) rev-parse --short HEAD`) catch; "unknown" end
    return joinpath(dir, "$prefix.$commit.$host.tsv")
end

"""
    grid_constants(sys, depth, a) -> (; k, h, varρ, area)

For the map `sys` on the grid of depth `depth` and the uniform noise on `[-a, a]²`: `k = 2^depth`
cells a side, the side `h` of a cell (a dyadic fraction, exact), `var(ρ_ξ) = V_1 + V_2 = 2/a`
(lem:square_variation) rounded upward, and the area of the domain.
"""
function grid_constants(sys, depth::Int, a::Rational)
    Float64(a) == a || throw(ArgumentError("a = $a is not a double"))
    side = sup(sys.domain[1]) - inf(sys.domain[1])    # 1 or 2, exact
    k = 2^depth
    h = side / k
    return (; k, h, varρ = div_up(2.0 * denominator(a), Float64(numerator(a))),
            area = side * side)
end

"""
    noisy_ulam_operator(sys, blk, depth, a) -> (; op, info, n, k, h, varρ, area, N, normP)

The noisy Ulam operator `P_h = π N_ξ π L π` of the map `sys` (`ulam_system`) on the grid of
depth `depth`, from a block `blk` of `bench/tree_deepen.jl` (`L_lo ≤ L ≤ L_lo + U`, indexed as
`build_uniform_partition`, permuted here to `grid_order`), with `N_ξ` the Ulam-projected uniform
noise on `[-a, a]²`, periodic, `a` a dyadic rational: `noisy_operator` of `L_lo, U` and
`SquareNoise`; with the constants of `grid_constants` and `normP ≥ ‖L_lo + U‖₁ ≥ ‖P_h‖₁`, upward.
"""
function noisy_ulam_operator(sys, blk, depth::Int, a::Rational)
    g = grid_constants(sys, depth, a)
    perm = grid_order(sys.domain, depth)
    L_lo, U = blk.L_lo[perm, perm], blk.U[perm, perm]
    N = SquareNoise(sys.domain, depth, Float64(a); bc = :periodic, slots = Threads.nthreads())
    op, info = noisy_operator(L_lo, U, N)
    normP = RP._max_colsum_up(dropzeros!(add_up.(L_lo, U)))        # ≥ ‖L‖₁ ≥ ‖P_h‖₁, ‖N‖₁ ≤ 1
    return (; op, info, n = size(L_lo, 1), g..., N, normP)
end

"""
    file_sha256(path) -> String

The SHA-256 of a file's bytes, in hexadecimal: written in the header of a record so that the
parts of a split run can be checked to have read the same input.
"""
file_sha256(path::AbstractString) = bytes2hex(open(SHA.sha256, path))

"""
    record_column_sweep(datadir, script, map, depth, a, C, header, environment, seconds; h, varρ, n)

What `bench/l1_mixing.jl` and `bench/l1_mixing_merge.jl` record of a whole column sweep: the
bounds `C[j] ≥ ‖P_h^j|_{V_h}‖₁`, refined by submultiplicativity, in their own record
`<script>.<map>_d<depth>_a<p>-<q>.<commit>.<host>.tsv` (columns `j`, `C`, `C_refined`, `bits`,
exact doubles), and one row of the summary `<script>.<commit>.<host>.tsv`: `sigma_from_norms` of
the refined bounds, the a priori defect `(3h/2) var(ρ_ξ)` (`defect_apriori`) and `Σ_V` times it.  Returns the
`sigma_from_norms` result, or `nothing` when no bound is below 1.
"""
function record_column_sweep(datadir, script, map, depth, a::Rational, C::Vector{Float64}, header,
                             environment, seconds; h, varρ, n)
    M = length(C)
    amp = "a$(numerator(a))-$(denominator(a))"
    summary = record_file(datadir, script, RigorousUlam2D)
    powers = record_file(datadir, "$script.$(map)_d$(depth)_$amp", RigorousUlam2D)
    Cr = refine_norms_of_powers(C, M)
    open_record(powers, header * "# environment = $environment\n", ["j", "C", "C_refined", "bits"])
    for j in 1:M
        append_record(powers, [j, write_endpoint(C[j])[1], write_endpoint(Cr[j])[1], 53])
    end
    s = any(<(1), Cr) ? sigma_from_norms(Cr) : nothing
    defect = defect_apriori(h, varρ)
    err = s === nothing ? Inf : mul_up(s.V, defect)
    open_record(summary, run_header(RigorousUlam2D, "bench/$script.jl"),
        ["map", "depth", "a", "n", "M", "powers_file", "Sigma_grid", "Sigma_V", "k", "defect", "err_apriori",
         "seconds", "environment"])
    append_record(summary, [map, depth, a, n, M, basename(powers),
        s === nothing ? "Inf" : write_endpoint(s.grid)[1], s === nothing ? "Inf" : write_endpoint(s.V)[1],
        s === nothing ? 0 : s.k, write_endpoint(defect)[1], err === Inf ? "Inf" : write_endpoint(err)[1],
        round(seconds; digits = 1), environment])
    return s
end
