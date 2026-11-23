"""
The named stages the drivers in `bench/` are thin over, and the names of the files they write:
the noisy Ulam operator of a map of `maps.jl` at one depth and one noise amplitude, read from a
block of `bench/tree_deepen.jl`; and the file names, which carry the script, the parameters,
the commit (for the records of `CertifiedRecords`) and the host.
"""

export ulam_system, block_file, record_file, grid_constants, noisy_ulam_operator

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
    grid_constants(sys, depth, a) -> (; k, h, δ, varρ, area)

For the map `sys` on the grid of depth `depth` and the uniform noise on `[-a, a]²`: `k = 2^depth`
cells a side, the side `h` of a cell (a dyadic fraction, exact), its diameter `δ = √2 h` and
`var(ρ_ξ) = 2/a` (lem:square_variation), both rounded upward, and the area of the domain.
"""
function grid_constants(sys, depth::Int, a::Rational)
    Float64(a) == a || throw(ArgumentError("a = $a is not a double"))
    side = sup(sys.domain[1]) - inf(sys.domain[1])    # 1 or 2, exact
    k = 2^depth
    h = side / k
    return (; k, h, δ = mul_up(sqrt_up(2.0), h), varρ = div_up(2.0 * denominator(a), Float64(numerator(a))),
            area = side * side)
end

"""
    noisy_ulam_operator(sys, blk, depth, a) -> (; op, info, n, k, h, δ, varρ, area, N, normP)

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
