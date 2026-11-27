# The records behind Table 1 and the figures of the paper, recomputed from the code of this
# commit and compared with what is on record, bit for bit.  The records are those of
# figures.sources.2a75ede.ibis.txt (paper/figures.jl at 2a75ede on ibis), in the data directory
# ~/ResearchData/RigorousUlam2D_Data/data (or RIGOROUSULAM2D_DATADIR); a record whose content is
# not on this machine is skipped, so on a machine without the data this testset checks nothing.
#
# For each (map, a) of the table, with the column sweep of the coarse depth d_c and the density
# record of the same run:
#
# - the powers record: `refine_norms_of_powers` of the C_j gives the recorded C_refined, and
#   `sigma_from_norms` the Σ_grid, Σ_V and k of the l1_mixing summary row;
# - every l1_transfer row computed from that powers record: Σ, k, the defect 3θ_h, the a priori
#   bound, ‖f̃ − f_h‖₁, the bound ‖f_ξ − f̃‖₁ and the swapped bound, from C_refined and the ε₁, ε₂ of
#   the density record, as bench/l1_transfer.jl computes them;
# - the block of d_c named in the powers record: `noisy_ulam_operator` gives the η₁, η₁ᴬ and
#   ‖P_h‖₁ bound written in its header (and the block's SHA-256, where the header has one);
# - the depth-8 row of the density record, from its block: the power iteration, ε₁, ε₂, the
#   coordinate variations and the density itself.
#
# Not recomputed: the column sweeps themselves (hours on 16 threads each), the trees of
# bench/tree_deepen.jl, and the depth-12 densities (their blocks are gigabytes).

using Test, RigorousUlam2D, CertifiedRecords
using BallArithmetic: add_up, mul_up
import Serialization

const RECDIR = expanduser(get(ENV, "RIGOROUSULAM2D_DATADIR", "~/ResearchData/RigorousUlam2D_Data/data"))

# (map, a, density record, powers record): the rows of figures.sources.2a75ede.ibis.txt
const TABLE = [
    ("lorenz", 77 // 256, "stationary_density.d78ed9d.ibis.tsv", "l1_mixing.lorenz_d8_a77-256.d78ed9d.ibis.tsv"),
    ("lorenz", 51 // 256, "stationary_density.d78ed9d.ibis.tsv", "l1_mixing.lorenz_d8_a51-256.d78ed9d.ibis.tsv"),
    ("lorenz", 51 // 512, "stationary_density.d78ed9d.ibis.tsv", "l1_mixing.lorenz_d9_a51-512.d78ed9d.ibis.tsv"),
    ("lorenz", 13 // 256, "stationary_density.bde4b2c.albatross.tsv", "l1_mixing.lorenz_d9_a13-256.9b89758.ibis.tsv"),
    ("mlorenz_rho18-5", 77 // 256, "stationary_density.172fde7.albatross.tsv", "l1_mixing.mlorenz_rho18-5_d8_a77-256.172fde7.albatross.tsv"),
    ("mlorenz_rho18-5", 51 // 256, "stationary_density.172fde7.albatross.tsv", "l1_mixing.mlorenz_rho18-5_d8_a51-256.172fde7.albatross.tsv"),
    ("mlorenz_rho18-5", 51 // 512, "stationary_density.172fde7.albatross.tsv", "l1_mixing.mlorenz_rho18-5_d8_a51-512.172fde7.albatross.tsv"),
    ("henon", 77 // 256, "stationary_density.d78ed9d.ibis.tsv", "l1_mixing.henon_d9_a77-256.d78ed9d.ibis.tsv"),
    ("henon", 51 // 256, "stationary_density.d78ed9d.ibis.tsv", "l1_mixing.henon_d8_a51-256.d78ed9d.ibis.tsv"),
]

_have(f) = isfile(joinpath(RECDIR, f))          # false for an annexed file whose content is elsewhere
_rd(s) = s == "Inf" ? Inf : Float64(read_endpoint(s, 53), RoundUp)
_wr(x) = x == Inf ? "Inf" : write_endpoint(x)[1]
_params(hdr) = Dict(m.captures[1] => strip(m.captures[2])
                    for l in hdr for m in eachmatch(r"(?:^#\s*|,\s*)([^=,#]+?) = ([^,]+)", l))

@testset "the records of the paper, recomputed" begin
    if !isdir(RECDIR) || !any(t -> _have(t[4]), TABLE)
        @info "test_records: no record of the paper's table on this machine ($RECDIR); nothing checked"
        @test true
    end
    summaries = filter(f -> occursin(r"^l1_mixing\.[0-9a-f]+\.\w+\.tsv$", f) && _have(f), readdir(RECDIR))
    transfers = filter(f -> occursin(r"^l1_transfer\.[0-9a-f]+\.\w+\.tsv$", f) && _have(f), readdir(RECDIR))
    for (map, a, densrec, powrec) in TABLE
        _have(powrec) || continue
        depth = parse(Int, match(r"_d([0-9]+)_", powrec).captures[1])
        @testset "$map a = $a, coarse depth $depth" begin
            hdr, rows = read_record(joinpath(RECDIR, powrec))
            sort!(rows; by = r -> parse(Int, r["j"]))
            C = [_rd(r["C"]) for r in rows]
            Cr = refine_norms_of_powers(C, length(C))
            @test _wr.(Cr) == [r["C_refined"] for r in rows]
            s = sigma_from_norms(Cr)
            g = grid_constants(ulam_system(map), depth, a)

            # the summary row of the sweep
            nsum = 0
            for f in summaries, r in last(read_record(joinpath(RECDIR, f)))
                r["powers_file"] == powrec || continue
                nsum += 1
                @test (r["Sigma_grid"], r["Sigma_V"], r["k"]) == (_wr(s.grid), _wr(s.V), string(s.k))
            end
            @test nsum >= 1

            # every l1_transfer row from this sweep
            densrows = _have(densrec) ? last(read_record(joinpath(RECDIR, densrec))) : nothing
            F = coarse_fine_norms(Cr, g.h, g.var12ρ)
            st = any(<(1), F) ? sigma_from_norms(F) : nothing
            for f in transfers, r in last(read_record(joinpath(RECDIR, f)))
                r["powers_file"] == powrec || continue
                densrows === nothing || r["densities_file"] != densrec && continue
                d = parse(Int, r["fine_depth"])
                g′ = grid_constants(ulam_system(map), d, a)
                defect = defect_apriori(g′.h, g′.var12ρ)
                @test r["defect"] == _wr(defect)
                if st === nothing
                    @test r["Sigma"] == "Inf"
                    continue
                end
                @test (r["Sigma"], r["k"]) == (_wr(st.grid), string(st.k))
                densrows === nothing && continue
                dr = only(filter(x -> x["map"] == map && x["depth"] == string(d) && x["a"] == string(a), densrows))
                ε1, ε2 = _rd(dr["eps1"]), _rd(dr["eps2"])
                err_apriori = mul_up(st.grid, defect)
                dist = fixed_point_distance(st.grid, ε1, ε2)
                @test (r["err_apriori"], r["dist"], r["err"], r["err_swapped"]) ==
                      (_wr(err_apriori), _wr(dist), _wr(add_up(err_apriori, dist)),
                       _wr(swapped_error(st.grid, g′.h, g′.var12ρ, ε1, ε2)))
            end

            # the operator of the sweep and its constants
            p = _params(hdr)
            blk = p["blockfile"]
            if _have(blk)
                haskey(p, "blocksha") && @test file_sha256(joinpath(RECDIR, blk)) == p["blocksha"]
                P = noisy_ulam_operator(ulam_system(map), Serialization.deserialize(joinpath(RECDIR, blk)), depth, a)
                @test (string(P.info.η1), string(P.info.eta1A), string(P.normP)) == (p["η1"], p["eta1A"], p["normP"])
            end

            # the depth-8 density of the density record, from its block
            if densrows !== nothing
                dr = only(filter(x -> x["map"] == map && x["depth"] == "8" && x["a"] == string(a), densrows))
                if _have(dr["blockfile"]) && _have(dr["density_file"])
                    P = noisy_ulam_operator(ulam_system(map), Serialization.deserialize(joinpath(RECDIR, dr["blockfile"])), 8, a)
                    fp = ulam_fixed_point(P.info.Amul, P.n; tol = 1e-14, maxiter = 20_000)   # the driver's defaults
                    @test string(fp.iterations) == dr["iterations"]
                    @test fp.v == Serialization.deserialize(joinpath(RECDIR, dr["density_file"])).density
                    res = fixed_point_residual(P.info.Amul, fp.v, P.info.η1, P.info.eta1A)
                    @test (_wr(res.ε1), _wr(res.ε2)) == (dr["eps1"], dr["eps2"])
                    cv = coordinate_variations(fp.v, P.k, P.h)
                    if haskey(dr, "var1f")
                        @test (_wr(cv.var1), _wr(cv.var2)) == (dr["var1f"], dr["var2f"])
                    end
                end
            end
        end
    end
end
