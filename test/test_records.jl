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
# - the latest l1_transfer row of each configuration computed from that powers record
#   (`latest_transfer_rows`, the rows paper/figures.jl reads): Σ, k, the defect 3θ_h, the a priori
#   bound, ‖f̃ − f_h‖₁, the bound ‖f_ξ − f̃‖₁ and the swapped bound, from C_refined and the ε₁, ε₂ of
#   the density record, as bench/l1_transfer.jl computes them;
# - the block of d_c named in the powers record: `noisy_ulam_operator` gives the η₁, η₁ᴬ and
#   ‖P_h‖₁ bound written in its header (and the block's SHA-256, where the header has one);
# - the coarsest row (depth 8, or 10) of the density record, from its block: the power iteration, ε₁, ε₂, the
#   coordinate variations and the density itself.
#
# Not recomputed: the column sweeps at depths 8 and 9 (2658 s on 16 threads of ibis for one at
# depth 8), the trees of bench/tree_deepen.jl, and the depth-12 densities.
#
# The second testset recomputes whole column sweeps at depth 6: the records
# l1_mixing.<map>_d6_a<p>-<q>.nighthawk.jls of the L¹ column sweep of 2026-11-16 (RigorousUlam2D
# b1f7625, held on nighthawk only), whose η₁, η₁ᴬ, ‖P_h‖₁ bound and C_j are compared bit for bit
# with those of `noisy_ulam_operator` and `norms_of_powers_columns` on the same block and density.

using Test, RigorousUlam2D, CertifiedRecords
import Serialization
using SparseArrays

const RECDIR = expanduser(get(ENV, "RIGOROUSULAM2D_DATADIR", "~/ResearchData/RigorousUlam2D_Data/data"))

const TABLE = [(e.map, e.a, e.density_record, e.powers_record) for e in PAPER_RECORDS]

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
    names = isdir(RECDIR) ? readdir(RECDIR) : String[]
    summaries = filter(f -> occursin(r"^l1_mixing\.[0-9a-f]+\.\w+\.tsv$", f) && _have(f), names)
    transfers = filter(f -> occursin(r"^l1_transfer\.[0-9a-f]+\.\w+\.tsv$", f) && _have(f), names)
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

            # recertify_from_records reproduces the recorded rows at depth 12
            if _have(densrec)
                e = only(filter(e -> e.powers_record == powrec, PAPER_RECORDS))
                rc = recertify_from_records(e; datadir = RECDIR)
                @test _wr.(rc.C_refined) == rc.recorded.C_refined
                for r in rc.recorded.transfer
                    @test (r["Sigma"], r["err"], r["err_swapped"]) == (_wr(rc.Σ.grid), _wr(rc.bounds.err), _wr(rc.bounds.err_swapped))
                end
            end

            # every l1_transfer row from this sweep
            densrows = _have(densrec) ? last(read_record(joinpath(RECDIR, densrec))) : nothing
            F = coarse_fine_norms(Cr, g.h, g.var12ρ)
            st = any(<(1), F) ? sigma_from_norms(F) : nothing
            for r in latest_transfer_rows(RECDIR)          # the rows paper/figures.jl reads
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
                b = transfer_bounds(st.grid, g′, _rd(dr["eps1"]), _rd(dr["eps2"]))
                @test (r["err_apriori"], r["dist"], r["err"], r["err_swapped"]) ==
                      (_wr(b.err_apriori), _wr(b.dist), _wr(b.err), _wr(b.err_swapped))
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
                # the coarsest depth of the record for this configuration: 8, or 10 for the
                # lorenz a = 13/256 record of bde4b2c on albatross, which has depths 10 and 12 only
                cands = filter(x -> x["map"] == map && x["a"] == string(a) && parse(Int, x["depth"]) <= 10, densrows)
                dr = isempty(cands) ? nothing : cands[argmin([parse(Int, x["depth"]) for x in cands])]
                if dr !== nothing && _have(dr["blockfile"]) && _have(dr["density_file"])
                    dd = parse(Int, dr["depth"])
                    P = noisy_ulam_operator(ulam_system(map), Serialization.deserialize(joinpath(RECDIR, dr["blockfile"])), dd, a)
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

@testset "the depth-6 sweeps of 2026-11-16, recomputed" begin
    files = isdir(RECDIR) ? filter(f -> occursin(r"^l1_mixing\.\w+_d6_a[0-9]+-[0-9]+\.nighthawk\.jls$", f) && _have(f),
                                   readdir(RECDIR)) : String[]
    isempty(files) && (@info "test_records: no depth-6 sweep of 2026-11-16 on this machine"; @test true)
    for f in files
        r = Serialization.deserialize(joinpath(RECDIR, f))
        (_have(r.block) && _have(r.u_from)) || continue
        @testset "$(r.map) a = $(r.a)" begin
            P = noisy_ulam_operator(ulam_system(r.map), Serialization.deserialize(joinpath(RECDIR, r.block)), r.depth, r.a)
            @test (P.info.w, P.info.colL, P.info.colE, P.info.γN, P.info.η1, P.info.eta1A, P.normP) ==
                  (r.operator.w, r.operator.colL, r.operator.colE, r.operator.γN, r.operator.η1, r.operator.eta1A, r.operator.normP)
            u = Serialization.deserialize(joinpath(RECDIR, r.u_from)).density
            C = norms_of_powers_columns(P.info.Bmul!, P.n, r.M; η1 = P.info.η1, eta1A = P.info.eta1A,
                                        normP = P.normP, u, block = r.columns_per_block)
            @test C == r.C
            @test refine_norms_of_powers(C, r.M) == r.C_refined
        end
    end
end
