# The L¹ error of the stationary density on finer grids, from the column sweep of a coarse one.
#
# For one (MAP, A) and a coarse depth DEPTH, the bounds C_k ≥ ‖P_h^k|_{V_h}‖₁ of the record
# l1_mixing.<map>_d<DEPTH>_a<p>-<q>.<host>.jls (bench/l1_mixing.jl, refined by submultiplicativity)
# go through `coarse_fine_norms` (Lemma 29 of Galatolo, Monge, Nisoli 2020 with the constant
# δ var(ρ_ξ) of Theorem D) to F_k ≥ ‖Λ^k|_V‖₁ for Λ the noisy Ulam operator P_{h'} of any finer
# dyadic grid, or L_ξ itself; `sigma_from_norms(F).grid` bounds Σ_{j ≥ 0} ‖Λ^j|_V‖₁, hence
# ‖(I − Λ)⁻¹|_V‖₁, the F_k bounding Λ on all of V.  For each finer depth in FINE with a
# stationary_density record of the same (map, a), whose `defect` is 2δ' var(ρ_ξ) ≥ ‖(L_ξ − P_{h'}) f_ξ‖₁
# and whose ε₁, ε₂ are the residuals of its computed density f̃:
#     ‖f_ξ − f_{h'}‖₁ ≤ Σ · 2δ' var(ρ_ξ),   ‖f̃ − f_{h'}‖₁ ≤ fixed_point_distance(Σ, ε₁, ε₂),
# and their sum bounds ‖f_ξ − f̃‖₁.  Every operation rounds upward.
#
# Env: MAP, DEPTH (8), A (rational such as 77//256), FINE (comma-separated depths, "10,12"),
# SRCHOST (host in the names of the input records, default this host).  Output, not rewritten:
#     ~/ResearchData/RigorousUlam2D_Data/data/l1_transfer.<map>_d<DEPTH>_a<p>-<q>.<host>.jls
#
#   julia --project=. bench/l1_transfer.jl

using RigorousUlam2D
using BallArithmetic: add_up, mul_up
using Printf, Dates
import Serialization

const HOST = chomp(read(`hostname`, String))
const SCRIPT = splitext(basename(@__FILE__))[1]
const DATADIR = expanduser(get(ENV, "DATADIR", "~/ResearchData/RigorousUlam2D_Data/data"))
const MAP = get(ENV, "MAP", "lorenz")
const DEPTH = parse(Int, get(ENV, "DEPTH", "8"))
const A = let s = get(ENV, "A", "77//256"); p, q = parse.(Int, split(s, "//")); p // q end
const FINE = parse.(Int, split(get(ENV, "FINE", "10,12"), ","))
const SRCHOST = get(ENV, "SRCHOST", HOST)

const PKG = dirname(dirname(pathof(RigorousUlam2D)))
gitrev(dir) = try chomp(read(`git -C $dir rev-parse --short HEAD`, String)) catch; "?" end
gitdirty(dir) = try !isempty(read(`git -C $dir status --porcelain --untracked-files=no -- src bench Project.toml Manifest.toml`, String)) catch; true end
gitdirty(PKG) && error("$PKG has uncommitted changes in src, bench or the project files; commit before running")
const COMMITS = (; RigorousUlam2D = gitrev(PKG),
    RigorousInvariantMeasures = gitrev(dirname(dirname(pathof(RigorousUlam2D.RIM)))))

amp = "a$(numerator(A))-$(denominator(A))"
out = joinpath(DATADIR, "$SCRIPT.$(MAP)_d$(DEPTH)_$amp.$HOST.jls")
isfile(out) && error("$out exists")
mixfile = joinpath(DATADIR, "l1_mixing.$(MAP)_d$(DEPTH)_$amp.$SRCHOST.jls")
mix = Serialization.deserialize(mixfile)
mix.map == MAP && mix.depth == DEPTH && mix.a == A || error("$mixfile is for $(mix.map), depth $(mix.depth), a = $(mix.a)")

println("$SCRIPT on $HOST, $(Dates.now()), julia $VERSION, commits $COMMITS")
@printf("%s, coarse depth %d, a = %s: δ = %.6g, var(ρ_ξ) = %.6g, δ var(ρ_ξ) = %.6g; C from %s (commits %s), M = %d\n",
    MAP, DEPTH, A, mix.δ, mix.varρ, mul_up(mix.δ, mix.varρ), basename(mixfile), mix.commits, length(mix.C_refined))
F = coarse_fine_norms(mix.C_refined, mix.δ, mix.varρ)
closes = any(<(1), F)
s = closes ? sigma_from_norms(F) : nothing
closes ? @printf("F_k < 1 first at k = %d; Σ ≤ %.6g at k = %d (α = %.6g)\n", findfirst(<(1), F), s.grid, s.k, s.α) :
         println("no F_k below 1 up to k = $(length(F))")

fine = []
for d in FINE
    densfile = joinpath(DATADIR, "stationary_density.$(MAP)_d$(d)_$amp.$SRCHOST.jls")
    isfile(densfile) || (println("depth $d: no $(basename(densfile))"); continue)
    dens = Serialization.deserialize(densfile)
    dens.a == A && dens.depth == d || error("$densfile is for depth $(dens.depth), a = $(dens.a)")
    closes || (push!(fine, (; depth = d, densfile = basename(densfile), defect = dens.defect, err_apriori = NaN, dist = NaN, err = NaN)); continue)
    err_apriori = mul_up(s.grid, dens.defect)
    dist = fixed_point_distance(s.grid, dens.ε1, dens.ε2)
    err = add_up(err_apriori, dist)
    @printf("depth %2d: 2δ' var(ρ_ξ) = %.6g, ‖f_ξ − f_h'‖₁ ≤ %.6g, ‖f̃ − f_h'‖₁ ≤ %.3e, ‖f_ξ − f̃‖₁ ≤ %.6g  (%s, commits %s)\n",
        d, dens.defect, err_apriori, dist, err, basename(densfile), dens.commits)
    push!(fine, (; depth = d, densfile = basename(densfile), defect = dens.defect, err_apriori, dist, err))
end

Serialization.serialize(out, (; map = MAP, depth = DEPTH, a = A, δ = mix.δ, varρ = mix.varρ,
    mixing = basename(mixfile), mixing_commits = mix.commits, F, sigma = s, fine,
    host = HOST, script = SCRIPT, commits = COMMITS, julia = string(VERSION), time = Dates.now()))
println("wrote $out")
