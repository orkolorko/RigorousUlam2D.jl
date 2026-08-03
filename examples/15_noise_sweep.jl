# NOISE SWEEP — how far can the noise be reduced before the certificate stops closing?
#
# The whole exterior certificate rests on ONE inequality, ‖G₂₂‖ < R, and ‖G₂₂‖ is made small by
# deflating the top singular directions. That works because noise-smoothing makes the singular
# values DECAY. Less noise ⇒ slower decay ⇒ more singular vectors needed ⇒ eventually the far block
# cannot be made deep and the method declines. This sweep measures where that boundary is for a real
# operator, rather than asserting it.
#
# The noise enters the discretization as an INTEGER cell radius, rx = round(σ / cellwidth), so at
# depth 6 (cellwidth = 2/64 = 0.03125 for Hénon) the reachable settings are
#     noise 0.2 → rx=3 (7-cell box) ; 0.1 → rx=2 (5-cell) ; 0.05 → rx=1 (3-cell, the minimum).
# Anything smaller still floors at rx=1, so a genuinely finer noise needs depth 7 — which the
# full-basis certificate cannot afford: selective reorthogonalization stores the Krylov basis, i.e.
# O(n·N) memory, ≈260 MB at n=4096 but ≈4 TB at n=16384. Depth 6 it is.
#
#   CACHED=1 DEPTH=6 RCIRC=0.6 MAP=henon julia --project=examples examples/15_noise_sweep.jl
import Pkg; Pkg.activate(@__DIR__)
include(joinpath(@__DIR__, "ulam_matvec.jl"))
using RigorousPseudospectra, Printf, LinearAlgebra
const RP = RigorousPseudospectra

build  = get(ENV, "CACHED", "") == "1" ? build_matvec_cached : build_matvec
depth  = parse(Int, get(ENV, "DEPTH", "6"))
R      = parse(Float64, get(ENV, "RCIRC", "0.6"))
mapnm  = get(ENV, "MAP", "henon")
noises = [parse(Float64, s) for s in split(get(ENV, "NOISES", "0.2,0.1,0.05"), ",")]

if mapnm == "henon"
    dom = IntervalBox(interval(-1.0, 1.0), interval(-1.0, 1.0))
    F, per, encl, ρfl = henon_rescaled, true, first_order_enclosure, 0.004
else
    dom = IntervalBox(interval(-0.5, 0.5), interval(-0.5, 0.5))
    F, per, encl, ρfl = contracting_lorenz, false, zeroth_order_enclosure, 0.001
end
cellw = diam(dom[1]) / 2^depth

@printf("NOISE SWEEP — %s, depth %d (n=%d), R=%.2f, cell width %.5f\n",
        mapnm, depth, 4^depth, R, cellw)
println("Less noise ⇒ slower singular decay ⇒ deeper deflation needed for ‖G₂₂‖ < R.")
println("="^100)
@printf("%-7s %-4s %-6s %-7s %-9s %-7s %-8s %-9s %s\n",
        "noise", "rx", "nsing", "dim(Q)", "‖G₂₂‖", "<R ?", "#out", "time(s)", "certified gap |λ₂| ≤")
println("="^100)

for w in noises
    NOISE = uniform_noise_box(w, 2)
    rx = max(1, Int(round((w/2) / cellw)))
    local L, La, n
    try
        L, La, n, _ = build(F, dom, NOISE, depth; periodic=per, enclosure_method=encl)
    catch e
        @printf("%-7.3f build FAILED: %s\n", w, first(sprint(showerror,e),60)); flush(stdout); continue
    end
    t0 = time()
    local out
    try
        out = certify_leading_eigs_sing(L, La, n; Rcirc=R, nblock_eig=80, margin=0.8, ρfloor=ρfl)
    catch e
        @printf("%-7.3f %-4d DECLINED: %s\n", w, rx, first(sprint(showerror,e),60)); flush(stdout); continue
    end
    el = time() - t0
    nout = length(out.results)
    allok = nout > 0 && all(r -> r.cert.certified, out.results)
    # CERTIFIED SPECTRAL GAP. By the exact-count proposition the certified list outside R is
    # complete, so:
    #   nout ≥ 2 : the second-largest modulus is KNOWN (it is the runner-up of the certified list,
    #              enclosed), so the gap |λ₁| − |λ₂| is certified SHARPLY, not merely bounded;
    #   nout = 1 : nothing else lies outside R, so |λ₂| ≤ R and the gap is ≥ |λ₁| − R.
    # Reporting a gap only in the second case (as an earlier version did) was backwards — the
    # many-eigenvalue case gives the *better* statement.
    mods = sort([abs(r.λ) for r in out.results]; rev = true)
    gap = if !(allok && out.far_deep)
        "—"
    elseif nout >= 2
        @sprintf("%.4f  (sharp: |λ₂|=%.4f)", mods[1] - mods[2], mods[2])
    else
        @sprintf("≥ %.4f  (|λ₂| ≤ %.3f)", mods[1] - R, R)
    end
    @printf("%-7.3f %-4d %-6d %-7d %-9.4f %-7s %-8d %-9.1f %s\n",
            w, rx, out.nsing, out.dimQ, out.G22norm, out.far_deep ? "yes ✓" : "NO ✗",
            nout, el, gap)
    for r in out.results
        @printf("           |λ|=%.4f λ=%+.4f%+.4fi  σ_min≥%.3e %s  enclosure=%.1e\n",
                abs(r.λ), real(r.λ), imag(r.λ), r.cert.floor,
                r.cert.certified ? "✓" : "✗", r.cert.enclosure)
    end
    flush(stdout)
end
println("="^100)
println("‖G₂₂‖ rising toward R as the noise falls IS the regime boundary — when it crosses,")
println("the single hypothesis fails and the method declines rather than returning a false answer.")
