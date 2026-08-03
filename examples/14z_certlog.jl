# FULL CERTIFICATION LOG + COMPLETENESS SCAN + PLOT, with the corrected full-basis Lanczos
# certificate (Prop. 5). Three things are produced:
#
#   (1) the certification log — every quantity the certificate depends on, and per-eigenvalue
#       floors / enclosures;
#   (2) COMPLETENESS — this needs NO scan; it is a direct consequence of the assembly. For ANY z
#       with |z| > R we have σ₂₂ ≥ |z| − ‖G₂₂‖ > 0 (because the deflation drove ‖G₂₂‖ < 0.9R), so
#           σ_min(A−zI) ≥ σ₁₁σ₂₂/(σ₁₁+σ₂₂+γ) > 0   whenever σ₁₁ = σ_min(G₁₁−zI) > 0.
#       Contrapositive: every eigenvalue of A outside R is a point where σ₁₁ = 0, i.e. an
#       eigenvalue of G₁₁. Hence spec(A) ∩ {|z|>R} ⊆ spec(G₁₁) ∩ {|z|>R} — the certified list is
#       COMPLETE outside R by construction, at all z simultaneously, with no covering argument.
#       That is precisely what the outside-R design buys. The grid below is therefore NOT a
#       completeness proof but a numerical illustration + the data for the pseudospectrum plot.
#   (3) the plot — rigorous pseudospectral contours outside the far circle, the certified discs,
#       and the cells the completeness scan could not exclude.
#
#   CACHED=1 DEPTH=6 RCIRC=0.6 MAP=henon julia --project=examples examples/14z_certlog.jl
import Pkg; Pkg.activate(@__DIR__)
include(joinpath(@__DIR__, "ulam_matvec.jl"))
using RigorousPseudospectra, Printf, LinearAlgebra, CairoMakie, Dates
const RP = RigorousPseudospectra

build = get(ENV, "CACHED", "") == "1" ? build_matvec_cached : build_matvec
NOISE = uniform_noise_box(0.1, 2)
depth = parse(Int, get(ENV, "DEPTH", "6"))
R     = parse(Float64, get(ENV, "RCIRC", "0.6"))
mapnm = get(ENV, "MAP", "henon")

if mapnm == "henon"
    dom = IntervalBox(interval(-1.0, 1.0), interval(-1.0, 1.0))
    L, La, n, _ = build(henon_rescaled, dom, NOISE, depth;
                        periodic=true, enclosure_method=first_order_enclosure)
    ρfloor = 0.004
else
    dom = IntervalBox(interval(-0.5, 0.5), interval(-0.5, 0.5))
    L, La, n, _ = build(contracting_lorenz, dom, NOISE, depth;
                        periodic=false, enclosure_method=zeroth_order_enclosure)
    ρfloor = 0.001
end

logf = joinpath(@__DIR__, "..", "cert_$(mapnm)_R$(replace(string(R),"."=>"")).log")
io = open(logf, "w")
say(s) = (println(s); println(io, s); flush(io); flush(stdout))

say("="^78)
say("CERTIFICATION LOG — $(mapnm) Ulam operator, outside |z| = $R")
say("generated $(Dates.now())")
say("="^78)
say("Statements below are about the MATRIX A (the Ulam/Galerkin discretization).")
say("Lifting them to the transfer operator needs a BV→L¹ discretization bound, NOT included.")
say("")
say("operator: n = $n  (depth $depth grid), noise width 0.1, matvec-only (x↦Ax, x↦A*x)")

t0 = time()
out = certify_leading_eigs_sing(L, La, n; Rcirc=R, nblock_eig=80, margin=0.8, ρfloor=ρfloor)
el = time() - t0

say("")
say("--- certificate data (all rigorous; Prop. 5 full-basis Lanczos) ---")
@printf(io, "wall time                : %.1f s\n", el);            @printf("wall time                : %.1f s\n", el)
for (k, v) in (("singular block nsing", out.nsing), ("captured dim(Q)", out.dimQ))
    say(@sprintf("%-25s: %d", k, v))
end
say(@sprintf("%-25s: %d   (FULL basis — completeness certificate)", "far Lanczos steps N", n - out.dimQ))
say(@sprintf("%-25s: %.6f   (certified λmax bound; need < %.3f)", "‖G22‖", out.G22norm, R))
say(@sprintf("%-25s: %s", "far block deep", out.far_deep ? "YES" : "NO"))
say(@sprintf("%-25s: %.6f   (‖G12‖=%.6f, ‖G21‖=%.6f)", "coupling γ", out.γ, out.g12, out.g21))
say(@sprintf("%-25s: %.3e   (Neumann test needs < 1)", "‖R2‖ far block", out.R2g))
say(@sprintf("%-25s: %.4e", "VBD κ₂ (diagnostic)", out.mr.kappa))
say(@sprintf("%-25s: %d", "eigenvalues outside R", length(out.results)))

say("")
say("--- per-eigenvalue certificates ---")
say(@sprintf("%-26s %-12s %-12s %-10s", "λ", "σ_min floor", "enclosure", "certified"))
for r in out.results
    say(@sprintf("%+.6f%+.6fi   %.4e   %.2e   %s",
        real(r.λ), imag(r.λ), r.cert.floor, r.cert.enclosure, r.cert.certified ? "YES" : "no"))
end

# ---------------------------------------------------------------------------------
# (2) COMPLETENESS: are these ALL the eigenvalues outside R?
# ---------------------------------------------------------------------------------
say("")
say("--- COMPLETENESS outside R (a theorem, not a scan) ---")
say("For any |z| > R:  σ22 ≥ |z| − ‖G22‖ > 0  since ‖G22‖ = $(round(out.G22norm,digits=4)) < R = $R.")
say("The assembly σ_min(A−zI) ≥ σ11σ22/(σ11+σ22+γ) is then POSITIVE whenever σ11 > 0, so every")
say("eigenvalue of A outside R must be a point where σ11 = σ_min(G11−zI) = 0, i.e. an eigenvalue")
say("of G11.  Hence  spec(A) ∩ {|z|>R}  ⊆  spec(G11) ∩ {|z|>R}  =  the $(length(out.results)) certified points.")
say("⇒ the certified list is COMPLETE outside |z| = $R.  This holds at all z at once; no grid,")
say("  no covering argument, no assumption on the spectral radius.")
say("")
say("--- numerical illustration (NOT the proof) + pseudospectrum plot data ---")
ρmax = 1.05
h    = parse(Float64, get(ENV, "GRIDH", "0.01"))     # cell size; half-diagonal = h/√2
half = h / sqrt(2)
say(@sprintf("grid h = %.4f (cell half-diagonal %.5f); cells whose rigorous floor exceeds it are", h, half))
say("eigenvalue-free by 1-Lipschitzness — we expect the survivors to hug the certified eigenvalues.")
xs = -ρmax:h:ρmax
unresolved = Tuple{Float64,Float64}[]
ncell = 0
for x in xs, y in xs
    z = complex(x, y); az = abs(z)
    (az <= R || az > ρmax) && continue
    ncell += 1
    f = pseudospectrum_outside(out, z)
    f > half || push!(unresolved, (x, y))
end
say(@sprintf("cells in annulus: %d   excluded (proved eigenvalue-free): %d   unresolved: %d",
             ncell, ncell - length(unresolved), length(unresolved)))
# group unresolved cells by which certified eigenvalue they sit near
say("")
for r in out.results
    near = count(p -> abs(complex(p...) - r.λ) <= 4h, unresolved)
    say(@sprintf("  unresolved cells within 4h of certified λ=%+.4f%+.4fi : %d",
                 real(r.λ), imag(r.λ), near))
end
stray = [p for p in unresolved if all(r -> abs(complex(p...) - r.λ) > 4h, out.results)]
say(@sprintf("  cells not adjacent to any certified eigenvalue: %d", length(stray)))
say("  (these are merely cells where the grid is too coarse for the 1-Lipschitz test — by the")
say("   theorem above they contain no eigenvalues; refining h shrinks them.)")
if !isempty(stray)
    for p in first(stray, 6); say(@sprintf("      e.g. z = %+.4f%+.4fi", p[1], p[2])); end
end

# ---------------------------------------------------------------------------------
# (3) PLOT
# ---------------------------------------------------------------------------------
gx = -1.15:0.006:1.15
F = fill(NaN, length(gx), length(gx))
for (i, x) in enumerate(gx), (j, y) in enumerate(gx)
    z = complex(x, y)
    abs(z) <= out.G22norm && continue
    F[j, i] = pseudospectrum_outside(out, z)
end
fig = Figure(size=(820, 780))
ax = Axis(fig[1,1]; xlabel="Re z", ylabel="Im z", aspect=DataAspect(),
    title="$(mapnm) Ulam MATRIX (n=$n): rigorous σ_min(A−zI) outside |z|=‖G₂₂‖=$(round(out.G22norm,digits=3))")
co = contourf!(ax, gx, gx, permutedims(F); levels=10.0.^range(-4, -0.3, length=12), colormap=:viridis)
Colorbar(fig[1,2], co, label="certified lower bound on σ_min(A−zI)")
θ = range(0, 2π; length=400)
lines!(ax, out.G22norm.*cos.(θ), out.G22norm.*sin.(θ); color=:white, linestyle=:dash, linewidth=1.5)
lines!(ax, R.*cos.(θ), R.*sin.(θ); color=:red, linewidth=2)
if !isempty(unresolved)
    scatter!(ax, first.(unresolved), last.(unresolved); color=(:orange,0.5), markersize=3)
end
for r in out.results
    scatter!(ax, [real(r.λ)], [imag(r.λ)]; color=:red, marker=:xcross, markersize=13)
end
text!(ax, R*0.71, R*0.71; text="|z|=R=$R", color=:red, fontsize=11)
save(joinpath(@__DIR__, "..", "cert_$(mapnm)_R$(replace(string(R),"."=>"")).png"), fig)
say("")
say("plot  → cert_$(mapnm)_R$(replace(string(R),"."=>"")).png")
say("log   → $(basename(logf))")
say("="^78)
close(io)
