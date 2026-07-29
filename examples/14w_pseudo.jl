# Rigorous pseudospectra of the Hénon transfer operator OUTSIDE the far circle, from the
# consolidated pipeline: σ_min surface = svdbox(G11 − zI) (the FAITHFUL, κ-free block, ≈σ_min(A−zI)),
# masked inside |z| ≤ ‖G₂₂‖; overlaid with the Miyajima–Rump certified eigenvalue discs and the
# certified isolation circles. The surface heatmap uses Float64 svdvals (fast candidate, = the
# rigorous svdbox to ~1e-14); the far circle, discs and isolation circles are the certified objects.
#   CACHED=1 DEPTH=6 julia --project=examples examples/14w_pseudo.jl
import Pkg; Pkg.activate(@__DIR__)
include(joinpath(@__DIR__, "ulam_matvec.jl"))
using RigorousPseudospectra, CairoMakie, Printf, LinearAlgebra
build = get(ENV,"CACHED","")=="1" ? build_matvec_cached : build_matvec
NOISE = uniform_noise_box(0.1,2)
d = IntervalBox(interval(-1.0,1.0), interval(-1.0,1.0))
L,La,n,_ = build(henon_rescaled, d, NOISE, 6; periodic=true, enclosure_method=first_order_enclosure)
out = certify_leading_eigs_sing(L, La, n; neig=12, nblock_eig=60, margin=0.8)
G11 = out.G11; G22n = out.G22norm
@printf("dim(Q)=%d ‖G₂₂‖=%.4f — plotting σ_min(G₁₁−zI) outside |z|=‖G₂₂‖\n", out.dimQ, G22n)
for r in out.results
    @printf("  |λ|=%.4f: tight enclosure r=%.3e  |  isolation r=%.4f  |  M-R disc=%.3e\n",
        abs(r.λ), r.cert.enclosure, r.cert.radius, r.disc)
end
flush(stdout)

# σ_min surface (Float64 svdvals of the faithful block), masked inside the far circle
N = 220
xs = range(-1.15, 1.15; length=N); ys = range(-1.15, 1.15; length=N)
S = fill(NaN, N, N)
γ = out.γ
assemble(a,b) = (a<=0||b<=0) ? 0.0 : a*b/(a+b+γ)
for (j,y) in enumerate(ys), (i,x) in enumerate(xs)
    z = complex(x,y)
    abs(z) <= G22n && continue                 # inside the far circle: not covered by this bound
    # CERTIFIED lower bound on σ_min(A−zI): assemble(σ_min(G₁₁−zI), |z|−‖G₂₂‖, γ) — its ε-contour is
    # a rigorous OUTER enclosure of the ε-pseudospectrum of A. (σ_min(G₁₁) via Float64 svdvals =
    # the rigorous svdbox to ~1e-14; the coupling γ and the deep far block make it a bound on A.)
    S[i,j] = assemble(minimum(svdvals(G11 - z*I)), abs(z) - G22n)
end
logS = map(v -> isnan(v) ? NaN : log10(max(v, 1e-6)), S)

fig = Figure(size=(760,720))
ax = Axis(fig[1,1]; xlabel="Re", ylabel="Im", title="Hénon — certified σ_min(A−zI) bound, A = Ulam MATRIX (no BV→L¹ error)",
    aspect=DataAspect())
hm = heatmap!(ax, xs, ys, logS; colormap=:viridis)
contour!(ax, xs, ys, logS; levels=log10.([3e-3,1e-2,3e-2,1e-1,3e-1]), color=:white, linewidth=0.7)
Colorbar(fig[1,2], hm; label="log₁₀ certified σ_min(A−zI) bound")
θu = range(0,2π; length=300)
lines!(ax, cos.(θu), sin.(θu); color=(:gray,0.4), linestyle=:dash)                     # unit circle
lines!(ax, G22n.*cos.(θu), G22n.*sin.(θu); color=:orange, linewidth=2.5,
    label="far circle |z|=‖G₂₂‖=$(round(G22n,digits=3))")                              # far circle
for r in out.results
    c = r.cert
    # ISOLATION circle (max-floor radius): σ_min(A−zI)>0 ⇒ exactly one eigenvalue of A inside.
    # Large = biggest certification margin, not a localization error. Faint, dashed.
    lines!(ax, real(c.center).+c.radius.*cos.(θu), imag(c.center).+c.radius.*sin.(θu);
        color=(:white,0.5), linestyle=:dash, linewidth=0.8)
    # TIGHT resolvent enclosure of A's eigenvalue (smallest radius still certifying floor>0).
    lines!(ax, real(c.center).+c.enclosure.*cos.(θu), imag(c.center).+c.enclosure.*sin.(θu);
        color=:crimson, linewidth=2.2)
    scatter!(ax, [real(r.λ)],[imag(r.λ)]; color=:black, marker=:xcross, markersize=10)
end
lines!(ax, [10.0],[10.0]; color=(:white,0.8), linestyle=:dash, linewidth=1.2, label="isolation circle (max margin)")
lines!(ax, [10.0],[10.0]; color=:crimson, linewidth=2.2, label="tight resolvent enclosure of λ (A)")
limits!(ax, -1.15, 1.15, -1.15, 1.15)
axislegend(ax; position=:lt, framevisible=false, labelsize=9)
save(joinpath(@__DIR__,"14_henon_pseudo.png"), fig)
println("saved 14_henon_pseudo.png")
