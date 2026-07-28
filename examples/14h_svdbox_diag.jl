# Diagnostic: is the certify_gap vs certify_gap_locked difference a bug, or the
# looseness of the verified SVD (svdbox) on the FULL non-normal Arnoldi block?
# Compare, at z on the gap circle:
#   (1) svdbox σ_min LOWER bound of the full 256×256  H − zI   (what certify_gap uses for σ11)
#   (2) TRUE σ_min = svdvals(H − zI)[end]                       (Float64 reference)
#   (3) svdbox σ_min lower of a SMALL near-circle band block    (what certify_gap_locked uses)
#   CACHED=1 DEPTH=6 julia --project=examples examples/14h_svdbox_diag.jl
import Pkg; Pkg.activate(@__DIR__)
include(joinpath(@__DIR__, "ulam_matvec.jl"))
using RigorousPseudospectra, Printf, LinearAlgebra
const RPS = RigorousPseudospectra
build = get(ENV,"CACHED","")=="1" ? build_matvec_cached : build_matvec
NOISE = uniform_noise_box(0.1,2)
DEPTH = get(ENV,"DEPTH","")=="" ? 6 : parse(Int, ENV["DEPTH"])
d = IntervalBox(interval(-1.0,1.0), interval(-1.0,1.0))
L,La,n,_ = build(henon_rescaled, d, NOISE, DEPTH; periodic=true, enclosure_method=first_order_enclosure)

nblock = 256
H, V, βN = RPS._arnoldi(L, n, min(nblock, n-2))
Hc = ComplexF64.(H)
mr = RPS._certify_block(Hc, 6)
λ = mr.centres
ord = sortperm(abs.(λ); rev=true)
r = 0.834
# near-circle band (as certify_gap_locked builds it), band=0.25
kk = [i for i in eachindex(λ) if abs(abs(λ[i]) - r) <= 0.25*r]
Hk = Matrix(RPS._captured_block(L, RPS._frame(V*mr.W[:,kk])))   # small band block, m×m
@printf("n=%d nblock=%d  band size m=%d  ‖H‖=%.3f  |λ1|=%.4f |λ2|=%.4f\n",
        n, size(Hc,1), size(Hk,1), opnorm(Hc,2), abs(λ[ord[1]]), abs(λ[ord[2]]))
println("z on circle |z|=$r :   σ11_full(svdbox lower) | true σ_min(full) | σ11_band(svdbox lower) | true σ_min(band)")
for t in 0:7
    z = r*cis(2π*t/8)
    full_lo = RPS._sigma_min_cert(Hc - z*I)                 # svdbox lower — certify_gap's σ11
    full_true = minimum(svdvals(Hc - z*I))                  # Float64 reference
    band_lo = RPS._sigma_min_cert(Hk - z*I)                 # svdbox lower — locked's σ11
    band_true = minimum(svdvals(Hk - z*I))
    @printf("  θ=%4.0f°   %+.4e        %+.4e        %+.4e        %+.4e\n",
            360t/8, full_lo, full_true, band_lo, band_true)
    flush(stdout)
end
