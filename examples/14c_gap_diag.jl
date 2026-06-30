# Diagnostic: at nsep=10 (circle radius below all 10 leading eigs), report per-sample
# σ₁₁ = svdbox(H−zI) and σ₂₂ = σ_min(z−G₂₂) SEPARATELY around the circle, plus ‖G₂₂‖, γ, R2.
# Tells us WHICH term kills the floor: captured block (no annular gap) or far block (needs lock).
#   CACHED=1 DEPTH=6 julia --project=examples --threads=auto examples/14c_gap_diag.jl
import Pkg
Pkg.activate(@__DIR__)
include(joinpath(@__DIR__, "ulam_matvec.jl"))
using RigorousPseudospectra, BallArithmetic, LinearAlgebra, Printf
import RigorousPseudospectra: _arnoldi, _certify_block, _coupling_cert, _sigma_min_cert,
    far_block_sigma_min, far_floor, lanczos_pro

DEPTH = get(ENV, "DEPTH", "") == "" ? 6 : parse(Int, ENV["DEPTH"])
NBLOCK = get(ENV, "NBLOCK", "") == "" ? 256 : parse(Int, ENV["NBLOCK"])
NSEP = get(ENV, "NSEP", "") == "" ? 10 : parse(Int, ENV["NSEP"])
build = get(ENV, "CACHED", "") == "1" ? build_matvec_cached : build_matvec
NOISE = uniform_noise_box(0.1, 2)

d = IntervalBox(interval(-1.0, 1.0), interval(-1.0, 1.0))
L, La, n, _ = build(henon_rescaled, d, NOISE, DEPTH; periodic = true, enclosure_method = first_order_enclosure)

H, V, βN = _arnoldi(L, n, min(NBLOCK, n - 2))
Hc = ComplexF64.(H)
mr = _certify_block(Hc, 6)
λ = mr.centres; ord = sortperm(abs.(λ); rev = true)
r = sqrt(abs(λ[ord[NSEP]]) * abs(λ[ord[min(NSEP + 1, length(ord))]]))
γ = _coupling_cert(L, La, V)

# ‖G₂₂‖ (far block beyond the whole Krylov basis V)
Pv(x) = x .- V * (V' * x)
GtG(x) = Pv(ComplexF64.(La(Pv(ComplexF64.(L(Pv(x)))))))
Lg = lanczos_pro(GtG, n, min(40, n - 2); v0 = Pv(randn(ComplexF64, n)), reorth = true)
_, _, R1g, R2g = far_floor(Lg)
θmaxg = maximum(eigvals(SymTridiagonal(Float64.(Lg.α), Float64.(Lg.β[1:end-1]))))
G22n = R2g < 1 ? sqrt(max(0.0, θmaxg + R1g / (1 - R2g))) : Inf

@printf("nsep=%d  radius=%.5f  |λ_nsep|=%.5f |λ_{nsep+1}|=%.5f  γ=%.3e  ‖G₂₂‖=%.4f\n",
    NSEP, r, abs(λ[ord[NSEP]]), abs(λ[ord[NSEP+1]]), γ, G22n)
println("  θ/2π     σ₁₁(svdbox H−z)   σ₂₂(z−G₂₂)   R2far")
s11min = Inf; s22min = Inf
for t in 0:15
    z = r * cis(2π * t / 16)
    σ11 = _sigma_min_cert(Hc - z * I)
    σ22, info = far_block_sigma_min(L, La, n, V, z; prec = Float64)
    global s11min = min(s11min, σ11); global s22min = min(s22min, σ22)
    @printf("  %.4f   %.6e     %+.6e   %.2e\n", t/16, σ11, σ22, info.R2)
end
@printf("MIN over circle:  σ₁₁=%.6e   σ₂₂=%.6e\n", s11min, s22min)
