# RigorousUlam2D experiment log — 2026-06-30

## Certified spectrum + stationary measure, depth 6 (ibis, 32 cores)

`examples/14_spectrum_and_measure.jl`, `CACHED=1 DEPTH=6 NEIG=10 NSEP=1 NBLOCK=28 BITS=192`.
Box noise `uniform_noise_box(0.1, 2)`, n = 4096 (2^6 × 2^6 cells).
Outputs: `examples/14_henon.png`, `examples/14_lorenz.png`.
VBD via BallArithmetic `schur_newton_vbd` with the warmup-then-merge fix (commit e7ff47e).

### Hénon (Float64)
- 10 leading eigenvalues all certified isolated (✓). Leading λ=1.00000, ρ=0.105,
  σ_min ≥ 7.07e-3. Discs shrink with modulus down to ρ≈2e-3 for |λ|≈0.38.
- Conjugate pairs resolved (0.4515±0.5290i, −0.1473±0.4948i, 0.0148±0.4018i, etc.).
- **GAP |z|=0.834 (nsep=1): σ_min ≥ −3.75e-2 ✗ (uncertified).** Block-assembly floor
  goes negative — the hard non-normal mixing gap, as expected.
- Stationary measure (λ≈1 eigenvector) reproduces the Hénon attractor density.

### Contracting Lorenz (BigFloat-192)
- κ₂=3027, β_N=4.36e-2. 10 leading eigenvalues all certified isolated (✓).
  Leading λ=0.99881, ρ=0.126, σ_min ≥ 8.95e-3. Discs down to ρ≈4e-5 / σ_min≈9e-10.
- **GAP |z|=0.561 (nsep=1): σ_min ≥ −2.43e-2 ✗ (uncertified).** Same story.
- Stationary measure shows the two-band contracting-Lorenz structure.

### Takeaways / caveats
- Eigenvalue isolation is solid at depth 6 for both examples (the warmup-then-merge
  VBD fix holds at n=4096). This is the deliverable that certifies cleanly.
- The **gap / exclusion circle remains uncertified** for both — block-assembly
  `certify_gap` floor is negative. This is the open non-normal mixing-gap problem
  (‖A‖_op≈1.6 transient, pseudospectral gap weakens under refinement). Reported
  honestly; the red gap circle in both figures is labeled "uncertified".
- Lorenz BigFloat-192 leg dominates wall time (~2.5 h vs Hénon ~minutes).

## Recovered + reimplemented locked gap certificate (certify_gap_locked), Hénon depth 6

Recovered the lost "singular-direction lock" gap certificate (was prototyped,
tested, and discarded via `git checkout` last session) → reimplemented as
`certify_gap_locked` in RigorousPseudospectra (branch feat/hkk-locking, commit
892bbbb). Captures the near-circle band H_kk (its right singular vectors of
(z−H_kk) attain σ_min) as a lock W; σ₁₁ = exact small svdbox(z−H_kk); σ₂₂ =
far_floor on the deflated complement (SOUND — the lock removes the only small
singular value, leaving a well-separated bottom). Assemble (★) + arc-cover.

Hénon depth 6 (n=4096), kmax=120, band=0.2:
- **nsep=1, circle |z|=0.834: CERTIFIED σ_min ≥ +9.76e-4 ✓ at nθ=512** (pointwise
  floor ~0.006, σ₁₁=0.130, σ₂₂=0.095, γ=2.21, R₂≈5e-7). Committed block-assembly
  gave −0.037 ✗ on the same circle. Floor vs nθ: 64→−0.035, 128→−0.014,
  256→−0.0041, 512→+0.00098 (arc-cover 2r·sin(π/2nθ) shrinks; pointwise floor stable).
- **nsep=10, r=0.38: NOT certifiable — no annular gap.** Circle threads the dense
  eigenvalue cloud (σ₁₁≈9e-4, true σ_min~1e-3); assembly throttled to ~5e-6 by
  γ≈3.5. Widening band fixes σ₂₂ (0.4→0.02) but doesn't create a gap that isn't there.

Caveats: (1) the (★) assembly loses ~20× to the coupling γ (non-normal transient
‖A‖~1.6) — floor 0.006 vs min(σ₁₁,σ₂₂)=0.095; reducing γ is the looseness lever.
(2) nθ=512 uniform is wasteful; the floor is coupling-limited (~uniform 0.006), so
adaptive pearls help less here than where the floor varies. (3) Soundness of the
locked complement rests on the lock capturing the σ_min direction (the band must be
wide enough); verified empirically, R₂ clean.

## Certified figures with certify_gap_locked, depth 6 (ibis) — 2026-07-01/02

Re-ran example 14 with the LOCKED gap certificate (RPS feat/hkk-locking @892bbbb,
provenance-stamped). NTHETA=512 both examples (BigFloat 256-cap was too coarse for
depth-6 radius → re-ran Lorenz at 512).
- **Hénon** (Float64): GAP |z|=0.834 (nsep=1, m=4) σ_min ≥ 9.76e-4 ✓, resolvent ≤ 1024. 10/10 isolated.
- **Lorenz** (BigFloat-192): GAP |z|=0.561 (nsep=1, m=1) σ_min ≥ 2.28e-3 ✓, resolvent ≤ 438. 10/10 isolated.
Both figures (14_henon.png, 14_lorenz.png) now show a GREEN certified gap circle.
Mixing-rate reading (Hénon): sub-leading spectral radius ≤ 0.834 ⇒ L¹ decay ≤ C·0.834ⁿ,
C_L1 ≈ √N·resolvent·r ≈ 5.5e4 (rate rigorous, constant loose from non-normality + √N).
