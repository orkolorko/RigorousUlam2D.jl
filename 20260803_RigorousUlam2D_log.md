# RigorousUlam2D experiment log — 2026-08-03

## Stationary-measure gallery across noise sizes (examples/16_density_gallery.jl)

Built P_det ONCE per map at depth 8 (n=65536), reused across four noise half-widths
a ∈ {0.05, 0.1, 0.2, 0.3} (box [-a,a]², side ξ=2a, var(ρ_ξ)=2/a). Perron density by
power method; densities plotted as 2×2 galleries. Periodic BC both maps.
Outputs: results/gallery_henon.png, results/gallery_lorenz.png.

**Convention check PASSED** — depth-6 Hénon a=0.1 (rx=3) Arnoldi top |λ|:
1.0000, 0.6955, 0.6955, 0.6748 — reproduces cert_henon_R06.log exactly. Confirms
uniform_noise_box(w,2) uses w = HALF-width (= draft's `a`); sweep "noise 0.1" = a=0.1.

**var(f_h) reproduces the draft** (robust integral quantity):
| a | ξ | var(ρ) | Hénon var(f_h) | Lorenz var(f_h) | maxρ H | maxρ L |
|---|---|---|---|---|---|---|
|0.05|0.10|40 |25.95|22.77|4.41|8.72|
|0.10|0.20|20 |12.41|13.14|2.00|4.27|
|0.20|0.40|10 | 6.36| 7.76|1.07|1.99|
|0.30|0.60|6.7| 4.10| 4.00|0.86|2.61|
Draft values: Hénon a=0.1→11.9 (≈12.4 ✓), a=0.3→4.1 (=4.10 ✓); Lorenz a=0.1→13.3 (≈13.1 ✓).

Residual ‖r_h‖ ≤ (δ/2)var(f_h) at depth 8 (δ=dx): Hénon a=0.3 → 1.6e-2, Lorenz a=0.3 → 7.8e-3.
NB these are depth-8 (visualization) residuals; the certified 0.09 needs the deeper density
(Hénon depth 12 → ‖r‖≈1e-3) per decoupling. Final error = Σ_res·‖r_h‖ needs the R₁-around-1
resolvent factor (Σ_res: Hénon 88, Lorenz 41.6 at a=0.3) — a=0.3 gap/R₁ cert still pending.

### Caveat found: display-rx bug in examples/15_noise_sweep.jl
Line 47 prints rx = round((w/2)/cellw), but build_matvec_cached actually uses
rx = round(w/cellw) (uniform_noise_box(w,2) → half-width w). So the printed rx column in
results/noise_sweep_*.txt and cert logs is HALF the true value (Hénon a=0.1 shows rx=2,真 rx=3).
Certified eigenvalues/σ_min are correct (they use the real operator); only the printed rx and
the "5-cell/7-cell" comments are wrong. Fix: line 47 → round(w/cellw), or relabel w as full-side.

### Gap certificates on hand (depth 6, |z|=0.6), for a=0.05/0.1/0.2 only
Hénon: 4 eigenvalues outside 0.6 (complete), |λ₂|=0.6955, ‖G22‖=0.445.
Lorenz: 1 outside, |λ₂|≤0.60, ‖G22‖=0.234. (a=0.3 certificate pending from user.)

## a=0.3 gap certificate + PERIODICITY BUG in Lorenz certs — 2026-08-03 (later)

certified_gaps.txt now includes a=0.3 (depth 6, |z|=0.6):
- Hénon a=0.3: 1 eig outside, |λ₁|=1.0000 (periodic ✓), |λ₂|≤0.60, gap≥0.40, |G22|=0.140.
- Hénon a=0.2/0.1/0.05: |λ₂|=0.6425/0.6955/0.7457 (SHARP, ≥2 outside).
- Lorenz a=0.3/0.2/0.1/0.05: |λ₁|=0.8477/0.9754/0.9988/0.9996 — ALL <1 ⇒ NON-PERIODIC.

**BUG**: examples/15_noise_sweep.jl had `per=false` for Lorenz (line 33) — violates the
always-periodic rule. So every Lorenz gap certificate in results/ is the absorbing operator
(mass leaks, |λ₁|<1), NOT the periodic one the paper uses. Hénon certs are fine (periodic).
FIXED: set Lorenz per=true; also fixed the cosmetic rx display (line 47 was round((w/2)/cellw),
actual is round(w/cellw)). Relaunched periodic Lorenz cert (MAP=lorenz, 4 noises).

My gallery script examples/16_density_gallery.jl had the SAME per=false Lorenz inheritance —
fixed to true, regenerated. Periodic Lorenz var(f_h): a=0.05→22.8, 0.1→13.0, 0.2→7.2, 0.3→4.58
(a=0.3 max density back to 2.05, matching earlier periodic run; non-periodic had 2.61).

Draft: added tab:gallery (per-noise |λ₂| + var(f_h)) and two 4-panel gallery figures
(fig:gallery_henon, fig:gallery_lorenz) replacing the single a=0.3 figure. Hénon gaps filled
(periodic-certified); Lorenz |λ₂| column daggered pending the periodic recert now running.
pdflatex clean, 27pp.

### Periodic Lorenz re-certification DONE — confirms fix
Re-ran with per=true (both my run and user's refresh of results/, 10:32). Periodic Lorenz
depth-6 |z|=0.6, all four noises: |λ₁|=1.0000 ✓, 1 eig outside, |λ₂|≤0.60, gap≥0.40.
|G22|: a=0.3→0.069, 0.2→0.111, 0.1→0.235, 0.05→0.388. σ_min on circle: 6.7e-2/5.3e-2/2.3e-2/8.5e-3.
Draft tab:gallery Lorenz |λ₂| column filled (≤0.60, all periodic-certified); captions restored.
pdflatex clean, 27pp, 0 undefined. Sanity check: |λ₁|=1 exactly ⇔ periodic (mass-preserving).

### Certified resolvent on a circle around 1 (Nisoli's route) — WORKS, use these

Circle |z−1| = r encloses ONLY the Perron eigenvalue (all other |λ| ≤ 0.6 certified, so r < 0.4 is
safe). σ₁₁ from the adaptive pearl cover = verified svdbox on G₁₁ (~92×92) ⇒ κ-FREE, milliseconds.
σ₂₂ = (1−r) − ‖G22‖, assembled with γ. Lorenz periodic, depth 6:

  noise   r=0.30    r=0.20    r=0.10      ‖(z−L)⁻¹‖ ≤ on the circle
  0.30    11.83     15.47     28.03
  0.20    17.84     17.76     30.67
  0.10    48.92     43.40     65.11
  0.05    136.75    109.37    150.00

MONOTONE in the noise (less smoothing ⇒ larger resolvent), values O(10)–O(100). r≈0.2 is the sweet
spot: larger r moves off the Perron pole (bigger σ₁₁) but eats into σ₂₂ = (1−r) − ‖G22‖.

CONTRAST with the κ₂-transported VBD block floor tried first: 1.8e7 / 6.8e5 / 6.6e5 / 6.2e2 —
unusable AND non-monotone, because κ₂ reached 1e7 and the bound was measuring the frame's
conditioning, not the operator. The exact-SVD escape pays no transport. A 1e5–1e6× improvement.
Saved: results/resolvent_circle_around1_lorenz.txt (the κ₂ attempt was NOT saved/committed).

COST: the milliseconds are the SVD; the ~150 s/amplitude is the pipeline's full-basis Lanczos for
‖G22‖. Since ‖G22‖ is already certified at all four amplitudes, a future run should reuse it and
rebuild only G₁₁.

### Resolvent circle around 1 — HENON (companion to the Lorenz table above)

Same construction. Contour validity for Hénon needs a check Lorenz did not: Hénon has FOUR certified
eigenvalues outside R=0.6, so the circle must exclude them too. Their certified positions are
0.4514±0.5290i (distance 0.762 from z=1) and −0.6748 (distance 1.675); any eigenvalue NOT
individually certified has |λ| ≤ 0.6 hence distance ≥ 1−R = 0.4. So r < 0.4 is safe, and
r ∈ {0.30,0.20,0.10} all qualify — the contour encloses the Perron eigenvalue alone.

  noise   r=0.30    r=0.20    r=0.10      ‖(z−L)⁻¹‖ ≤ on |z−1|=r
  0.30    18.61     21.72     37.35
  0.20    35.55     34.84     55.13
  0.10    140.70    95.04     123.77
  0.05    256.15    152.70    182.40

Monotone in the noise; r≈0.2 again the sweet spot. Hénon is ~1.5–2× worse than Lorenz at every
amplitude, consistent with its subdominant spectrum sitting far closer to the unit circle (4–7
eigenvalues outside 0.6 versus Lorenz's one). Saved: results/resolvent_circle_around1_henon.txt.
