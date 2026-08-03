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
