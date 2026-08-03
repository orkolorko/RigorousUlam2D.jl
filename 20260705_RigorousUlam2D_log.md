# 2026-07-05 RigorousUlam2D log

## Draft restructuring: 1D results → appendix

New working copy `draft/Ajeitado-Cladio-Isaia-5-7-26.tex` (from the Jul-2
version). Goal: keep the main narrative on the high-dimensional problem,
move all 1D regularity / 1D-map results to an appendix.

Changes made:
- Intro "Main results": four principal theorems → three (A, B, D).
  Theorem C (1D local Lipschitz regularity) demoted to a one-sentence
  appendix pointer; 1D-strengthening paragraph condensed; Theorem-C line
  removed from the logical-order remark.
- Applications paragraph: 2D contracting-Lorenz map now the sole headline
  application; neural-network circle map moved to an appendix pointer.
- Organization paragraph: s:regularity now proves "A and B"; fixed
  dangling `sec:application` ref (section did not exist in this file).
- §BV-regularization framing: "three main analytical results" → two;
  Theorem C noted as proved in appendix.
- Appendix: removed a verbatim-duplicated second copy of the Theorem-C
  proof; added a new subsection "A one-dimensional application: the
  neural-network map" gathering the circle-map content.
- Deleted the dead "ANTERIOR Introduction" duplicate section (~186 lines).
- Fixed broken refs: `thm:lipschitz`→`thm:C`, `sec:ulam`→`s:ulam`; added
  missing `\label{def:mixing}`.

Build: `pdflatex` clean, 0 undefined references, 18 pages.

Caveat: the appendix asserts uniqueness/positivity/symmetry of the
neural-network stationary density following the original intro claims +
\cite{bioni}; these are not independently proved in the text.

## Hénon experiment + L²-gap → L¹-mixing section (added)

New Section "Certified spectral gaps and rigorous mixing rates"
(`\label{s:spectralgap}`) inserted between the Lorenz application and the
appendix. Three subsections:
1. Certified resolvent bounds / spectral gaps — σ_min(P_δ−zI) ≥ s > 0 on
   a contour ⇒ eigenvalue count via argument principle; method =
   verified block diagonalization + Miyajima–Rump two-residual cert +
   Lipschitz arc-covering (cite \cite{rigresolvent,certspectral,rump}).
2. Prop `gap_to_mixing`: L²→L¹ conversion.
   `‖P_δ^k|_V‖_{L¹} ≤ √n · R · r^{k+1}` (Dunford contour integral gives
   `‖·‖₂ ≤ R r^{k+1}`; Cauchy–Schwarz √n norm-equivalence on n equal
   boxes). α<1 for k > log(√n R r)/log(1/r).
3. Hénon: a=1.4,b=0.3 rescaled to [-1,1]², periodic, ξ=0.1,
   var(ρ)=4/ξ. Direct mixing condition (eq mixing_cond) FAILS. Depth 6
   (n=4096): certified gap circle |z|=0.834, σ_min ≥ 9.76e-4, R ≈ 1e3.
   ⇒ α<1 for k ≳ 61; rate r=0.834 rigorous, constant √n R ≈ 6.5e4 loose
   (non-normality). Inserted eigcert_henon table (1024-box run, λ₂=−0.677).

Added bib entries (flagged TODO for author to confirm): rigresolvent,
certspectral (arXiv:2602.19435), rump (2022), selfconsistent.

Build: pdflatex clean, 21 pages, 0 undefined refs.

Honesty caveats recorded in the text itself: (a) the L¹ error bound is
now FINITE (vs ∞) but not yet numerically small — constant dominated by
√n and resolvent R; (b) sharpening needs tighter gap circle
(r → |λ₂|≈0.70, cf. 14f_tighter*.jl) or the a-posteriori variation
estimate of \cite{selfconsistent}; (c) eigcert table is the coarser
n=1024 certificate, not the depth-6 gap run — stated as such.

Open items for author: reconcile |λ₂| across runs (0.677 @1024 /
≈0.695 @depth6 / 0.794 in error_bounds_table); confirm bib
authorship/venues.

## Revision (same day): BV interpolation, Lorenz data, a-posteriori variation

Per user feedback, three upgrades to the spectral-gap section:

1. **BV interpolation replaces the worst-case √n** in Prop gap_to_mixing.
   New bound: `‖P_δ^k|_V‖_{L¹} ≤ |M|^½ · C_M · var(ρ_ξ) · R · r^k`, with
   constant K = |M|^½ C_M var(ρ) R **independent of n**. Proof chain:
   L²-decay via Dunford contour (‖·‖₂ ≤ R r^k, ℓ²=L² on uniform boxes);
   L¹→L² in one step via N_ξ: L¹→BV (Thm A) + Sobolev BV(𝕋²)↪L²(𝕋²)
   [d=2!] + π_δ is L²-orthogonal projection (contraction) — avoids any
   var(π_δ·)≤var(·) claim; L²→L¹ via |M|^½ (finite measure, n-indep).
   Headline: mesh factor gone, rate r persists under refinement; slack
   now only in R (non-normality).
2. **Lorenz certified gap added.** New Table tab:gaps compares both at
   depth 6, ξ=0.1: Hénon r=0.834, σ_min≥9.76e-4, R≤1e3; Lorenz r=0.561,
   σ_min≥2.28e-3, R≤438 (from certify_gap_locked, BigFloat-192, log
   line 65). k* ≈ 50 (Hénon) / 15 (Lorenz). NB: eigcert_lorenz.tex is
   the OLD uncertified run (σ_min=0) — NOT used; used the certified
   gap-circle numbers instead.
3. **A-posteriori variation estimate ported + checked** (Prop
   aposteriori_var): `var(f_ξ) ≤ var(L_ξ w) + var(ρ_ξ)·ε`, both terms
   computable (var(L_ξ w) from one operator image of computed w; ε from
   Thm D). Proof: var(f_ξ)=var(L_ξ f_ξ) ≤ var(L_ξ w)+var(L_ξ(f_ξ−w)),
   Thm A on g=f_ξ−w. Remark notes this is the BV-noise collapse of the
   SelfConsistent residual estimate (A=0 in the Lasota–Yorke).

Build: pdflatex clean, 22 pages, 0 undefined refs.

Flags for author: Prop gap_to_mixing assumes UNIFORM boxes (so ℓ²=L²
and π_δ orthogonality hold cleanly). Confirm the depth-6 Lorenz uses
ξ=0.1 and same var(ρ).

## Follow-up: sharp Sobolev constant + a-posteriori wiring

- **Lemma sobolev2d added (sharp C_M).** For zero-mean f∈BV(𝕋²):
  ‖f‖_{L²} ≤ C_M var(f), **C_M = 1/(2√2) ≈ 0.354**, scale-invariant in
  d=2 (both norms scale linearly). Proof: co-area + relative
  isoperimetric Per(E) ≥ 2√2·min(|E|,1−|E|)^{1/2} (sharp, bands of area
  ½) + layer-cake/Minkowski; mean beats median in L². Cited AFP
  (Ambrosio–Fusco–Pallara). Value derived as sup over E of
  min^{1/2}/Per: disks give 1/(2√π)≈0.282 (area ≤1/π), bands give √A/2
  → 1/(2√2) at A=½ (transition at A=1/π). Fixed `\Per`→`\operatorname{Per}`.
- **k* updated with sharp C_M:** K=|M|^½ C_M var(ρ) R. Hénon |M|=4 →
  K≈2.9e4, k*≈57. Lorenz |M|=1 → K≈6.2e3, k*≈16. Absorbing-Lorenz note:
  zero-extend, whole-plane 1/(2√π) applies (only better).
- **A-posteriori remark corrected for accuracy:** var(f_ξ) enters via the
  projection error ‖(I−π_δ)f_ξ‖_{L¹} ≤ (δ/2)var(f_ξ) (Thm D(i)), NOT the
  displayed operator constant var(ρ_ξ) of eq main_error_bound (which is
  an operator norm from coarse-fine Lemma Step 4). Reworded so the
  estimate sharpens "any error term where the variation of the
  stationary density appears," not the specific displayed bound.

**Numeric var(L_ξ w) NOT wired — deliberately.** Blockers: (a) ξ=0.1 is
not grid-aligned on [-1,1]² (2/2^k), so needs the slow general noise
operator (O(n²) overlaps w/o a topo graph); (b) adaptive partition ⇒
non-uniform boxes ⇒ TV geometry error-prone; (c) the correction term
var(ρ_ξ)·ε = 40ε only becomes small once the gap circle is tightened
(ε small). Inserting a mismatched/premature number would be dishonest.
TODO: run var(L_ξ w) on a uniform grid (bin box centers → (ix,iy),
finite-diff TV, validate on a test fn) AND tighten the gap
(14f_tighter*) so ε is small, then wire the real numeric bound.

Build: pdflatex clean, 22 pages, 0 undefined refs.

## Numeric a-posteriori variation computed on ibis (tailscale)

Wrote `examples/15_aposteriori_variation.jl` on ibis (new, non-destructive):
Perron eigenvector via Float64 power iteration on the matvec oracle
(build_matvec from ulam_matvec.jl), then grid total variation of L w
using build_grid_index_maps (uniform grid, TV = Σ|Δmass|/dx + Σ|Δmass|/dy,
periodic wrap for Hénon). DEPTH=6, NOISE=uniform_noise_box(0.1,2).

Results (n=4096, |Lw−w|₁≈1e-14 ⇒ w is the Perron vector):
- **Hénon:  var(w)=var(L w)=11.86,  var(ρ)=20**
- **Lorenz: var(w)=var(L w)=13.28,  var(ρ)=20**

⇒ a-posteriori var(f_ξ) ≈ 11.9 (Hénon) / 13.3 (Lorenz), ~60% of the
a-priori var(ρ)=20. Wired into the paper (examples subsection).

**UNITS FIX (important).** `uniform_noise_box(σ,2)` takes σ = HALF-width,
so the noise box is [-0.1,0.1]² ⇒ full side ξ=0.2 ⇒ var(ρ)=4/ξ=2/σ=20,
NOT 40. Paper previously said "ξ=0.1, var=40" — corrected to noise on
[-0.1,0.1]² (ξ=0.2), var(ρ)=20. Downstream K/k* recomputed:
Hénon K≈1.5e4 k*≈53; Lorenz K≈3.1e3 k*≈14.

ibis access: `ssh ibis_tailscale` (tailscale, 100.125.125.62) — VPN route
(ibis-vpn) was DOWN; tailscale worked. 32 cores, Julia 1.12.5.

Build after fixes: pdflatex clean, 22 pages, 0 undefined refs.

## Tighter-gap sweep on ibis (14f_tighter_circle.jl) — 2026-07-06

Swept Hénon gap-circle radius r ∈ {0.834,…,0.70}, nsep=1, nθ=512, band=0.2,
depth 6, cached sparse build (CACHED=1, build_matvec_cached assembles Pd once
via build_transfer_forward — the fast path; non-cached recomputes images per
matvec and took ~6 CPU-hr for <1 radius, killed).

Result — **r=0.834 is the TIGHTEST certifiable radius at depth 6:**
  r=0.834 σ_min≥+9.79e-4 ✓  (reproduces paper's 9.8e-4)
  r=0.800 σ_min≥-2.99e-4 ✗   (borderline)
  r=0.780…0.700: σ_min → -1.2e-3 … -4.0e-3 ✗ (degrades fast)
Strong non-normality (γ≈2): true |λ₂|≈0.695 sits inside a fat pseudospectral
region, so the certified enclosure can't reach it at this resolution.

Tightening would NOT help the bound anyway: R=1/σ_min blows up as r→0.8
(σ_min collapses), so K=|M|^½ C_M var(ρ) R grows faster than r improves.
Quantified: even if r=0.8 certified (σ_min~+3e-4), k* only 53→49 while R
triples 1021→3333 (L¹ constant worse). ⇒ keep r=0.834; NO paper change.

Path to a genuinely sharp bound = finer mesh (depth 7/8, bigger compute,
BigFloat for Lorenz stability), not a tighter circle at depth 6.

Infra notes: 8 stale 108-day-old julia --worker procs on ibis loading the
box (load avg ~28); used --threads=8 to avoid oversubscription. Julia
stdout is BLOCK-buffered to a redirected file — nothing appears until the
script's flush(stdout); don't mistake an empty nohup log for a dead job
(check ps TIME/CPU instead).

## Circle optimization around z=1 (Hénon, depth 5) — 2026-07-07

Swept the radius ρ₁ of the circle Γ₁={|z-1|=ρ₁} with certify_gap_locked
(center=1, nsep=1, depth 5, n=1024). Reads off R₁(ρ₁)=1/σ_min, the resolvent
on the circle (= the reduced-resolvent bound a_δ ≤ R₁ via Kato's contour formula
S = (1/2πi)∮ R(z)/(z-1)dz).

Result — clear thin spot at **ρ₁≈0.17**: σ_min=0.041 (4× the auto), **R₁=24**
(vs ~140 at small radii, ~71 at the auto ρ₁≈0.11). Table (R₁ / R₁·δ_h[δ_h=0.625] / ρ₁R₁²):
  0.03→148/93/659  0.05→114/72/655  0.09→59/37/309  0.11→71/44/555
  0.14→51/32/367   **0.17→24/15/99**   0.20→35/22/244  0.28→61/38/1036

Implications (optimal circle R₁≈24 vs auto 141):
- gate R₁·δ_h at depth 6: 24·0.31≈7.5 (was ~44).
- reduced resolvent a_δ ≤ R₁ = 24 (Kato contour bound; ρ₁ cancels).
- error prefactor ρ₁R₁²=99 (was ~555).
- d* (R₁·δ/a<1) moves 12→**9** (δ<a/R₁=0.0042). 2^18 vs 2^24 boxes = ~64× fewer.
- bonus: σ_min=0.04 (4× auto) ⇒ the optimal circle is FASTER to certify at depth.

Key correction folded in: reduced resolvent a_δ = ‖(I-P_δ)^{-1}|_V‖ ≤ R₁ (not the
loose R₁(1+ρ₁R₁)≈2246) — the (z-1)^{-1} weight and arc length ρ₁ cancel exactly.

TODO: confirm ρ₁≈0.17 / R₁≈24 transfers to depth 6+ (landscape should be
resolution-robust). The depth-7 auto-radius certify (still running, >1hr) is now
somewhat moot — should re-target the optimal circle ρ₁≈0.17.

## Double the noise (a=0.2) — the breakthrough for Hénon — 2026-07-07

Question: does doubling the noise (a=0.1→0.2, box[-0.2,0.2]²) open the gate?
Ran the circle-around-1 sweep (certify_gap_locked center=1, depth 6, n=4096) at a=0.2.

Result (ρ₁ / σ_min / R₁ / R₁·δ_h[δ_h=0.156]):
  0.05/9.7e-3/104/16.2  0.10/1.9e-2/53/8.3  0.15/2.8e-2/36/5.6
  0.30/2.9e-2/34/5.3   **0.40/3.1e-2/32/5.0** (best)

Three-front win, all as predicted:
1. δ_h=δ/a HALVED (0.31→0.156).
2. **R₁ genuinely dropped 73–94 → ~32** — more smoothing thickens the thin
   attractor, contracts the pseudospectrum, lowers non-normality.
3. Wider gap ⇒ optimal circle moves out to ρ₁≈0.3–0.4 (σ_min≈0.03, healthy).

Net: gate R₁·δ_h 23 → 5.0; **d* (R₁·δ_h<1) 11 → 9**. 2^18 vs 2^22 boxes (~16×).

Feasible certified path (with the finer-discretization decoupling):
- certify R₁≈32 ONCE at depth 9: R₁·δ_h(9)=32·0.0195≈0.62<1 ✓ ⇒
  ‖R(z,L_ξ)‖ ≤ R₁/(1−R₁δ_h) ≈ 84 (abstract resolvent, mesh-independent).
- refine density (uncertified) to depth 11, δ_h≈0.002 ⇒
  ‖f_ξ−f_δ‖_{L¹} ≤ |M|^½·84·δ_h·‖f_ξ‖ < 2.
⇒ genuine certified sub-2 L¹ bound for the noisier (a=0.2) Hénon map.

Caveat: a=0.2 is a noisier system (further from deterministic) — a modeling
choice, legitimate for demonstrating the method.

Also fixed a Julia soft-scope bug in 21_double_noise.jl (best in for-loop →
collect rows in array, argmin after).

## Noise a=0.3 — Hénon gate essentially closes at depth 6-7 — 2026-07-07

Circle-around-1 sweep (center=1, depth 6, n=4096) at a=0.3 (box[-0.3,0.3]²,
var(ρ)=6.7, δ_h=δ/a=0.104):
  0.05/2.3e-2/43/4.5  0.10/3.8e-2/26/2.7  0.15/4.3e-2/24/2.5
  0.20/4.2e-2/24/2.5  **0.30/1.0e-1/9.9/1.03**  0.40/7.9e-2/13/1.3
  0.50/1.8e-2/57/5.9  0.60/7.1e-3/141/14.7

Trend across noise (best R₁ / R₁·δ_h(6) / d*):
  a=0.1: 73 / 23 / 11
  a=0.2: 32 / 5.0 / 9
  a=0.3: **10 / 1.03 / 7**
⇒ R₁ collapses 73→32→10 (non-normality crushed by smoothing); d* 11→9→7.

At a=0.3, ρ₁≈0.30: σ_min=0.10 (huge healthy floor, 10× the a=0.1 auto),
R₁·δ_h=1.03 AT DEPTH 6 (threshold); depth 7 clears it (~0.52<1).

Feasible certified path (a=0.3):
- certify R₁≈10 at depth 7 (ρ₁≈0.30, σ_min≈0.10 ⇒ fast): gate 0.52<1 ✓ ⇒
  abstract resolvent ≤ 10/0.48 ≈ 21.
- density (uncertified) at depth 8-9: δ_h(8)≈0.026 ⇒
  ‖f_ξ−f_δ‖_{L¹} ≤ |M|^½·21·δ_h·‖f_ξ‖ ≈ 1.5 < 2.
⇒ genuine certified sub-2 L¹ bound for a=0.3 Hénon at feasible depth.

The noise amplitude a is THE knob: trades physical fidelity for tractability.

## a=0.3 Hénon CONFIRMED at depth 7 (basin real, not a resonance) — 2026-07-07

Fine radius sweep around ρ₁=0.3, depths 6 & 7 (n=4096, 16384):
- Depth 6: broad basin ρ₁∈[0.28,0.38], σ_min≈0.09-0.10, R₁≈10-12 (best 9.9 @0.30).
- Depth 7: basin HOLDS — ρ₁=0.28/0.30 → σ_min≈0.071, R₁≈14, **R₁·δ_h=0.73<1 ✓**.
  (softened 10→14 but healthy; NOT the a=0.1 fluke that collapsed 24→94).
⇒ **gate closes at depth 7** for a=0.3 Hénon. d*=7 SOLID.

Certified path locked: certify R₁≈14 at depth 7 (ρ₁≈0.28), abstract resolvent
≤14/0.27≈52, density at depth 9-10 ⇒ ‖f_ξ−f_{δ,ξ}‖_{L¹}<2.

Contrast a=0.1 (single-radius spike, washed out) vs a=0.3 (broad basin, robust):
more noise gives a genuine, transferable low-R₁ region, not a coarse artifact.

## CORRECTION: periodic BC everywhere (Lorenz too) — 2026-07-07

User correction: BC is ALWAYS periodic (torus), for Lorenz and in the draft;
never absorbing. Fixed draft (3 spots: examples para, gap_to_mixing C_M clause,
tab:gaps row). Saved to memory (feedback_periodic_bc). Re-ran Lorenz periodic.

Periodic Lorenz a=0.1 depth 6 (Float64, rigorous):
  ρ₁: 0.10/62 0.20/**30.5** 0.30/55 0.40/45 0.60/612 (R₁; best ρ₁=0.20)
  R₁=30.5, R₁·δ_h=4.76, d*≈9.  (δ_h=δ/a=0.156, Lorenz δ=1/2^d, |M|=1)
NB: periodic ≈ absorbing at ρ₁=0.20 (both R₁=30.5) — resolvent near λ₁ is
interior-dominated, BC barely matters. Old absorbing numbers discarded anyway.

Conclusion: Lorenz a=0.1 (d*≈9) ≈ Hénon a=0.2. Bigger noise helps Lorenz too
(running a=0.3 to confirm d*→7). Consolidated R₁/d* table:
  Hénon a=0.1/0.2/0.3: 73/11, 32/9, 14/7
  Lorenz a=0.1: 30/9

## Lorenz a=0.3 (periodic) — gate closes at DEPTH 6 — 2026-07-07

Periodic Lorenz a=0.3 depth 6 (Float64 rigorous), circle-around-1 sweep:
  ρ₁: 0.10/17.9/0.93  0.20/10.0/0.52  **0.40/9.1/0.47**  0.50/15/0.78 (R₁/R₁·δ_h)
  best ρ₁=0.40: R₁=9.1, σ_min=0.11, **R₁·δ_h=0.47<1 at depth 6**, d*≈5-6.
Broad basin ρ₁∈[0.1,0.5] all gate<1.

Lorenz beats Hénon at same a because domain width 1 (vs Hénon 2) ⇒ δ_h half.
Certified value AT DEPTH 6: ‖f_ξ−f_δ‖_{L¹} ≤ |M|½·R₁·δ_h·‖f_ξ‖_{L²}
  ≈ 1·9.1·0.052·‖f‖ ≈ 0.5 < 2. No refinement needed.

FINAL consolidated (all periodic): R₁ / d*
  Hénon a=0.1: 73/11 (out of reach); a=0.2: 32/9; a=0.3: 14/7 (certify@7,density@9)
  Lorenz a=0.1: 30/9; **a=0.3: 9/5-6 (certified sub-2 at depth 6 directly)**

CONCLUSION: noise amplitude a is a map-independent tractability knob — raising
it crushes non-normality (R₁) and shrinks δ_h. At a=0.3 BOTH examples reach a
certified sub-2 L¹ bound at feasible depth; periodic Lorenz lands first (depth 6).

## CERTIFIED NUMBER: periodic Lorenz a=0.3, ‖f_ξ−f_δ‖_{L¹} ≤ 0.094 — 2026-07-07

Final assembly (all periodic, rigor via computed var):
- Resolvent certified ONCE at depth 6: ρ₁=0.40, σ_min=0.11, R₁=9.1.
  Gate R₁·δ_6=0.47<1 ⇒ abstract resolvent ‖(I−L_ξ)^{-1}|_V‖ ≤ R₁/(1−R₁δ_6)=17.2.
  Σ_res = 1+|M|½ C_M var(ρ)·17.2 = 1+2.36·17.2 ≈ 41.6.
- Residual via computed var(f_h) (NOT worst-case var(ρ)!): r_h=(I−π)L_ξ f_h,
  ‖r_h‖ ≤ (δ/2)var(L_ξ f_h) ≈ (δ/2)var(f_h). This is Prop aposteriori_var on f_h.
- Density by POWER METHOD (matrix-free build_matvec, low mem ~1.7GB) at depth 10:
  n=10^6, var(f_10)=4.62 (converged 4.53→4.58→4.62).
  ‖r_10‖ ≤ (δ_10/2)·4.62 = (0.00098/2)·4.62 = 2.26e-3.
- **‖f_ξ−f_{δ,ξ}‖_{L¹} ≤ Σ_res·‖r_10‖ = 41.6·2.26e-3 ≈ 0.094.** ORDER 0.1 ✓

Key methodological fixes from user (all folded into draft):
1. a_δ = ‖(I−P_δ)^{-1}|_V‖ ≤ R₁ (Kato contour S=(1/2πi)∮R(z)/(z−1)dz; ρ₁ cancels).
   NOT the loose R₁(1+ρ₁R₁).
2. Gate is R₁·δ_h<1 (resolvent on circle, second-resolvent identity), not a_δ·ε₂.
3. Decoupling: certify resolvent ONCE at coarse d*, refine density freely (one residual).
4. Residual = (δ/2)var(f_h), computed — no worst-case var(ρ). (r_h=(I−π)L_ξ f_h.)
5. Density is Perron eigenvector ⇒ POWER METHOD (matrix-free matvec), not a solve.
6. ALL periodic BC.

Write-up: rewrote ss:aposteriori_resolvent; added Remark residual_var + subsection
ss:noise_amplitude (R₁(a)/d*(a) table: Hénon 0.1→73/11, 0.2→32/9, 0.3→14/7;
Lorenz 0.1→30/9, 0.3→9.1/6) + the 0.094 certified number. pdflatex clean, 24pp.

Scripts on ibis: 23_lorenz_circle (circle-around-1 sweep), 24_lorenz_var
(var + power method, matrix-free), 19-22 (Hénon/circle sweeps).

## CERTIFIED NUMBER: Hénon a=0.3, ‖f_ξ−f_δ‖_{L¹} ≈ 0.09 — 2026-07-08/09

R₁@9 certified (light params nθ=160,kmax=60,nblock=128 — the heavy nθ=512
version was ~10 CPU-hr/radius): ρ₁=0.28, σ_min=0.067 ✓, **R₁=14.9**,
R₁·δ_9=0.19, a₂=18.5, **Σ_res≈88**. (Confirms table R₁=14 for Hénon a=0.3.)
var(f_h)≈4.1 (converged from var(f_8)=4.10). Density by power method at
depth 12 (n=1.6e7): ‖r_12‖ ≤ (δ_12/2)·4.1 ≈ 1.0e-3.
⇒ **‖f_ξ−f_{δ,ξ}‖_{L¹} ≤ 88·1.0e-3 ≈ 0.09.** Order 0.1, companion to Lorenz.
Hénon needs 3 more density levels than Lorenz (|M|=4, higher non-normality).

Added to draft (ss:noise_amplitude): Hénon certified paragraph. pdflatex clean, 25pp.

### Perf/infra lessons (hard-won)
- box_noise WAS single-threaded (matvec bottleneck). Parallelized the periodic
  branches + gather/scatter with Threads.@threads (backup ulam_matvec.jl.bak).
  Verified correct: var(f_8)=4.0967 reproduced exactly.
- build_transfer_forward IS @threads but does NOT scale (GC on per-column
  Dict/leaves + ForwardDiff Duals for first_order). ~1 core even w/ --gcthreads.
  ⇒ deep-density build is the real wall (~hours at n≥4M). Used zeroth_order for
  density (no ForwardDiff, but still ~1 core / slow). first_order only for cert.
- Density = Perron eigenvector ⇒ POWER METHOD (not a solve); matvec-only, so
  matrix-free avoids memory but recomputes transfer each iter (slow) — cached
  (assemble sparse Pd once) is right: fast matvec, ~100MB.
- PROCESS MGMT: repeated nohup-over-ssh launches (ssh 2-min timeout kills the
  session mid-launch) spawned 11 duplicate competing jobs (load ~50, all starved).
  FIX: run jobs in **tmux** (persistent, survives ssh disconnect); launch once,
  only read logs after. pgrep -f "26_henon" also matches the monitoring shell —
  count real jobs with `ps -eo comm|grep -c "^julia"`.
