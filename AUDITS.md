# Audits

One entry per audit that passed: the commit audited, the date, what was checked, against what,
and how. A change to audited code says in its commit message whether the entry still holds.
Failures are listed under the audit that found them and are not passes.

## 2026-11-23, at ae95e23: the chain behind the L¹ column sweep

By reading the code against the statements below, on nighthawk. Dependencies at
RigorousInvariantMeasures 87aed84, RigorousPseudospectra 268225c, BallArithmetic 19c6924.

Passed:

- `norms_of_powers_columns` (`src/spectral_analysis.jl`): for `v ∈ V_h` and any `u`,
  `P^j v = Σ_i v_i (P^j e_i − u)`, so `max_i ‖P^j e_i − u‖₁ ≥ ‖P^j|_{V_h}‖₁`; every column is
  iterated; `|x − u|`, the sums and the maxima are rounded upward; the error recursion
  `ε_j = η₁‖x_{j−1}‖₁ + η₁ᴬ + ‖P‖₁ ε_{j−1}` is the one for `x_j − P^j e_i`. Since ca2a2c3 the
  maxima are merged under a lock; the depth 6 C_j of the stored records are reproduced bit for
  bit (Lorenz 77/256, Hénon 13/256).
- `noisy_operator` (`src/noisy_operator.jl`), the ℓ¹ model `η1`: with `L_lo ≤ L ≤ L_lo + U`,
  `E = max(L̂ − L_lo, L_lo + U − L̂)` upward, `γ_w` the a priori bound of the sparse product and
  `γ_N` that of the noise, `fl(N fl(L̂x)) − NLx` splits into the rounding of `N`, `N` applied to
  the rounding of `L̂x`, and `N(L̂ − L)x`, which gives
  `(‖E‖₁ + ‖L̂‖₁(γ_w + γ_N(1 + γ_w)))‖x‖₁ + η₁ᴬ` with `‖N‖₁ ≤ 1`. The value of `γ_N` is taken
  from the docstring of `UlamProjectedUniformKernel` in RigorousInvariantMeasures, whose code
  was not audited here.
- `ulam_matrices`, `_read_cell!`, `_area`, `_windows!`, `_range_in`, `_cell_range`,
  `_down`, `_up` (`src/transition_graph.jl`), periodic boundary condition only: the masses
  placed exactly are dyadic and summed exactly; the enclosed fractions are intersected with
  `1 − Σ(others)` and `[0, 1]`, their lower ends summed downward into `L_lo`, their upper ends
  upward, and `U = (lo + uhi) − L_lo` upward; TwoSum gives the directed sums; a window with
  interval endpoints is read between the inner and the outer window.
- `fibred_area` (`src/transition_graph.jl`), after ae95e23: the three stages and the cut at
  `xinv`, given the hypotheses of its docstring (monotone fibres, `xinv` enclosing every
  crossing in the piece).
- `var(ρ_ξ) = 2/a` (`bench/*.jl`): the variation of the paper (§ "The space of functions of
  bounded variation") of `ξ⁻² 1_{[-ξ/2, ξ/2]²}` is `4/ξ` (lem:square_variation), and `ξ = 2a`;
  the value is the same for the Euclidean and the coordinatewise norm of the test field, so it
  does not depend on which one Theorem D is proved for.
- `fixed_point_residual`, `fixed_point_distance`, `sigma_from_norms` (`src/l1_error.jl`):
  against the derivations in the file docstring; `v ≥ 0` is checked since ae95e23, and holds
  for every stored stationary density.

Failed or not auditable:

- The blocks `tree_deepen.*` and `tree_stream.*` in `~/ResearchData/RigorousUlam2D_Data` were
  made by drivers `tree_deepen.jl` and `tree_stream.jl` that are in no commit and on no machine
  (nighthawk, ibis, albatross searched). The maps `F`, the enclosure method, and the `xcomp`,
  `yof`, `yint` and `xinv` passed to `fibred_area` for them cannot be checked.
- `_cut!` before ae95e23 started the last gap at `sup(C2)` after sorting the crossing
  enclosures, which is wrong when `C2` lies inside `C1`; shown on a test case with a lower end
  above the true fraction. Whether the stored `xinv` blocks met that case depends on the lost
  driver's `xinv`; with the single-branch inverses of the tests in `test_transition_graph.jl`
  and grid lines at `x = 0` it does not arise.

Not audited: `UlamTree`, `refine!`, `_children!`, `_targets!`, `_descend!` (the candidate
lists), the reflecting boundary condition, the code of `UlamProjectedUniformKernel`, and the
coarse-fine transfer, which is not yet committed.

## 2026-11-25, at the commit that adds this entry: the rounding of every bound routine

By reading each routine and by tests against 256-bit evaluations (test/test_l1_error.jl,
"rounding: …", "swapped_error against BigFloat", "the constants against BigFloat"), on nighthawk.
RigorousInvariantMeasures at ceae111, CertifiedRecords at 26bd9a4.

Passed, every operation in the direction that keeps a bound:

- `grid_constants` (`src/experiments.jl`): `h` exact (a dyadic side over `2^d`); `var(ρ_ξ) = 2q/p`
  by `div_up` with `2q` exact.  `noisy_ulam_operator`: `normP` by `_max_colsum_up` of
  `add_up.(L_lo, U)`.
- `defect_apriori` `(3h/2) var(ρ_ξ)`, `mul_up`; `coarse_fine_norms`: `(h/2) var(ρ_ξ)` with `h/2`
  exact and `mul_up`, inputs converted to doubles upward, the recursion of RigorousInvariantMeasures
  `norms_of_powers_recursive_noise(a, b, C)` in `⊕₊`, `⊗₊`, and its extension
  `refine_norms_of_powers` in `⊗₊`; checked against the recursion in BigFloat.
- `sigma_from_norms`: the partial sums upward, `1 − C_k` downward as `−sub_up(C_k, 1)`, the
  quotient upward, a bound given above double precision converted upward (this was a
  conversion to nearest before this commit); checked against BigFloat.
- `fixed_point_residual` (`|·|` by `_absdiff_up`, sums `add_up`, `Σf̃` both ways),
  `fixed_point_distance` and `swapped_error` (`1 − ε₂` downward, quotients and sums upward, the
  mass term `ε₂`); checked against BigFloat.
- `grid_variation`: edge differences and their sum upward, divided by `h` upward; checked against
  BigFloat.
- `norms_of_powers_columns`: as in the entry above.
- The records: bounds written by `write_endpoint` (the shortest decimal of the double, exact) and
  read back by `read_endpoint` at 53 bits, converted upward in `l1_transfer` and
  `l1_mixing_merge`.

Not covered by this entry: `noisy_operator`'s `η` and `epsA` (the ℓ² model, used by the spectral
routines and not by the L¹ chain), and the code of RigorousInvariantMeasures and BallArithmetic
beyond the two routines named.
