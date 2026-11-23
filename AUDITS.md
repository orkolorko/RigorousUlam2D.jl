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
