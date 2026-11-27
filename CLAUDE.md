# CLAUDE.md

Guidance for Claude Code in this repository.

## What this is

The code of the paper in `~/Write/RigorousUlam2D` (`.links`): certified Ulam approximation of the
stationary densities of maps of the two-torus with additive uniform noise, and certified L¹ bounds
on the distance between the computed density and the stationary one.  `README.md` lists the
stages; `PIPELINE` the drivers and what they write; `AUDITS.md` the audits that passed.

## Rules specific to this package

- Every number of the paper comes from a CertifiedRecords record written by a driver in `bench/`
  (or `paper/figures.jl`) at a commit that contains it.  Do not change the arithmetic of
  `src/noisy_operator.jl`, `src/norms_of_powers.jl`, `src/l1_error.jl`, `src/transition_graph.jl`
  or `src/maps.jl` without running `test/test_records.jl` on a machine that holds the records
  (ibis), which recomputes the cheap parts of the records behind the table bit for bit.
- A change to audited code says in its commit message whether the entry of `AUDITS.md` still holds.
- Notation follows the paper: `Var` isotropic variation, `var₁`, `var₂` coordinate variations,
  `var12ρ = var₁(ρ_ξ) + var₂(ρ_ξ)`, `θ_h = (h/2) var12ρ`; the paper's `δ` is `h`, its `ξ` is `2a`.
- Heavy runs (column sweeps, depth-12 blocks and densities) on ibis or albatross, not nighthawk.
- Matrices are column-stochastic, `P[i, j]` the transition from cell `j` to cell `i`; blocks are
  indexed as `build_uniform_partition`, the noisy operator in grid order (`grid_order`).

## Commands

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
julia --project=. test/test_records.jl          # the records check alone (needs the data)
```
