# RigorousUlam2D.jl

[![Build Status](https://github.com/orkolorko/RigorousUlam2D.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/orkolorko/RigorousUlam2D.jl/actions/workflows/CI.yml?query=branch%3Amain)

The computations of the paper

> M. J. Pacifico, R. Bioni Liberalquino, I. Nisoli, *Stationary densities of random maps with
> additive BV noise: existence, regularity and certified Ulam approximation*,

Section 6 ("Examples") and Theorem E: for a map `T` of the two-torus with additive uniform noise
on a square of side `ξ`, a computed stationary density `f̃` on the grid of cells of side `δ`
together with a certified upper bound on `‖f_ξ − f̃‖_{L¹}`, where `f_ξ` is the stationary density
of the annealed transfer operator `L_ξ = N_ξ L_T`.  The data are on Harvard Dataverse,
[doi:10.7910/DVN/2OTJ8O](https://doi.org/10.7910/DVN/2OTJ8O).

## The stages

| stage | file | what it computes |
|---|---|---|
| grids | `src/grid.jl` | the uniform dyadic grids and the two orders of their cells |
| deterministic block | `src/transition_graph.jl` | `L_lo ≤ L ≤ L_lo + U` entrywise for the Ulam matrix `L` of `T` (`UlamTree`, `refine!`, `ulam_matrices`, `fibred_area`) |
| maps | `src/maps.jl` | the Hénon map, the Lorenz-like map with contracting fibres and the contracting Lorenz map of Section 6 |
| noisy operator | `src/noisy_operator.jl` | `P_δ = π N_ξ π L π` applied matrix-free, and its ℓ¹ rounding model (`η₁`, `η₁ᴬ`, eq:rounding) |
| powers on `V` | `src/norms_of_powers.jl` | `C_j ≥ ‖P_δ^j|_{V}‖₁` by iterating every column |
| L¹ bounds | `src/l1_error.jl` | the computed density and its residuals, the coarse-fine recursion eq:coarsefine (Proposition "coarsefine"), and the bounds of Proposition "l1error" with `θ_δ = (δ/2)(var₁(ρ_ξ) + var₂(ρ_ξ))` |
| records | `src/experiments.jl`, `src/archive.jl`, `src/deposit.jl` | the stages the drivers are thin over, the names of their files, the archival HDF5 copies and the deposit |

Notation follows the paper: `Var` is the isotropic total variation, `var₁`, `var₂` the variations
in the coordinate directions; the code computes only the latter (`coordinate_variations`), and
`var12ρ` is `var₁(ρ_ξ) + var₂(ρ_ξ) = 4/ξ`.  The grid side `δ` of the paper is `h` in the code and
the noise is given by its half-width `a = ξ/2`.

## The drivers

`PIPELINE` lists the drivers in the order they run and what each writes; every certified number
is written into a [CertifiedRecords](https://github.com/orkolorko/CertifiedRecords.jl) record
under `~/ResearchData/RigorousUlam2D_Data/data`, named after the driver, the commit and the host,
with the machine, the thread counts and the wall-clock time of the run.

```bash
julia --project=. -t 32 bench/tree_deepen.jl                       # MAP, D0, D1, STORE, LEVEL, ...
MAP=lorenz DEPTH=8 A=77//256 julia --project=. -t 16 bench/stationary_density.jl
MAP=lorenz DEPTH=8 A=77//256 julia --project=. -t 16 bench/l1_mixing.jl
MAP=lorenz DEPTH=8 A=77//256 FINE=12 julia --project=. bench/l1_transfer.jl
julia --project=paper paper/figures.jl                             # Table 1 and the figures
```

A column sweep at depth 8 took 2658 s on 16 threads of ibis (l1_mixing.d78ed9d.ibis.tsv), and the
densities of the table are at depth 12, `2^24` cells.  The worked examples, which regenerate each
number of the paper's table step by step and check a small depth against the records, are in [RigorousUlam2DExperiments.jl](https://github.com/orkolorko/RigorousUlam2DExperiments.jl).

## Tests

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

`test/test_records.jl` recomputes, from the code of the commit, the quantities of the records
behind the paper's table that are cheap to recompute, and compares them bit for bit with the
records; it checks nothing on a machine that does not hold the records.  `AUDITS.md` lists the
audits of the code behind the certified numbers.
