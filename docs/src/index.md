# RigorousUlam2D.jl

```@meta
CurrentModule = RigorousUlam2D
```

The computations of Section 6 ("Examples") and Theorem E of Pacifico, Bioni Liberalquino, Nisoli,
*Stationary densities of random maps with additive BV noise: existence, regularity and certified
Ulam approximation*: for a map of the two-torus with additive uniform noise, a computed stationary
density on a uniform grid and a certified upper bound on its L¹ distance from the stationary density.

```@docs
RigorousUlam2D
```

The stages, in the order a run goes through them, are listed in the [API reference](api.md); the
drivers that chain them and write the records are in `bench/` and `paper/` of the repository,
and the worked examples in RigorousUlam2DExperiments.jl.
