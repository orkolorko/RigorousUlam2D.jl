# API Reference

```@meta
CurrentModule = RigorousUlam2D
```

## Enclosure Methods

Functions for computing rigorous interval enclosures of function images.

See the **Enclosure Methods** page in the User Guide for detailed documentation.

## Partition Utilities

Functions for managing and manipulating interval boxes and partitions.

```@docs
bisect_interval
bisect_box
quadrisect
issubset_box
isdisjoint_box
intersects
is_empty_box
box_measure
box_width
box_radius
```

## Domain Restriction

Optimization functions for computing on effective domains.

See the **Domain Restriction** page in the User Guide for detailed documentation.

## Adaptive Refinement

Core algorithm for computing rigorous transition probabilities.

```@docs
rel_size
build_adaptive_partition
build_uniform_partition
```

## Sparse Transition Matrices

Functions for building and managing sparse Ulam discretization matrices.

```@docs
UlamPartition
build_transition_matrix
normalize_columns!
build_normalized_transition_matrix
partition_index
estimate_sparsity
```

## Noise Convolution

Support for additive noise through Minkowski sums.

See the **Noise and Boundary Conditions** page in the User Guide for detailed documentation.

## Boundary Conditions

Periodic boundary conditions for systems on the torus.

See the **Noise and Boundary Conditions** page in the User Guide for detailed documentation.

## Index

```@index
```
