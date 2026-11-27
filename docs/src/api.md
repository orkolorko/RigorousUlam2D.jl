# API reference

```@meta
CurrentModule = RigorousUlam2D
```

## Grids and enclosures

```@docs
UlamPartition
bisect_interval
bisect_box
quadrisect
build_uniform_partition
grid_order
wrap_to_torus
zeroth_order_enclosure
first_order_enclosure
```

## The deterministic block

```@docs
UlamTree
refine!
repair!
ulam_matrices
ulam_hierarchy
transition_graph
fibred_area
```

## The maps of the examples

```@docs
henon_system
lorenz_system
metzger_lorenz_system
ulam_system
```

## The noisy operator and the norms of its powers

```@docs
SquareNoise
noisy_operator
noisy_ulam_operator
grid_constants
norms_of_powers_columns
norms_of_powers_sparse
```

## The L¹ bounds

```@docs
ulam_fixed_point
fixed_point_residual
fixed_point_distance
coordinate_variations
x_marginal
sigma_from_norms
defect_apriori
coarse_fine_norms
swapped_error
```

## Records, archival copies and the deposit

```@docs
block_file
record_file
file_sha256
record_column_sweep
write_density_hdf5
DENSITY_HDF5_ORDERING
write_block_hdf5
BLOCK_HDF5_FORMAT
BLOCK_HDF5_SCRIPT
DATAVERSE_FILE_LIMIT
deposit_plan
write_manifest
```

## The stages of a run

```@docs
run_blocks
run_stationary_density
run_column_sweep
run_transfer
transfer_bounds
PAPER_RECORDS
recertify_from_records
```
