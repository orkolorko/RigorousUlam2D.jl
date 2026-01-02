# RigorousUlam2D.jl Examples

This directory contains standalone Julia examples demonstrating the key features of RigorousUlam2D.jl. Each example can be run directly from the terminal.

## Running the Examples

From the package root directory:

```bash
# Activate the package environment
julia --project=.

# Run an example
julia --project=. examples/01_contracting_map.jl
```

Or from the examples directory:

```bash
cd examples
julia --project=.. 01_contracting_map.jl
```

## Example Files

### 01_contracting_map.jl
**Basic workflow for contracting maps**

Demonstrates:
- Defining a simple contracting map
- Building adaptive partition with diameter-based stopping
- Constructing transition matrix with graph optimization
- Computing invariant measure

Estimated runtime: ~10-30 seconds

### 02_henon_periodic.jl
**Periodic boundary conditions**

Demonstrates:
- Hénon map with periodic boundary conditions (torus topology)
- Adaptive partitioning with `periodic=true`
- Spectral analysis (eigenvalues, spectral gap, mixing time)

Estimated runtime: ~30-60 seconds

### 03_operator_decomposition.jl
**Operator decomposition: P_noisy = N * P**

Demonstrates:
- Building deterministic and noise operators separately
- Operator composition via matrix multiplication
- Reusing deterministic operator for multiple noise levels
- Performance comparison between methods

Estimated runtime: ~20-40 seconds

### 04_diameter_error_control.jl
**Diameter-based error control**

Demonstrates:
- Direct error control via partition diameter
- Comparison with fixed refinement levels
- Trade-off between accuracy and computational cost
- Theoretical error bounds

Estimated runtime: ~15-30 seconds

## Output

Each example prints:
- System parameters
- Partition statistics (boxes, diameter, graph sparsity)
- Matrix statistics (size, non-zeros, sparsity)
- Performance metrics
- Analysis results

Example output format:
```
======================================================================
Example 1: Simple Contracting Map
======================================================================

System:
  F(x,y) = (0.5x + 0.25, 0.5y + 0.25)
  Domain: [0,1]²
  Noise: ±0.01 uniform

Building adaptive partition...
  ...

Partition statistics:
  Target diameter: 0.03
  Achieved diameter: 0.029531
  Number of boxes: 87
  Graph non-zeros: 1247
  Graph sparsity: 83.5%

...
```

## Next Steps

After running these examples:

1. **Modify parameters**: Try different noise levels, target diameters, or systems
2. **Explore notebooks**: See `notebooks/` for interactive Jupyter notebooks
3. **Read documentation**: See `docs/` for detailed API reference
4. **Run tests**: `julia --project=. -e 'using Pkg; Pkg.test()'`

## Tips

- Start with smaller target diameters (e.g., 0.05) for faster runs
- Use `verbose=true` to see detailed progress information
- For large partitions (>500 boxes), eigenvalue computation may be slow
- Graph optimization provides 10-100× speedup for matrix construction

## Requirements

All examples require:
- Julia 1.12+
- RigorousUlam2D.jl (this package)
- IntervalArithmetic.jl
- IntervalBoxes.jl
- LinearAlgebra (standard library)

These dependencies are automatically installed when you add RigorousUlam2D.jl.
