# Adaptive Partitioning

Adaptive partitioning optimizes transfer operator computation for contracting and dissipative systems by only refining regions where the dynamics lives.

## Algorithm

Forward flagging with adaptive refinement:

1. Initialize with coarse partition (`initial_depth=2`)
2. For each box: compute `img = F(box) ⊕ noise ⊕ margin`, flag intersecting boxes
3. Refine flagged boxes exceeding diameter criterion
4. Repeat until stopping criterion met

## Usage

```julia
partition, transition_graph = build_adaptive_partition(
    F, domain, noise,
    max_diameter=0.01,       # Target error tolerance
    initial_depth=2,         # Start coarse
    safety_margin=1.1,       # 10% safety buffer
    enclosure_method=first_order_enclosure
)
```

## API Reference

See the [API Reference](api.md) page for function documentation.
