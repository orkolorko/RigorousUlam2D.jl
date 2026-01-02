# Noise and Boundary Conditions

## Additive Noise

RigorousUlam2D supports rigorous treatment of additive noise through operator composition.

### Noise Model

For a noisy system `x_{n+1} = F(x_n) + ξ_n` where `ξ_n` has support in box `W`, the noisy transfer operator is:

```
P_noisy = N * P_det
```

where `N` is the noise convolution operator and `P_det` is the deterministic transition operator.

### Creating Noise Boxes

```julia
# Uniform noise
noise = uniform_noise_box(0.01, 2)  # ±0.01 in both dimensions

# Anisotropic noise
noise = anisotropic_noise_box([0.01, 0.005])  # Different per dimension
```

### Building Noisy Transition Matrices

Use operator composition:

```julia
# Build deterministic operator
P_det, U = build_transition_matrix(F, partition,
    transition_graph=graph,
    enclosure_method=first_order_enclosure,
    maxdepth=8, ε=1e-4
)

# Build noise convolution operator
N = build_noise_convolution_operator(noise, partition,
    transition_graph=graph,
    ε=1e-4
)

# Compose operators
P_noisy = N * P_det
```

## Periodic Boundary Conditions

For systems on the torus, use `periodic=true`:

```julia
# Build transition matrix with periodic BC
P, U = build_transition_matrix(F, partition,
    transition_graph=graph,
    periodic=true,
    enclosure_method=first_order_enclosure,
    maxdepth=8, ε=1e-4
)

# Build noise operator with periodic BC
N = build_noise_convolution_operator(noise, partition,
    transition_graph=graph,
    periodic=true,
    ε=1e-4
)
```

Periodic boundary conditions wrap coordinates to the domain, treating opposite edges as identified.

## API Reference

```@docs
uniform_noise_box
anisotropic_noise_box
minkowski_sum
noisy_image
build_noise_convolution_operator
wrap_to_torus
apply_periodic_bc
noisy_image_periodic
```
