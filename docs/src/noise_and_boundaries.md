# Noise and Boundary Conditions

## Additive Noise

RigorousUlam2D supports **rigorous treatment** of additive noise through Minkowski sums.

### Noise Model

Consider a noisy dynamical system:
```
x_{n+1} = F(x_n) + ξ_n
```
where `ξ_n` is random noise with support in some box `W`.

The **noisy image** is:
```
F(Q) ⊕ W = {F(x) + ξ : x ∈ Q, ξ ∈ W}
```

This is rigorously computed as a Minkowski sum of interval boxes.

### Creating Noise Boxes

```julia
# Uniform noise: same in all directions
noise = uniform_noise_box(0.01, 2)
# Creates: [-0.01, 0.01] × [-0.01, 0.01]

# Anisotropic noise: different per dimension
noise_aniso = anisotropic_noise_box([0.01, 0.005])
# Creates: [-0.01, 0.01] × [-0.005, 0.005]
```

### Computing Noisy Images

```julia
# Deterministic image
img_clean = first_order_enclosure(F, box)

# Noisy image (Minkowski sum)
img_noisy = noisy_image(F, box, noise)

# Verify: noisy image is larger
@assert box_measure(img_noisy) > box_measure(img_clean)
```

### Noisy Transition Matrices

```julia
# Build matrix with noise
P, U = build_noisy_transition_matrix(
    F, partition, noise,
    maxdepth=8,
    ε=1e-4
)
```

The algorithm computes:
```
P[i,j] = measure({x ∈ Q_j : F(x) + ξ ∈ Q_i for some ξ ∈ W}) / measure(Q_j)
```

**Effect of noise:**
- Increases connectivity (more off-diagonal entries)
- Smooths the operator (better mixing)
- Enlarges effective domain

### Example: Noise Effects

```julia
using RigorousUlam2D
using IntervalArithmetic, IntervalBoxes

F(x, y) = (0.5*x + 0.25, 0.5*y + 0.25)
domain = IntervalBox(interval(0.0, 1.0), interval(0.0, 1.0))

# Create partition
partition, _, _ = create_restricted_partition(
    F, domain, nothing,
    partition_depth=4,
    verbose=false
)

# Without noise
P_clean, _ = build_transition_matrix(F, partition,
                                     maxdepth=8, ε=1e-4)

# With noise
noise_levels = [0.01, 0.05, 0.1]
for noise_level in noise_levels
    noise = uniform_noise_box(noise_level, 2)
    P_noisy, _ = build_noisy_transition_matrix(F, partition, noise,
                                                maxdepth=8, ε=1e-4)

    println("Noise level: $noise_level")
    println("  Clean non-zeros: ", nnz(P_clean))
    println("  Noisy non-zeros: ", nnz(P_noisy))
    println("  Increase: ", round(100*(nnz(P_noisy)/nnz(P_clean)-1), digits=1), "%")
end
```

## Periodic Boundary Conditions

For systems on the **torus** (periodic domain), use periodic boundary conditions.

### Torus Topology

The torus `𝕋² = [a,b] × [c,d]` with periodic identification:
- Points `(b, y)` and `(a, y)` are the same
- Points `(x, d)` and `(x, c)` are the same

### Wrapping Coordinates

```julia
# Wrap a point to the domain
x_wrapped = wrap_to_torus(1.2, 0.0, 1.0)  # → 0.2

# Wrap an interval
I = interval(0.8, 1.2)
I_wrapped = wrap_to_torus(I, 0.0, 1.0)  # May return full domain if wraps around

# Wrap a box
box = IntervalBox(interval(0.9, 1.1), interval(-0.1, 0.3))
domain = IntervalBox(interval(0.0, 1.0), interval(0.0, 1.0))
box_wrapped = apply_periodic_bc(box, domain)
```

**Conservative approach:** If wrapping creates a disconnected set, we return the full domain to maintain rigorous enclosure.

### Noisy Image with Periodic BC

```julia
# Compute noisy image, then wrap to domain
img = noisy_image_periodic(F, box, noise, domain)
```

Workflow:
1. Compute `F(box)`
2. Add noise: `F(box) ⊕ W`
3. Apply periodic BC: wrap to domain

### Periodic Transition Matrix

```julia
# Build matrix for system on torus
P, U = build_noisy_transition_matrix_periodic(
    F, partition, noise, domain,
    maxdepth=8,
    ε=1e-4
)
```

**Use case:** Classical maps on the torus:
- Standard map: `x' = x + y mod 1, y' = y + K sin(2πx) mod 1`
- Arnold cat map: `x' = x + y mod 1, y' = x + 2y mod 1`
- Hénon map (with rescaling)

### Example: Hénon Map on Torus

```julia
using RigorousUlam2D
using IntervalArithmetic, IntervalBoxes
using LinearAlgebra

# Rescaled Hénon to fit in [-1,1]²
function henon_rescaled(x, y; a=1.4, b=0.3)
    x_phys = 1.5 * x
    y_phys = 0.5 * y
    x_new = (1 - a * x_phys^2 + y_phys) / 1.5
    y_new = (b * x_phys) / 0.5
    return (x_new, y_new)
end

domain = IntervalBox(interval(-1.0, 1.0), interval(-1.0, 1.0))
noise = uniform_noise_box(0.025, 2)

# Create restricted partition
partition, eff_dom, info = create_restricted_partition(
    henon_rescaled, domain, noise,
    max_diameter=0.02,
    verbose=true
)

# Build matrix with periodic BC
P, U = build_noisy_transition_matrix_periodic(
    henon_rescaled, partition, noise, domain,
    enclosure_method=first_order_enclosure,
    maxdepth=8,
    ε=1e-4
)

# Analyze
normalize_columns!(P)
stats = estimate_sparsity(P)

println("\nMatrix Statistics:")
println("  Size: ", size(P))
println("  Non-zeros: ", stats.nnz)
println("  Sparsity: ", round(100*stats.sparsity, digits=2), "%")

# Spectral analysis
if size(P, 1) <= 500
    λ = eigvals(Matrix(P))
    λ_sorted = sort(abs.(λ), rev=true)

    println("\nSpectral Properties:")
    println("  Leading eigenvalues:")
    for i in 1:min(5, length(λ_sorted))
        println("    λ_$i = ", round(λ_sorted[i], digits=6))
    end

    if length(λ_sorted) >= 2
        gap = 1.0 - λ_sorted[2]
        println("  Spectral gap: ", round(gap, digits=6))
        println("  Mixing rate: ", round(-log(λ_sorted[2]), digits=6))
    end
end
```

## Noise and Domain Restriction

**Important:** When using domain restriction with noise, the effective domain must account for noise expansion:

```julia
# Compute effective domain WITH noise
eff_domain, img_clean, img_noisy = compute_effective_domain(
    F, initial_domain, noise,
    subdivision_depth=2
)

println("Image (no noise): ", img_clean)
println("Image (with noise): ", img_noisy)
println("Noise expansion: ",
        round(box_measure(img_noisy)/box_measure(img_clean), digits=2), "×")
```

The effective domain is based on `img_noisy`, ensuring the partition covers all possible noisy trajectories.

## Rigorous Guarantees

All noise computations maintain rigorous enclosures:

1. **Minkowski sum** rigorously contains all sums `x + ξ` for `x ∈ A`, `ξ ∈ B`
2. **Periodic wrapping** conservatively returns full domain when ambiguous
3. **Transition probabilities** provide lower and upper bounds with explicit uncertainties

The uncertainty matrix `U` quantifies regions where the subdivision didn't resolve containment at the specified `maxdepth`.

## API Reference

```@docs
uniform_noise_box
anisotropic_noise_box
minkowski_sum
noisy_image
noisy_rel_size
build_noisy_transition_matrix
RigorousUlam2D.convolve_transition_matrix
wrap_to_torus
apply_periodic_bc
noisy_image_periodic
noisy_rel_size_periodic
build_noisy_transition_matrix_periodic
```
