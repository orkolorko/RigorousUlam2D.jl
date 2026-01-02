"""
Boundary condition handling for transfer operators.

This module implements periodic boundary conditions for the Ulam method,
where the phase space is topologically a torus (with wraparound).
"""

using IntervalArithmetic, IntervalBoxes

export apply_periodic_bc, wrap_to_torus
export noisy_image_periodic

"""
    wrap_to_torus(x::Real, domain_min::Real, domain_max::Real)

Wrap a single coordinate to the periodic domain [domain_min, domain_max).

# Example
```julia
wrap_to_torus(1.2, 0.0, 1.0)  # → 0.2
wrap_to_torus(-0.3, 0.0, 1.0) # → 0.7
```
"""
function wrap_to_torus(x::Real, domain_min::Real, domain_max::Real)
    period = domain_max - domain_min
    shifted = x - domain_min
    wrapped = mod(shifted, period)
    return wrapped + domain_min
end

"""
    wrap_to_torus(I::Interval, domain_min::Real, domain_max::Real)

Wrap an interval to the periodic domain.

If the interval is wider than the period, returns the full domain.
If the interval wraps around, returns the full domain (conservative).
Otherwise returns the wrapped interval.

# Note
This is conservative: if wrapping would create a disconnected set,
we return the full domain to maintain interval enclosure.
"""
function wrap_to_torus(I::Interval, domain_min::Real, domain_max::Real)
    period = domain_max - domain_min

    # If interval is wider than period, it covers everything
    if diam(I) >= period
        return interval(domain_min, domain_max)
    end

    # Wrap the endpoints
    inf_wrapped = wrap_to_torus(inf(I), domain_min, domain_max)
    sup_wrapped = wrap_to_torus(sup(I), domain_min, domain_max)

    # Check if the interval wraps around
    if inf_wrapped <= sup_wrapped
        # No wraparound
        return interval(inf_wrapped, sup_wrapped)
    else
        # Wraparound: interval spans [domain_min, sup_wrapped] ∪ [inf_wrapped, domain_max]
        # Conservative: return full domain
        return interval(domain_min, domain_max)
    end
end

"""
    apply_periodic_bc(box::IntervalBox{N}, domain::IntervalBox{N}) where {N}

Apply periodic boundary conditions to a box.

Each dimension is wrapped independently to stay within the domain.

# Arguments
- `box`: The box to wrap
- `domain`: The periodic domain (typically [-1,1]^N or [0,1]^N)

# Returns
A wrapped IntervalBox, or the full domain if wrapping creates ambiguity

# Example
```julia
domain = IntervalBox(interval(0.0, 1.0), interval(0.0, 1.0))
box = IntervalBox(interval(0.8, 1.2), interval(-0.1, 0.3))
wrapped = apply_periodic_bc(box, domain)
# x wraps around: returns full domain conservatively
```
"""
function apply_periodic_bc(box::IntervalBox{N}, domain::IntervalBox{N}) where {N}
    wrapped_intervals = [wrap_to_torus(box[i], inf(domain[i]), sup(domain[i]))
                        for i in 1:N]
    return IntervalBox(wrapped_intervals...)
end

"""
    noisy_image_periodic(F, box::IntervalBox{N}, noise::IntervalBox{N},
                        domain::IntervalBox{N};
                        enclosure_method=first_order_enclosure) where {N}

Compute the noisy image with periodic boundary conditions.

Workflow:
1. Compute F(box)
2. Add noise (Minkowski sum)
3. Apply periodic boundary conditions (wrap to domain)

# Arguments
- `F`: The map (should map domain to itself mod periodicity)
- `box`: Source box
- `noise`: Noise support
- `domain`: The periodic domain
- `enclosure_method`: Enclosure computation method

# Returns
IntervalBox enclosing the wrapped noisy image

# Example
```julia
# Standard map on torus
F(x, y) = (mod(x + y, 1.0), mod(y + K*sin(2π*x), 1.0))
domain = IntervalBox(interval(0.0, 1.0), interval(0.0, 1.0))
noise = uniform_noise_box(0.01, 2)
box = IntervalBox(interval(0.9, 0.95), interval(0.4, 0.5))

img = noisy_image_periodic(F, box, noise, domain)
# Image wraps around the boundary
```
"""
function noisy_image_periodic(F, box::IntervalBox{N}, noise::IntervalBox{N},
                             domain::IntervalBox{N};
                             enclosure_method=first_order_enclosure) where {N}

    # Compute deterministic image
    deterministic_image = enclosure_method(F, box)

    # Convert to IntervalBox if needed
    if deterministic_image isa Tuple
        image_box = IntervalBox(deterministic_image...)
    else
        image_box = deterministic_image
    end

    # Add noise
    noisy_image_box = minkowski_sum(image_box, noise)

    # Apply periodic boundary conditions
    wrapped = apply_periodic_bc(noisy_image_box, domain)

    return wrapped
end

# DEPRECATED: noisy_rel_size_periodic has been removed.
# This function computed the annealed operator (F followed by noise with periodic BC),
# which violates the operator decomposition architecture where dynamics F and noise N
# are separate operators.
#
# Use the operator composition architecture instead:
#
#   # Build deterministic transition matrix with periodic BC
#   P, U = build_transition_matrix(F, partition, periodic=true, transition_graph=graph)
#
#   # Build noise convolution operator with periodic BC
#   N = build_noise_convolution_operator(noise, partition, periodic=true, transition_graph=graph)
#
#   # Compose to get noisy transition operator
#   P_noisy = N * P
#
# This follows the operator decomposition: P_noisy = N ∘ P
# where N is the noise convolution operator and P is the deterministic transition operator.

# DEPRECATED: build_noisy_transition_matrix_periodic has been removed.
# Use the unified architecture instead:
#
#   # For periodic BC with noise, use operator composition:
#   partition, graph = build_adaptive_partition(F, domain, noise, max_diameter=0.05, periodic=true)
#   P, U = build_transition_matrix(F, partition, periodic=true, transition_graph=graph)
#   N = build_noise_convolution_operator(noise, partition, periodic=true, transition_graph=graph)
#   P_noisy = N * P
#
# This follows the operator decomposition: P_noisy = N ∘ P
# where N is the noise convolution operator and P is the deterministic transition operator.
