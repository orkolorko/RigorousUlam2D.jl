"""
Boundary condition handling for transfer operators.

This module implements periodic boundary conditions for the Ulam method,
where the phase space is topologically a torus (with wraparound).
"""

using IntervalArithmetic, IntervalBoxes

export apply_periodic_bc, wrap_to_torus
export noisy_image_periodic, build_noisy_transition_matrix_periodic

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

"""
    noisy_rel_size_periodic(F, Q_source::IntervalBox{N}, Q_target::IntervalBox{N},
                           noise::IntervalBox{N}, domain::IntervalBox{N};
                           enclosure_method=first_order_enclosure,
                           ε=1e-6, maxdepth=30, depth=0) where {N}

Compute relative size with periodic boundary conditions and noise.

This is the periodic version of `noisy_rel_size`, where images are wrapped
to the torus before checking containment.

# Arguments
- `F`: The map
- `Q_source`: Source box
- `Q_target`: Target box
- `noise`: Noise support
- `domain`: Periodic domain
- Other args: see `rel_size`

# Returns
`(lower_bound, upper_bound, uncertain_boxes)` as in `rel_size`
"""
function noisy_rel_size_periodic(F, Q_source::IntervalBox{N}, Q_target::IntervalBox{N},
                                noise::IntervalBox{N}, domain::IntervalBox{N};
                                enclosure_method=first_order_enclosure,
                                ε=1e-6, maxdepth=30, depth=0) where {N}

    # Base case: empty source
    if box_measure(Q_source) == 0
        return (0.0, 0.0, IntervalBox{N}[])
    end

    # Compute wrapped noisy image
    image_box = noisy_image_periodic(F, Q_source, noise, domain;
                                    enclosure_method=enclosure_method)

    # Case 1: Image definitely contained in target
    if issubset_box(image_box, Q_target)
        measure = box_measure(Q_source)
        return (measure, measure, IntervalBox{N}[])
    end

    # Case 2: Image definitely disjoint from target
    if isdisjoint_box(image_box, Q_target)
        return (0.0, 0.0, IntervalBox{N}[])
    end

    # Case 3: Uncertain - need to refine
    if depth >= maxdepth || box_width(Q_source) < ε
        measure = box_measure(Q_source)
        return (0.0, measure, [Q_source])
    end

    # Bisect and recurse
    left, right = bisect_box(Q_source)
    sub_boxes = [left, right]

    total_lower = 0.0
    total_upper = 0.0
    all_uncertain = IntervalBox{N}[]

    for sub_box in sub_boxes
        lower, upper, uncertain = noisy_rel_size_periodic(
            F, sub_box, Q_target, noise, domain;
            enclosure_method=enclosure_method,
            ε=ε, maxdepth=maxdepth, depth=depth+1
        )
        total_lower += lower
        total_upper += upper
        append!(all_uncertain, uncertain)
    end

    return (total_lower, total_upper, all_uncertain)
end

"""
    build_noisy_transition_matrix_periodic(F, partition::UlamPartition{N},
                                          noise::IntervalBox{N},
                                          domain::IntervalBox{N};
                                          kwargs...) where {N}

Build a transition matrix for a noisy system with periodic boundary conditions.

# Arguments
- `F`: The map (should respect periodicity mod wraparound)
- `partition`: Phase space partition (should tile the domain)
- `noise`: Noise support box
- `domain`: Periodic domain (e.g., [-1,1]² or [0,1]²)
- Additional kwargs passed to `noisy_rel_size_periodic`

# Returns
`(P, U)` where P is the transition matrix and U is the uncertainty matrix

# Example
```julia
# Henon map with noise on periodic torus
F_henon(x, y) = (1 - 1.4*x^2 + y, 0.3*x)
domain = IntervalBox(interval(-1.0, 1.0), interval(-1.0, 1.0))
partition = create_restricted_partition(F_henon, domain, nothing, partition_depth=4)
noise = uniform_noise_box(0.01, 2)

P, U = build_noisy_transition_matrix_periodic(F_henon, partition, noise, domain)
```
"""
function build_noisy_transition_matrix_periodic(F, partition::UlamPartition{N,T,I},
                                               noise::IntervalBox{N},
                                               domain::IntervalBox{N};
                                               enclosure_method=first_order_enclosure,
                                               ε=1e-6, maxdepth=30,
                                               use_lower_bound=true) where {N,T,I}

    n = partition.n_boxes
    I_indices = Int[]
    J_indices = Int[]
    values = Float64[]
    uncertainties_I = Int[]
    uncertainties_J = Int[]
    uncertainties_vals = Float64[]

    for j in 1:n
        Q_source = partition.boxes[j]
        source_measure = box_measure(Q_source)

        if source_measure == 0
            continue
        end

        for i in 1:n
            Q_target = partition.boxes[i]

            lower, upper, _ = noisy_rel_size_periodic(
                F, Q_source, Q_target, noise, domain;
                enclosure_method=enclosure_method,
                ε=ε, maxdepth=maxdepth
            )

            transition_measure = use_lower_bound ? lower : upper
            uncertainty = upper - lower

            if transition_measure > 0
                push!(I_indices, i)
                push!(J_indices, j)
                push!(values, transition_measure / source_measure)
            end

            if uncertainty > 0
                push!(uncertainties_I, i)
                push!(uncertainties_J, j)
                push!(uncertainties_vals, uncertainty / source_measure)
            end
        end
    end

    P = sparse(I_indices, J_indices, values, n, n)
    U = sparse(uncertainties_I, uncertainties_J, uncertainties_vals, n, n)

    return P, U
end
