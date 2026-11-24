"""
The L¹ distance between the stationary density `f_ξ` of `L_ξ = N_ξ L_T` and the stationary
density `f_h` of its Ulam discretisation `P_h = π_h N_ξ π_h L_T π_h` on the square grid of cell
side `h`, and between `f_ξ` and a computed vector `f̃`, from bounds on the norms of the powers of
`P_h` on the zero-sum vectors (`norms_of_powers_columns`, `coarse_fine_norms`).

Vectors are in the grid order of `noisy_operator.jl` and hold cell masses: the function with
values `m_i / h²` on the cells, `‖f‖_{L¹} = Σ|m_i|`.

**The constants.** For `ρ_ξ` the uniform density on `[-a, a]²`, `V_i(ρ_ξ) = 1/a` is its variation
in the direction `e_i`, and `var(ρ_ξ) := V_1(ρ_ξ) + V_2(ρ_ξ) = 2/a`.  On the grid of squares of
side `h` (math/local_ly_2d.tex, Lemma "square cell" and Proposition "two projection constants"):

    ‖(I − π_h) N_ξ‖₁ ≤ (h/2) var(ρ_ξ),   ‖N_ξ (I − π_h)‖₁ ≤ (h/2) var(ρ_ξ),
    ‖(I − π_h) f‖₁ ≤ (h/2) (V_1(f) + V_2(f)).

**The scheme.** With `e := f_ξ − f_h ∈ V` (both densities integrate to one),

    (I − P_h) e = f_ξ − P_h f_ξ = (L_ξ − P_h) f_ξ,

so `‖e‖₁ ≤ Σ_V · ‖(L_ξ − P_h) f_ξ‖₁` with `Σ_V ≥ ‖(I − P_h)⁻¹|_V‖_{L¹→L¹}`.

**The defect** (`defect_apriori`). Writing
`L_ξ − P_h = (I − π_h) N_ξ L_T + π_h N_ξ (I − π_h) L_T + π_h N_ξ π_h L_T (I − π_h)`, the first two
terms are at most `(h/2) var(ρ_ξ)` each on a density, and the third at most
`(h/2)(V_1 + V_2)(f_ξ)`; `f_ξ = N_ξ (L_T f_ξ)` gives `V_i(f_ξ) ≤ V_i(ρ_ξ)` (Lemma "moll" there), so
`‖(L_ξ − P_h) f_ξ‖₁ ≤ (3h/2) var(ρ_ξ)` a priori.

**The resolvent on `V`.** For `v ∈ V`, `(I − P_h)⁻¹ v = v + (I − P_h)⁻¹ P_h v` with
`P_h v ∈ V_h`, the zero-sum grid vectors, and `‖P_h v‖₁ ≤ ‖v‖₁`; so `Σ_V ≤ 1 + Σ_grid` with
`Σ_grid ≥ ‖(I − P_h)⁻¹|_{V_h}‖_{L¹→L¹}`, which `sigma_from_norms` bounds from `C_i ≥ ‖P_h^i|_{V_h}‖₁`.

**The computed vector** (`fixed_point_residual`, `fixed_point_distance`).  For `f̃ ≥ 0` with
`s := Σ f̃`, `|s − 1| ≤ ε₂`, and `‖P_h f̃ − f̃‖₁ ≤ ε₁`: `u := f̃/s` has `u − f_h ∈ V_h` and
`(I − P_h)(u − f_h) = (f̃ − P_h f̃)/s`, so `‖u − f_h‖₁ ≤ Σ_grid ε₁ / (1 − ε₂)`, and
`‖f̃ − u‖₁ = |1 − 1/s| ‖f̃‖₁ = |s − 1| ≤ ε₂` (`f̃ ≥ 0`, so `‖f̃‖₁ = s`); together

    ‖f̃ − f_h‖₁ ≤ Σ_grid ε₁ / (1 − ε₂) + ε₂.

The residual `ε₁` is the ℓ¹ norm of the computed `fl(N L̂ f̃) − f̃` plus the ℓ¹ rounding model of
`noisy_operator` (`η₁ ‖f̃‖₁ + η₁ᴬ`), which covers the rounding of both products and the
enclosure `L_lo ≤ L ≤ L_lo + U`.  The variation of the computed density on the torus is
`grid_variation`.  Every constant is rounded upward.
"""

export grid_variation, ulam_fixed_point, fixed_point_residual, fixed_point_distance,
    sigma_from_norms, defect_apriori, coarse_fine_norms, swapped_error

_sum_up(v) = foldl(add_up, v; init = 0.0)
_sum_down(v) = -foldl((s, x) -> add_up(s, -x), v; init = 0.0)
_absdiff_up(a, b) = max(sub_up(a, b), sub_up(b, a))

"""
    grid_variation(m, k, h) -> var

An upper bound on the variation on the torus of the piecewise-constant function with cell masses
`m` (grid order, `k × k` cells of side `h`): the jump `|m_i − m_j| / h²` across each of the `2k²`
edges of length `h`, so `var = Σ_edges |m_i − m_j| / h`, rounded upward.
"""
function grid_variation(m::AbstractVector{Float64}, k::Int, h::Float64)
    length(m) == k * k || throw(DimensionMismatch("expected $(k * k) masses, got $(length(m))"))
    M = reshape(m, k, k)                   # M[iy, ix]
    s = 0.0
    @inbounds for ix in 1:k
        ixn = mod1(ix + 1, k)
        for iy in 1:k
            a = M[iy, ix]
            s = add_up(s, _absdiff_up(a, M[mod1(iy + 1, k), ix]))
            s = add_up(s, _absdiff_up(a, M[iy, ixn]))
        end
    end
    return div_up(s, h)
end

"""
    ulam_fixed_point(Amul, n; tol = 1e-14, maxiter = 10_000, v0 = fill(1/n, n)) -> (; v, iterations, diff, converged)

Power iteration `v ← Amul(v) / Σ Amul(v)` from `v0` until the ℓ¹ change is below `tol`.  Not a
certificate: `fixed_point_residual` bounds the residual of the result.
"""
function ulam_fixed_point(Amul, n::Int; tol::Float64 = 1e-14, maxiter::Int = 10_000,
        v0::AbstractVector{Float64} = fill(1.0 / n, n))
    v = Vector{Float64}(v0); v ./= sum(v)
    d = Inf
    for it in 1:maxiter
        w = Vector{Float64}(Amul(v)); w ./= sum(w)
        d = sum(abs, w .- v)
        v = w
        d < tol && return (; v, iterations = it, diff = d, converged = true)
    end
    return (; v, iterations = maxiter, diff = d, converged = false)
end

"""
    fixed_point_residual(Amul, v, η1, eta1A) -> (; ε1, ε2)

`ε1 ≥ ‖P_h v − v‖₁` and `ε2 ≥ |Σv − 1|` for the product `Amul(v) = fl(N L̂ v)` with the ℓ¹
rounding model `‖fl(N L̂ v) − P_h v‖₁ ≤ η1 ‖v‖₁ + eta1A` of `noisy_operator`:
`ε1 = Σ_i |fl(Av)_i − v_i| + η1 ‖v‖₁ + eta1A`, upward.  `v ≥ 0` is required, since
`fixed_point_distance` uses `‖v‖₁ = Σv`.
"""
function fixed_point_residual(Amul, v::AbstractVector{Float64}, η1::Float64, eta1A::Float64)
    all(>=(0), v) || throw(ArgumentError("v has negative entries; fixed_point_distance needs v ≥ 0"))
    w = Vector{Float64}(Amul(v))
    r = 0.0
    @inbounds for i in eachindex(v)
        r = add_up(r, _absdiff_up(w[i], v[i]))
    end
    ε1 = add_up(r, add_up(mul_up(η1, _sum_up(abs.(v))), eta1A))
    ε2 = max(sub_up(_sum_up(v), 1.0), sub_up(1.0, _sum_down(v)), 0.0)
    return (; ε1, ε2)
end

"""
    fixed_point_distance(Σgrid, ε1, ε2) -> bound

`‖f̃ − f_h‖₁ ≤ Σgrid ε1 / (1 − ε2) + ε2` of the file docstring, upward.
"""
function fixed_point_distance(Σgrid::Float64, ε1::Float64, ε2::Float64)
    ε2 < 1 || throw(ArgumentError("the mass defect ε2 = $ε2 must be below 1"))
    den = -sub_up(ε2, 1.0)                 # 1 − ε2 rounded downward
    return add_up(div_up(mul_up(Σgrid, ε1), den), ε2)
end

"""
    sigma_from_norms(C) -> (; grid, V, k, α)

`Σ_{i<k} min(1, C_i) / (1 − α)`, `α = C_k`, `C_0 = 1`, over the `k` with `C_k < 1`, at the `k`
giving the smallest value (`grid`), and `V = 1 + grid`, for bounds `C[i] ≥ ‖P_h^i|_{V_h}‖₁`,
`i = 1, …, length(C)`: on `V_h`, `Σ_{j ≥ 0} P_h^j = Σ_{m ≥ 0} P_h^{mk} Σ_{i < k} P_h^i` and
`‖P_h^{mk}|_{V_h}‖ ≤ αᵐ`, and `C_i ≤ 1` may be used for every `i` since `P_h` is a Markov
operator; see the file docstring.  Upward.
"""
function sigma_from_norms(C::AbstractVector{<:Real})
    best = (; grid = Inf, V = Inf, k = 0, α = NaN)
    partial = 1.0                          # C_0
    for k in eachindex(C)
        α = Float64(C[k])
        if α < 1
            S = div_up(partial, -sub_up(α, 1.0))
            S < best.grid && (best = (; grid = S, V = add_up(1.0, S), k, α))
        end
        partial = add_up(partial, min(1.0, α))
    end
    isfinite(best.grid) || throw(ArgumentError("no bound C_k is below 1"))
    return best
end

"""
    defect_apriori(h, varρ) -> bound

`‖(L_ξ − P_h) f_ξ‖₁ ≤ (3h/2) var(ρ_ξ)` on the grid of squares of side `h`, with
`var(ρ_ξ) = V_1(ρ_ξ) + V_2(ρ_ξ)`; see the file docstring.  Upward.
"""
defect_apriori(h::Float64, varρ::Float64) = mul_up(mul_up(1.5, h), varρ)

"""
    coarse_fine_norms(C, h, varρ; first = 1.0) -> F

Bounds `F_k ≥ ‖Λ^k|_V‖_{L¹}`, `k = 1, …, length(C) + 1`, for `Λ = σ N_ξ σ L_T σ` with `σ` the
Ulam projection of any dyadic refinement of the grid of squares of side `h`, or the identity
(`Λ = L_ξ`), from bounds `C_k ≥ ‖P_h^k|_{V_h}‖₁` for the coarse `P_h = π_h N_ξ π_h L_T π_h`
(`norms_of_powers_columns`; they bound `‖P_h^k|_V‖₁` as well, `P_h` ending with `π_h`).  It is the
recursion of math/local_ly_2d.tex, Proposition "recursion" (Galatolo, Monge, Nisoli 2020,
Lemma 29 with (36), with the two constants kept apart), as `norms_of_powers_recursive_noise(a,
b, C)` of RigorousInvariantMeasures, with `a ≥ ‖(I − π_h) N_ξ‖₁` and `b ≥ ‖N_ξ (I − π_h)‖₁` both
`(h/2) var(ρ_ξ)`, rounded upward.  `first ≥ ‖Λ‖` on `V`; 1 for a Markov operator.
"""
function coarse_fine_norms(C::AbstractVector{<:Real}, h::Float64, varρ::Float64; first::Real = 1.0)
    c = mul_up(0.5 * h, varρ)                   # h/2 is exact
    return RIM.norms_of_powers_recursive_noise(c, c, Vector{Float64}(C); first)
end

"""
    swapped_error(Σ, h, varρ, ε1, ε2) -> bound

`‖f_ξ − f̃‖₁` for a computed `f̃ ≥ 0` that is constant on the cells of the grid of squares of side
`h` of `P_h`, with `ε1 ≥ ‖P_h f̃ − f̃‖₁`, `ε2 ≥ |Σf̃ − 1|` (`fixed_point_residual`) and
`Σ ≥ ‖(I − L_ξ)⁻¹|_V‖₁` (`sigma_from_norms` of `coarse_fine_norms`, which bound the powers of
`L_ξ` itself).  With `u = f̃/Σf̃`, `u − f_ξ ∈ V` and

    (I − L_ξ)(u − f_ξ) = (u − P_h u) + (P_h − L_ξ) u;

in `L_ξ − P_h = (I − π_h) N_ξ L + π_h N_ξ (I − π_h) L + π_h N_ξ π_h L (I − π_h)` the last term
vanishes on `u`, which `π_h` fixes, and the other two are at most `(h/2) var(ρ_ξ)` each
(`‖u‖₁ = 1`; file docstring).  Hence

    ‖u − f_ξ‖₁ ≤ Σ (h var(ρ_ξ) + ε1/(1 − ε2)),   ‖f̃ − u‖₁ = |Σf̃ − 1| ≤ ε2,

and the bound is their sum, rounded upward; against `Σ · (3h/2) var(ρ_ξ)` of the direct route it
spends `h` in place of `3h/2`.
"""
function swapped_error(Σ::Float64, h::Float64, varρ::Float64, ε1::Float64, ε2::Float64)
    ε2 < 1 || throw(ArgumentError("the mass defect ε2 = $ε2 must be below 1"))
    den = -sub_up(ε2, 1.0)                                  # 1 − ε2, downward
    defect = add_up(mul_up(h, varρ), div_up(ε1, den))
    return add_up(mul_up(Σ, defect), ε2)
end
