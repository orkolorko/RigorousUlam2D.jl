"""
The L¹ distance between the stationary density `f_ξ` of `L_ξ = N_ξ L_T` and the stationary
density `f_h` of its Ulam discretisation `P_h = π_h N_ξ π_h L_T π_h` on the square grid of cell
side `h`, from the certified spectral data of `P_h` and the estimates of the paper (Theorems A
and D, prop:gap_to_mixing, eq:reduced_resolvent, eq:l1tol2 of RigorousUlam2D.tex), and the
distance between `f_h` and a computed vector `f̃`.

Vectors are in the grid order of `noisy_operator.jl` and hold cell masses: the function with
values `m_i / h²` on the cells, `‖f‖_{L¹} = Σ|m_i|`.

**The scheme.** With `e := f_ξ − f_h ∈ V` (both densities integrate to one),

    (I − P_h) e = f_ξ − P_h f_ξ = (L_ξ − P_h) f_ξ,

so `‖e‖₁ ≤ Σ_V · ‖(L_ξ − P_h) f_ξ‖₁` with `Σ_V ≥ ‖(I − P_h)⁻¹|_V‖_{L¹→L¹}`.

**The defect** (`defect_apriori`). Writing
`L_ξ − P_h = (I − π_h) N_ξ L_T + π_h N_ξ (I − π_h) L_T + π_h N_ξ π_h L_T (I − π_h)` and using
Theorem D with `δ` the diameter of a cell (`√2 h` for a square cell): (i) and (iii) on the first
term, `‖(I − π_h) N_ξ g‖₁ ≤ (δ/2) var(N_ξ g) ≤ (δ/2) var(ρ_ξ) ‖g‖₁`; (ii) and (iv) on the second,
`‖N_ξ (I − π_h) g‖₁ ≤ var(ρ_ξ) ‖(I − π_h) g‖_W ≤ δ var(ρ_ξ) ‖g‖₁`; (i) on the third, the operators
in front being L¹ contractions, `≤ (δ/2) var(g)`.  Hence, for `g = f_ξ`,

    ‖(L_ξ − P_h) f_ξ‖₁ ≤ (3δ/2) var(ρ_ξ) + (δ/2) var(f_ξ),

and `var(f_ξ) = var(L_ξ f_ξ) ≤ var(ρ_ξ)` by Theorem A, so `2δ var(ρ_ξ)` a priori.  For the uniform
density on `[-a, a]²`, `var(ρ_ξ) = 8a / (4a²) = 2/a`.

**The resolvent on `V`.** For `v ∈ V`, `(I − P_h)⁻¹ v = v + (I − P_h)⁻¹ P_h v` with
`P_h v ∈ V_h`, the zero-sum grid vectors, and `‖P_h v‖₁ ≤ ‖v‖₁`; so
`Σ_V ≤ 1 + Σ_grid` with `Σ_grid ≥ ‖(I − P_h)⁻¹|_{V_h}‖_{L¹→L¹}`, and two bounds on it.

- `sigma_from_gap(K, r)`: prop:gap_to_mixing gives `C_i := ‖P_h^i|_{V_h}‖_{L¹} ≤ K rⁱ` for
  `i ≥ 2`, with `K = |M|^{1/2} C_M var(ρ_ξ) R_Γ`, `R_Γ ≥ sup_{|z| = r} ‖(P_h − z)⁻¹‖₂`, and
  `C_i ≤ 1` for every `i` since `P_h` is a Markov operator.  For `k` with `α := K rᵏ < 1`,
  `Σ_{j ≥ 0} P_h^j = Σ_{m ≥ 0} P_h^{mk} Σ_{i < k} P_h^i` on `V_h` and `‖P_h^{mk}|_{V_h}‖ ≤ αᵐ`, so

      Σ_grid ≤ Σ_{i < k} min(1, K rⁱ) / (1 − α),

  minimised over `k ≥ 2`.
- `sigma_from_circle(area, varρ, R1)`: for `v ∈ V`, eq:l1tol2 gives
  `‖P_h v‖_{L²} ≤ C_M var(ρ_ξ) ‖v‖₁` (Theorem A and the zero-mean BV–Sobolev embedding,
  `C_M = 1/(2√2)`), eq:reduced_resolvent gives `‖(I − P_h)⁻¹|_{V_h}‖₂ ≤ R₁` from a circle
  `|z − 1| = ρ` holding only the eigenvalue 1 with `‖(P_h − z)⁻¹‖₂ ≤ R₁` on it (the Kato
  integral, in which `ρ` cancels), and `‖g‖₁ ≤ |M|^{1/2} ‖g‖₂`; so

      Σ_V ≤ 1 + |M|^{1/2} C_M var(ρ_ξ) R₁,

  which bounds `Σ_grid` as well (`V_h ⊂ V`).  The ℓ² and L² operator norms coincide on the
  uniform grid.

**The computed vector** (`fixed_point_residual`, `fixed_point_distance`).  For `f̃ ≥ 0` with
`s := Σ f̃`, `|s − 1| ≤ ε₂`, and `‖P_h f̃ − f̃‖₁ ≤ ε₁`: `u := f̃/s` has `u − f_h ∈ V_h` and
`(I − P_h)(u − f_h) = (f̃ − P_h f̃)/s`, so `‖u − f_h‖₁ ≤ Σ_grid ε₁ / (1 − ε₂)`, and
`‖f̃ − u‖₁ = |1 − 1/s| ‖f̃‖₁ ≤ ε₂ (1 + ε₂) / (1 − ε₂)`; together

    ‖f̃ − f_h‖₁ ≤ (Σ_grid ε₁ + ε₂ (1 + ε₂)) / (1 − ε₂).

The residual `ε₁` is the ℓ¹ norm of the computed `fl(N L̂ f̃) − f̃` plus the ℓ¹ rounding model of
`noisy_operator` (`η₁ ‖f̃‖₁ + η₁ᴬ`), which covers the rounding of both products and the
enclosure `L_lo ≤ L ≤ L_lo + U`.  The variation of the computed density on the torus is
`grid_variation`.  Every constant is rounded upward.
"""

export grid_variation, ulam_fixed_point, fixed_point_residual, fixed_point_distance,
    sigma_from_gap, sigma_from_circle, defect_apriori

"`1/(2√2) = √2/4` rounded upward: the constant of lem:sobolev2d."
const C_M_UP = sqrt_up(2.0) / 4

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
`ε1 = Σ_i |fl(Av)_i − v_i|⁺ + η1 ‖v‖₁ + eta1A`, upward.
"""
function fixed_point_residual(Amul, v::AbstractVector{Float64}, η1::Float64, eta1A::Float64)
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

`‖f̃ − f_h‖₁ ≤ (Σgrid ε1 + ε2 (1 + ε2)) / (1 − ε2)` of the file docstring, upward.
"""
function fixed_point_distance(Σgrid::Float64, ε1::Float64, ε2::Float64)
    ε2 < 1 || throw(ArgumentError("the mass defect ε2 = $ε2 must be below 1"))
    num = add_up(mul_up(Σgrid, ε1), mul_up(ε2, add_up(1.0, ε2)))
    den = -sub_up(ε2, 1.0)                 # 1 − ε2 rounded downward
    return div_up(num, den)
end

"""
    sigma_from_gap(K, r; kmax = 100_000) -> (; grid, V, k, α)

`Σ_{i<k} min(1, K rⁱ) / (1 − α)`, `α = K rᵏ`, over `k ≥ 2` with `α < 1`, at the `k` giving the
smallest value (`grid`), and `V = 1 + grid`; from prop:gap_to_mixing, see the file docstring.
`C_0 = C_1 = 1` are used as they stand.  Upward.
"""
function sigma_from_gap(K::Float64, r::Float64; kmax::Int = 100_000)
    0 < r < 1 || throw(ArgumentError("the rate r = $r must be in (0, 1)"))
    K > 0 || throw(ArgumentError("K = $K must be positive"))
    best = (; grid = Inf, V = Inf, k = 0, α = NaN)
    partial = 2.0                          # C_0 + C_1
    rk = mul_up(r, r)                      # rᵏ for k = 2, upward
    for k in 2:kmax
        α = mul_up(K, rk)
        if α < 1
            S = div_up(partial, -sub_up(α, 1.0))
            S < best.grid && (best = (; grid = S, V = add_up(1.0, S), k, α))
            # beyond this point 1/(1 − α) is 1 to working precision while the sum keeps growing
            α < eps(Float64) && break
        end
        partial = add_up(partial, min(1.0, α))
        rk = mul_up(rk, r)
        rk == 0 && break
    end
    isfinite(best.grid) || throw(ArgumentError("K rᵏ never fell below 1 within kmax"))
    return best
end

"""
    sigma_from_circle(area, varρ, R1) -> (; grid, V)

`1 + |M|^{1/2} C_M var(ρ_ξ) R₁`, the same number for `V_h` and `V`; from eq:reduced_resolvent
and eq:l1tol2, see the file docstring.  Upward.
"""
function sigma_from_circle(area::Float64, varρ::Float64, R1::Float64)
    S = add_up(1.0, mul_up(mul_up(mul_up(sqrt_up(area), C_M_UP), varρ), R1))
    return (; grid = S, V = S)
end

"""
    defect_apriori(δ, varρ, varf) -> bound

`‖(L_ξ − P_h) f‖₁ ≤ (3δ/2) var(ρ_ξ) + (δ/2) var(f)` for a density `f`, with `δ` the diameter
of a cell; see the file docstring.  Upward.
"""
defect_apriori(δ::Float64, varρ::Float64, varf::Float64) =
    add_up(mul_up(mul_up(1.5, δ), varρ), mul_up(mul_up(0.5, δ), varf))
