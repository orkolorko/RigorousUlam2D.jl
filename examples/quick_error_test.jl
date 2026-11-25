import Pkg
Pkg.activate(@__DIR__)

using RigorousUlam2D
using IntervalArithmetic, IntervalBoxes
using LinearAlgebra, SparseArrays
using Printf

# Contracting Lorenz Map — quick test with large noise
function contracting_lorenz(x, y; s=0.5, r=3.0, ρ=1.3, c0=-0.25, c1=0.25)
    ax = abs(x)
    if x isa Interval
        if inf(x) > 0
            return (-ρ * ax^s + 0.5, y * ax^r + c0)
        elseif sup(x) < 0
            return (ρ * ax^s - 0.5, -y * ax^r + c1)
        else
            Tx_pos = -ρ * ax^s + 0.5; Gxy_pos = y * ax^r + c0
            Tx_neg = ρ * ax^s - 0.5; Gxy_neg = -y * ax^r + c1
            return (RigorousUlam2D.hull(Tx_pos, Tx_neg), RigorousUlam2D.hull(Gxy_pos, Gxy_neg))
        end
    else
        x_val = x isa Real ? x : ForwardDiff.value(x)
        if x_val > 0
            return (-ρ * ax^s + 0.5, y * ax^r + c0)
        else
            return (ρ * ax^s - 0.5, -y * ax^r + c1)
        end
    end
end

domain = IntervalBox(interval(-0.5, 0.5), interval(-0.5, 0.5))

# Try several noise levels (grid-aligned: σ = domain_width / 2^k)
# domain_width = 1.0, so ξ = 2σ = 1.0/2^(k-1)
# Coarse grid: compute contraction norms with large noise
# These norms will be used with the coarse-fine strategy
σ = 0.5  # half-width → ξ = 1.0 = full domain width
ξ = 2σ
noise = uniform_noise_box(σ, 2)

println("="^60)
@printf("Lorenz: σ = %.4f, ξ = %.4f (coarse grid)\n", σ, ξ)
println("="^60)

partition, dyn_graph, topo_graph = build_adaptive_partition(
    contracting_lorenz, domain, noise,
    max_diameter=0.05,  # coarse
    initial_depth=2,
    enclosure_method=zeroth_order_enclosure,
    verbose=false
)
n = partition.n_boxes
δ = maximum(box_width(box) for box in partition.boxes)
println("Boxes: $n")
@printf("Max diameter δ: %.6f\n", δ)
flush(stdout)

P_det, _ = build_transition_matrix(
    contracting_lorenz, partition,
    transition_graph=dyn_graph,
    enclosure_method=zeroth_order_enclosure,
    max_iterations=12, ε=1e-6
)
N_op = build_grid_aligned_noise_operator(noise, partition, topological_graph=topo_graph)
P = N_op * P_det
normalize_columns!(P)

@printf("Matrix: %d × %d, nnz = %d\n", size(P)..., nnz(P))
flush(stdout)

# Compute norms of powers on the coarse grid
println("\nNorms of powers (coarse grid):")
raw_norms = norms_of_powers_sparse(P, 20, rigorous=true, verbose=true)
refined = refine_norms_of_powers(raw_norms, 20)
println("\nRefined norms:")
for k in 1:20
    @printf("  k=%2d: raw=%.6f  refined=%.6f\n", k, raw_norms[k], refined[k])
end

# Coarse-fine strategy: transfer coarse norms to fine grid
println("\n\nCoarse-fine strategy:")
println("  Coarse grid: $(n) boxes, δ_coarse = $(δ)")
var_ρ = noise_var12(noise)
@printf("  var(ρ) = %.4f, DFLY: A=0, B=var(ρ)=%.4f\n", var_ρ, var_ρ)

for n_fine_target in [4096, 16384, 65536]
    # Fine grid: just need the mesh size, not the actual matrix
    # For uniform grid on domain [-0.5,0.5]^2, area=1:
    # n_fine boxes → δ_fine ≈ 1/√n_fine (for 2D uniform)
    depth = Int(round(log2(n_fine_target) / 2))
    n_fine_actual = 4^depth
    δ_fine = 1.0 / 2^depth  # side length of each box

    fine_norms = coarse_fine_norms(refined[1:20], Float64(δ), Float64(var_ρ))
    fine_refined = fine_norms                # already refined by submultiplicativity

    @printf("\n  Fine grid: n_F=%d, δ_F=%.6f\n", n_fine_actual, δ_fine)
    for k in 1:min(10, 20)
        @printf("    k=%2d: C_k^coarse=%.4f  C_k^fine=%.4f  (refined=%.4f)\n",
                k, refined[k], fine_norms[k], fine_refined[k])
    end

    # Check mixing condition with fine δ
    ξ_min = minimum(diam(noise[i]) for i in 1:length(noise))
    for k in 1:20
        α = fine_refined[k]
        mixing_term = (3 * δ_fine / ξ_min) * var_ρ * k
        if α + mixing_term < 1.0
            error_bound = (3 * δ_fine * var_ρ) / (ξ_min * (1 - α)) * k
            @printf("    → Mixing satisfied at N=%d, α=%.4f, bound=%.4f\n",
                    k, α, error_bound)
            break
        end
        if k == 20
            best_total = minimum(fine_refined[j] + (3*δ_fine/ξ_min)*var_ρ*j for j in 1:20)
            @printf("    → Mixing NOT satisfied (best total=%.4f)\n", best_total)
        end
    end
end
flush(stdout)
