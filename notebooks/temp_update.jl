using JSON

# Read notebook
notebook = JSON.parsefile("contracting_lorenz.ipynb")

# Find and update cell 7 (Build Transition Matrix with Noise)
for cell in notebook["cells"]
    if haskey(cell, "source") && any(line -> occursin("Building noise convolution operator", line), cell["source"])
        # Update the cell source
        cell["source"] = [
            "println(\"Building deterministic transition matrix...\")\n",
            "@time P_det, U = build_transition_matrix(\n",
            "    contracting_lorenz, partition,\n",
            "    transition_graph=graph,\n",
            "    enclosure_method=zeroth_order_enclosure,\n",
            "    max_iterations=8,\n",
            "    ε=1e-4\n",
            ")\n",
            "\n",
            "println(\"\\nBuilding noise convolution operator...\")\n",
            "@time N = build_noise_convolution_operator(\n",
            "    noise, partition,\n",
            "    transition_graph=graph,\n",
            "    ε=1e-3,\n",
            "    max_iterations=10\n",
            ")\n",
            "\n",
            "println(\"\\nComposing operators: P_noisy = N * P_det\")\n",
            "P_noisy = N * P_det\n",
            "\n",
            "println(\"\\nMatrix statistics:\")\n",
            "@printf(\"  Size: %d × %d\\n\", size(P_noisy)...)\n",
            "@printf(\"  Non-zeros: %d\\n\", nnz(P_noisy))\n",
            "@printf(\"  Sparsity: %.1f%%\\n\", 100*(1 - nnz(P_noisy)/prod(size(P_noisy))))"
        ]
        break
    end
end

# Write back
open("contracting_lorenz.ipynb", "w") do io
    JSON.print(io, notebook, 2)
end

println("Updated notebook")
