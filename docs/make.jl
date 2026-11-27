using RigorousUlam2D
using Documenter, Literate

# The worked examples of docs/literate as pages, not executed: a run is a computation
# (see example_settings in docs/literate/settings.jl).
for name in ("henon", "lorenz_like", "contracting_lorenz")
    Literate.markdown(joinpath(@__DIR__, "literate", "$name.jl"), joinpath(@__DIR__, "src", "examples");
                      documenter = true, execute = false,
                      codefence = "```julia" => "```")   # shown, not run by Documenter
end

DocMeta.setdocmeta!(RigorousUlam2D, :DocTestSetup, :(using RigorousUlam2D); recursive=true)

makedocs(;
    modules=[RigorousUlam2D],
    authors="Isaia Nisoli",
    sitename="RigorousUlam2D.jl",
    format=Documenter.HTML(;
        canonical="https://orkolorko.github.io/RigorousUlam2D.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "The Hénon map" => "examples/henon.md",
        "The Lorenz-like map" => "examples/lorenz_like.md",
        "The contracting Lorenz map" => "examples/contracting_lorenz.md",
        "API Reference" => "api.md",
    ],
    checkdocs=:exports,  # Only check exported functions
)

deploydocs(;
    repo="github.com/orkolorko/RigorousUlam2D.jl",
    devbranch="main",
)
