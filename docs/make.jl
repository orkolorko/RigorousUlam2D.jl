using RigorousUlam2D
using Documenter

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
        "API Reference" => "api.md",
    ],
    checkdocs=:exports,  # Only check exported functions
)

deploydocs(;
    repo="github.com/orkolorko/RigorousUlam2D.jl",
    devbranch="main",
)
