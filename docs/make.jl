using KryosTools
using Documenter

DocMeta.setdocmeta!(KryosTools, :DocTestSetup, :(using KryosTools); recursive=true)

makedocs(;
    modules=[KryosTools],
    authors="JanJereczek <jan.jereczek@gmail.com> and contributors",
    sitename="KryosTools.jl",
    format=Documenter.HTML(;
        canonical="https://fesmc.github.io/KryosTools.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/fesmc/KryosTools.jl",
    devbranch="main",
)
