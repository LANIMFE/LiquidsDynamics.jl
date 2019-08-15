using Documenter, LiquidsDynamics

makedocs(;
    modules=[LiquidsDynamics],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
    ],
    repo="https://gitlab.com/lanimfe/LiquidsDynamics.jl/blob/{commit}{path}#L{line}",
    sitename="LiquidsDynamics.jl",
    authors="Pablo Zubieta",
    assets=String[],
)

deploydocs(;
    repo="gitlab.com/lanimfe/LiquidsDynamics.jl",
)
