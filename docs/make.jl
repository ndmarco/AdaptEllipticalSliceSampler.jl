using Documenter
using AdaptEllipticalSliceSampler

makedocs(
    sitename = "AdaptEllipticalSliceSampler",
    format = Documenter.HTML(),
    modules = [AdaptEllipticalSliceSampler],
    pages = [
        "Tutorials" => [
            "Banana" => "assets\\Banana.md",
            "Deep GP Surrogates" => "assets\\Deep_GP.md",
            "Regression" => "assets\\Regression.md",
            "Constrained Inference" => "assets\\Constrained.md",
        ],
    ],
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo = "github.com//ndmarco/AdaptEllipticalSliceSampler.jl.git",
)
