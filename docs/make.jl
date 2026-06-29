using Documenter
using AdaptEllipticalSliceSampler

makedocs(
    sitename = "AdaptEllipticalSliceSampler",
    format = Documenter.HTML(),
    modules = [AdaptEllipticalSliceSampler],
    pages = [
        "Introduction" => "index.md",
        "Installation" => "assets/install.md",
        "Documentation" => "assets/Documentation.md",
        "Performance Tips" => "assets/Performance.md",
        "Tutorials" => [
            "Banana" => "assets/Banana.md",
            "Bayesian Neural Networks" => "assets/BayesNN.md",
            "Constrained Inference" => "assets/Constrained.md",
            "Deep GP Surrogates" => "assets/Deep_GP.md",
            "Regression" => "assets/Regression.md",
        ],
        "Advanced Tutorials" => [
            "Factor Models" => "assets/Factor.md",
        ],
    ],
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo = "github.com/ndmarco/AdaptEllipticalSliceSampler.jl.git",
    devbranch = "main"
)
