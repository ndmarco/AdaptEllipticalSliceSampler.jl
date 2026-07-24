using Documenter
using Literate
using AdaptEllipticalSliceSampler

# Pages listed here are authored as Literate.jl scripts in `docs/literate/` rather than
# hand-written markdown. Literate regenerates both the Documenter markdown page (into
# `docs/src/generated/`) and a downloadable Jupyter notebook (into
# `docs/src/generated/notebooks/`) from the same source script, so the two never drift
# apart. Notebooks are generated but not executed (`execute=false`): these tutorials are
# compute-heavy and/or hit the network, so the notebook is left as a runnable template
# rather than doubling the doctest execution cost during the docs build.
const LITERATE_DIR = joinpath(@__DIR__, "literate")
const GENERATED_DIR = joinpath(@__DIR__, "src", "generated")
const NOTEBOOK_DIR = joinpath(GENERATED_DIR, "notebooks")

const LITERATE_PAGES = [
    "Motivation",
    "Performance",
    "Banana",
    "BayesNN",
    "Constrained",
    "Deep_GP",
    "Regression",
    "Factor",
]

mkpath(NOTEBOOK_DIR)
for name in LITERATE_PAGES
    src = joinpath(LITERATE_DIR, "$(name).jl")
    Literate.markdown(src, GENERATED_DIR; documenter = true, execute = false)
    Literate.notebook(src, NOTEBOOK_DIR; documenter = true, execute = false)
end

@info "Starting makedocs (doctest phase begins here)"
makedocs(
    sitename = "AdaptEllipticalSliceSampler",
    format = Documenter.HTML(),
    modules = [AdaptEllipticalSliceSampler],
    pages = [
        "Introduction" => "index.md",
        #"Why AGESS?" => "generated/Motivation.md",
        "Installation" => "assets/install.md",
        "Documentation" => "assets/Documentation.md",
        "Performance Tips" => "generated/Performance.md",
        "Tutorials" => [
            "Banana" => "generated/Banana.md",
            "Bayesian Neural Networks" => "generated/BayesNN.md",
            "Constrained Inference" => "generated/Constrained.md",
            "Deep GP Surrogates" => "generated/Deep_GP.md",
            "Regression" => "generated/Regression.md",
        ],
        "Advanced Tutorials" => [
            "Factor Models" => "generated/Factor.md",
        ],
    ],
)
@info "makedocs completed successfully"

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo = "github.com/ndmarco/AdaptEllipticalSliceSampler.jl.git",
    devbranch = "main"
)
