using Documenter
using AdaptEllipticalSliceSampler

makedocs(
    sitename = "AdaptEllipticalSliceSampler",
    format = Documenter.HTML(),
    modules = [AdaptEllipticalSliceSampler],
    pages = [
        "Tutorials" => [
            "Banana" => "assets\\Banana.md",
        ],
    ],
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#
