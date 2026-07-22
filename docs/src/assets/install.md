# Installation

This package requires Julia v1.10.10 or later. Once a sufficient version of Julia is installed,
call Julia and open up the REPL.

```@eval
using REPL
io = IOBuffer()
REPL.banner(io)
banner = String(take!(io))
import Markdown
Markdown.parse("```\n\$ julia\n\n$(banner)\njulia>\n```")
```

The package can be easily installed by entering Pkg mode of the REPL (type `]` in the REPL)
and running the command `add AdaptEllipticalSliceSampler`.

Alternatively one can install the package directly from the GitHub repository as follows:

```@repl
using Pkg
Pkg.add(url = "https://github.com/ndmarco/AdaptEllipticalSliceSampler.jl")
```
Once the package is installed, checkout one of the [tutorials](generated/Regression.md) to get started.