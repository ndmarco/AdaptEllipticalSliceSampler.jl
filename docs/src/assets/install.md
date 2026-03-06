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

The package can be easily installed as follows:

```@repl
using Pkg
Pkg.add(url = "https://github.com/ndmarco/AdaptEllipticalSliceSampler.jl.git")
```


Alternatively, we can enter the Pkg mode of the REPL by pressing `]` in the REPL and install 
the package using the command `add https://github.com/ndmarco/AdaptEllipticalSliceSampler.jl.git`.

Once the package is installed, checkout one of the tutorials to get started.