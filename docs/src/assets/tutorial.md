```@setup tutorial
import Random
using AdaptEllipticalSliceSampler
using Distributions
using Plots
gr(size=(816,400), margin=6Plots.mm)
Random.seed!(1234)
```

## Test 

``@example tutorial

plot(randn(100))
``