# Performance Tips

Julia is a dynamic programming language that allows for high performance computing. However, 
Julia features optional typing can lead to slow performance. Since the `AGESS` function essentially
only requires the user to specify a function evaluating the log target distribution, it is 
paramount that the user specifies an efficient implementation of this function, as this function 
will constantly be called in the `AGESS` function. Here, we will give a quick example of 3 
evaluations of the same target distribution; each leading to completely different computational
costs. While a full guide to writing performance oriented code in Julia is out of the scope 
of this documentation, here are some useful resources:

* [Performance Tips from Julia](https://docs.julialang.org/en/v1/manual/performance-tips/)

* [Learning Resources for Julia](https://julialang.org/learning/)

* [JET.jl](https://aviatesk.github.io/JET.jl/dev/)

* [BenchmarkTools.jl](https://juliaci.github.io/BenchmarkTools.jl/stable/)

## Linear Regression
 
Consider the standard model for linear regression (see the "Regression" tutorial for more details):

$$Y_i \sim \mathcal{N}(\mathbf{x}_i' \boldsymbol{\beta}, \sigma^2),$$

$$\boldsymbol{\beta} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}),$$

$$\sigma^2 \sim \text{Inv-Gamma}(1,1).$$

Let's consider a simple implementation of this function, where we do not specify any types:

```@example Perf
import Random
import LogExpFunctions
using BenchmarkTools
using AdaptEllipticalSliceSampler
using Distributions
using Plots
using LinearAlgebra

function lm_log_posterior_1(Param, X, y)
    P = length(Param)
    N = length(y)
    lpdf = logpdf(MvNormal(X * Param[1:(P-1)],  exp(Param[P]) * diagm(ones(N))), y)
    lpdf += logpdf(MvNormal(zeros(P-1),  diagm(ones(P-1))), Param[1:(P-1)])
    lpdf += logpdf(InverseGamma(1, 1), exp(Param[P])) + Param[P]

    return lpdf
end
```

Let's generate some synthetic data and benchmark how long it takes to run `lm_log_posterior_1`.

```@example Perf
Random.seed!(123)

function generate_data(N::T, D::T) where {T<:Integer}
    β = randn(D) * (2 * log(D))^(1.0 / 4)
    x = randn(N, D)
    y = zeros(Float64, N)
    for i in 1:N
        y[i] = randn() * 0.5 + dot(x[i,:], β)
    end

    return β, x, y
end

### Generate data with 1000 observations and 10 covariates
D = 10
β, X, y = generate_data(1000, D)

### Benchmark function
Param = ones(D + 1)
@benchmark lm_log_posterior_1($Param, $X, $y)
```

Let's see if we can improve on this by pre-allocating some of our variables and specifying the
type of variables.

```@example Perf
function lm_log_posterior_2(Param::AbstractVector{Y}, X::AbstractMatrix{Y}, 
                            y::AbstractVector{Y}, μ::AbstractVector{Y}, 
                            μ_0::AbstractVector{Y}, Σ_I_N::AbstractMatrix{Y},
                            Σ_I_P::AbstractMatrix{Y}) where {Y<:AbstractFloat}
    P = length(Param)
    @views μ .= X * Param[1:(P-1)]
    lpdf = logpdf(MvNormal(μ, exp(Param[P]) * Σ_I_N), y)
    @views lpdf += logpdf(MvNormal(μ_0,  Σ_I_P), Param[1:(P-1)])
    lpdf += logpdf(InverseGamma(1, 1), exp(Param[P])) + Param[P]

    return lpdf
end

### Pre-allocate parameters
μ = zeros(1000)
Σ_I_N = diagm(ones(1000))
Σ_I_P = diagm(ones(D))
μ_0 = zeros(D)
@benchmark lm_log_posterior_2($Param, $X, $y, $μ, $μ_0, $Σ_I_N, $Σ_I_P)
```

We can see that there was a modest improvement in performance. We can see that we are allocating
less memory. However, it is slow to actually construct these multivariate distributions and 
evaluate the log pdf of these distributions. We can just perform the calculations ourselves and
get significantly better performance. Since Julia is compiled just-in-time, we should feel free
to use for-loops as we please! (Unlike R)

```@example Perf
function lm_log_posterior_3(Param::AbstractVector{Y}, X::AbstractMatrix{Y}, 
                            y::AbstractVector{Y}, ph::AbstractVector{Y}) where {Y<: AbstractFloat}
    P = length(Param)
    ## Normal Likelihood
    @views ph .= X * Param[1:P-1]
    ph .-= y
    lpdf = -0.5 * (1 / exp(Param[P])) *  norm(ph)^2 - 
            (0.5 * length(y) * Param[P])

    ## Priors
    ## Std Normal prior on coefficients
    @views lpdf += -0.5 * norm(Param[1:P-1])^2

    ## IG(1,1) prior on scale parameter (log-transformed)
    lpdf += -1 * Param[P]  -  (1 / exp(Param[P]))
    
    return lpdf
end

ph = zeros(1000)
@benchmark lm_log_posterior_3($Param, $X, $y, $ph)
```

We can see that we have a 1000-fold speed-up by just efficiently evaluating the log posterior 
density. This directly translates into a similar magnitude increase in the effective sample size 
per second achieved by AGESS. 
**Therefore, it is paramount to write efficient functions that evaluate the log posterior density when using AGESS.** 


Tips:

* Packages like `JET.jl` can help catch inefficiencies in coding.

* Pre-allocate variables (especially for intermediate computations)

* `@views` can help reduce allocating new arrays when doing computations on subarrays