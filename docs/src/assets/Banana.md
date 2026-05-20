# Banana Distribution

The Banana distribution[^1] [^2] is a classical example of a non-convex and non-elliptical two-dimensional
distribution. While it is a relatively low-dimensional target distribution, gradient-based samplers often
have trouble drawing samples from the target distribution due to areas of high curvature[^3] [^4]. The Banana
distribution on $\boldsymbol{\theta} := (\theta_1, \theta_2)$ can be specified in the following hierarchical way:

$$Y_i \sim \mathcal{N}\left((\theta_1 - \mu_1) + (\theta_2 - \mu_2)^2, 1\right)\;\;\;\;\; \theta_1 \sim \mathcal{N}(0, 4)\;\;\;\;\; \theta_2 \sim \mathcal{N}(0.5, 4),$$

where $Y_i$ is generated from $\mathcal{N}(0.1, 1)$.

### Specification of the Log Posterior Density

When using the `AdaptEllipticalSliceSampler.jl` package, the main component that the user must specify
is the log pdf of the target distribution. We will construct a Julia function to evaluate the log pdf 
(up to a constant) of the posterior distribution of $\boldsymbol{\theta}$ based on the 
hierarchical representation above.

```@example Banana
import Random
using AdaptEllipticalSliceSampler
using Distributions
using Plots
using LinearAlgebra

Random.seed!(123)

function log_posterior(theta::AbstractVector{<:AbstractFloat}, Y::Vector{<:AbstractFloat},
                       mu::Vector{<:AbstractFloat})
    mean = (theta[1] - mu[1]) + (theta[2] - mu[2])^2 
    ## Normal distribution for likelihood
    lpdf = -0.5 * norm(Y .- mean)^2
    ## Prior distributions 
    lpdf += logpdf(Normal(0, 2), theta[1]) + logpdf(Normal(0.5, 2), theta[2])

    return lpdf
end
```

We can see that the function `log_posterior` takes 3 arguments: `theta` (parameters of interest), `Y` (data),
and `mu` the mean parameters.

### Sampling via AGESS

To generate samples from the posterior distribution, we will use adaptive generalized elliptical slice
sampling[^3] (AGESS). We will also specify the prior mean $\mu_0$ and prior scale $\Sigma_0$ parameters.

```@example Banana
### Generate our data (100 observations)
Y = randn(100)

### random mean component of our model
mu = randn(2) * 3

### Sample Via AGESS
### 10000 MCMC samples (N_MCMC = 10000)
### Target distribution dimension (P = 2)
μ_0 = [0.0, 0.5]
Σ_0 = diagm(4*ones(2))
results = AGESS(t -> log_posterior(t, Y, mu), 10000, 2; μ_0 = μ_0, Σ_0 = Σ_0)
```

We can see that we use an anonymous function to specify the likelihood, where `theta` (our parameters of interest)
are the only parameters of interest. Here, `Y` and `mu` are the globally specified parameters (not-changing). We can visualize 
the results as follows:

```@example Banana
## Discard the first 2500 samples to burn-in
scatter(results.samps[2501:10000,1], results.samps[2501:10000,2], alpha = 0.4, 
        legend = false)
```

We can also view the log posterior density evaluated at each iteration of the Markov chain:

```@example Banana
plot(results.l_pdf[1:10000], legend = false)
```

# Twin Banana Distribution

The Twin Banana[^3] expands the banana distribution to a target distribution with two **isolated** bananas. Similarly,
we can construct through a similar hierarchical representation:

$$Y_i \sim \mathcal{N}\left(0.1(\theta_1 - \mu_1)^2  - 0.5(\theta_2 - \mu_2)^4 - 10(\theta_1 - \mu_1)(\theta_2 - \mu_2), 100\right),$$

$$\theta_1 \sim \mathcal{N}(0, 4)\;\;\;\;\; \theta_2 \sim \mathcal{N}(0, 4),$$

where $Y_i$ is generated from $\mathcal{N}(100, 100)$.

### Specification of the Log Posterior Density

We will construct a Julia function to evaluate the log pdf 
(up to a constant) of the posterior distribution of $\boldsymbol{\theta}$ based on the 
hierarchical representation of the Twin Banana distribution.

```@example Banana
function TB_log_posterior(theta::AbstractVector{<:AbstractFloat}, Y::Vector{<:AbstractFloat}, 
                          mu::Vector{<:AbstractFloat})
    m = (0.1 * (theta[1] - mu[1])^2  -  0.5 * (theta[2] - mu[2])^4 - 
            10 * (theta[1]- mu[1]) * (theta[2]- mu[2])) 
    ## Normal distribution for likelihood
    lpdf = -(0.5 / 100) * norm(Y .- m)^2
    ## Prior distributions 
    lpdf += logpdf(Normal(0, 2), theta[1]) + logpdf(Normal(0, 2), theta[2])

    return lpdf
end
```

Again, we can see that the function `TB_log_posterior` takes 3 arguments: `theta` (parameters of interest), `Y` (data),
and `mu` the mean parameters.

### Sampling via AGESS

To generate samples from the posterior distribution, we will use adaptive generalized elliptical slice
sampling[^3] (AGESS). We will specify the prior scale $\Sigma_0$ parameter, however since the prior mean
is centered at 0, we do not have to explicitly specify $\mu_0$.

```@example Banana
### Generate our data (100 observations)
Y_TB = randn(100) * 10 .+ 100 

### random mean component of our model
mu_TB = randn(2) .* 3

### Sample Via AGESS
### 10000 MCMC samples (N_MCMC = 50000)
### Target distribution dimension (P = 2)
Σ_0 = diagm(ones(2) .* 4.0)
results_TB = AGESS(t -> TB_log_posterior(t, Y_TB, mu_TB), 500000, 2; Σ_0 = Σ_0)
```

We can see that we use an anonymous function to specify the posterior pdf, where `theta` (our parameters of interest)
are the only parameters of interest. Here, `Y` and `mu` are the globally specified parameters (not-changing). We can visualize 
the results as follows:

```@example Banana
## Discard the first 125000 samples to burn-in
scatter(results_TB.samps[125001:500000,1], results_TB.samps[125001:500000,2], alpha = 0.4, 
        legend = false)
```

Again, we can also view the log posterior density evaluated at each iteration of the Markov chain:

```@example Banana
plot(results_TB.l_pdf[1:500000], legend = false)
```

### Conclusion

Sampling from these challenging, yet low-dimensional distributions is fairly simple using adaptive 
generalized elliptical slice sampling[^3]. Using this package, sampling from these target distributions
is as simple as efficiently constructing a function that evaluates the log posterior density. Comparisons between
AGESS and other standard sampling schemes can be found in the main manuscript[^3]. 

[^1]: A. W. Long, K. C. Wolfe, M. J. Mashner, G. S. Chirikjian, et al. The banana distribution is gaussian: A localization study with exponential coordinates. Robotics: Science and Systems VIII, 265(1), 2013.

[^2]: E. Cameron and A. Pettitt. Recursive pathways to marginal likelihood estimation with prior-sensitivity analysis. Statistical Science, pages 397–419, 2014.

[^3]: N. Marco and S. T. Tokdar. Adaptive Generalized Elliptical Slice Sampling

[^4]: M. Betancourt. A conceptual introduction to hamiltonian monte carlo. arXiv preprint arXiv:1701.02434, 2017.