# Regression

In this tutorial, we will illustrate how to use the `AdaptEllipticalSliceSampler.jl` package, to
perform Bayesian computation using the adaptive generalized elliptical slice sampler[^1] in 
regression settings. We consider three settings in this tutorial: (1) Bayesian linear regression,
(2) high-dimensional linear sparse regression[^2] [^3], and (3) generalized ReLU regression.

## Bayesian Linear Regression

In this section, we consider using AGESS in the context of standard Bayesian linear regression.
Letting $\mathbf{Y} \in \mathbb{R}^N$ be the response variable and $\mathbf{X} \in \mathbb{R}^{N \times D}$
be the set of covariates, we can specify our model as follows:

$$Y_i \sim \mathcal{N}(\mathbf{x}_i' \boldsymbol{\beta}, \sigma^2),$$
$$\boldsymbol{\beta} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}),$$
$$\sigma^2 \sim \text{Inv-Gamma}(1,1).$$

Let's start by first generating data from our model.

```@example Regression
import Random
import LogExpFunctions
using AdaptEllipticalSliceSampler
using Distributions
using Plots
using LinearAlgebra

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
```

We can see that we generated the data from our model under $\sigma = 0.5$ ($N = 1000$, $D = 10$). 
Next, we can write a function that calculates the log posterior density. When conducting inference 
on variables that have positive support (i.e., $\sigma^2 > 0$), it is often useful to transform
the variables to remove these constraints. Thus, we will transform $\sigma^2$ using a log 
transformation to remove the positivity constraints (don't forget the Jacobian term 
when calculating the log pdfs). Additionally, we will concatenate all of our variables into 
one vector $\text{Param} = [\boldsymbol{\beta}, \log(\sigma^2)]$.

```@example Regression
function log_posterior(Param::AbstractVector{Y}, X::AbstractMatrix{Y}, 
                       y::AbstractVector{Y}) where {Y<: AbstractFloat}
    P = length(Param)
    ## Normal Likelihood
    @views lpdf = -0.5 * (1 / exp(Param[P])) *  norm(X * Param[1:P-1] - y)^2 - 
            (0.5 * length(y) * Param[P])

    ## Priors
    ## Std Normal prior on coefficients
    @views lpdf += -0.5 * norm(Param[1:P-1])^2

    ## IG(1,1) prior on scale parameter (log-transformed)
    lpdf += -1 * Param[P]  -  (1 / exp(Param[P]))
    
    return lpdf
end
```

Now that we have specified a function to efficiently evaluate the log posterior density, we can simply
use the `AGESS` function to draw samples from the posterior distribution. When calling the 
function `AGESS`, we will specify an anonymous function with $\text{Param}$ as the only input 
to the function `log_posterior`, using the global parameter values for $\mathbf{X}$ and $\mathbf{Y}$. 

```@example Regression
### Specify the dimension of the target distribution
P = D + 1
### Specify the number of MCMC iterations
n_MCMC = 10000

### Run AGESS
results = AGESS(Param -> log_posterior(Param, X, y), n_MCMC, P)
```

We can visualize some of the trace plots of the $\beta$ parameters using the output of `AGESS`.

```@example Regression
### Plot trace plot
plot(results.samps[2501:n_MCMC, findall(β .> 0)])
### Plot true values
hline!(β[findall(β .> 0)])
```

Similarly, we can view the trace plot of $\sigma^2$.

```@example Regression
### Plot trace plot, don't forget to transform the transformed variables back
plot(exp.(results.samps[2501:n_MCMC, D+1]))
### Plot true value
hline!([0.25])
```

Lastly, we can plot the log pdf of the posterior at every iteration of the Markov chain to potentially
detect convergence issues.

```@example Regression
plot(results.l_pdf[1:n_MCMC], legend = false)
```

## High-Dimensional Sparse Linear Regression

In many modern applications, the number of covariates may be larger than the sample size ($N < D$),
often requiring regularization in order to achieve good predictive properties. Here we consider
the sparse regression setting, where sparsity is induced via continuous shrinkage through global-local 
shrinkage priors. Here, we consider the horseshoe prior[^2] [^3]. Specifically, we consider the following
model:

$$Y_i \sim \mathcal{N}(\mathbf{x}_i'\boldsymbol{\beta}, \sigma^2)$$

$$\beta_j \sim \mathcal{N}(0, \sigma^2\tau^2\lambda_j^2)  \;\;\;\;\; p(\sigma^2) \propto \frac{1}{\sigma^2},$$

$$\tau \sim C^+(0,1) \;\;\;\;\; \lambda_j \sim C^+(0,1),$$

for $i = 1, \dots, N$ and $j = 1, \dots, D$, where $C^+$ denotes the half-Cauchy distribution. 
We will first start by generating data, where the design matrix has correlated covariates. We
will consider the case where we have 25 observations and 25 covariates. Here we will generate
a dataset where the probability of $\beta_j = 0$ is $0.9$ ($1 \le j \le 25$). 

```@example Regression
function gen_data_AR1(N::T, P::T; sparsity::Y = 0.8, ρ::Y = 0.2, 
                      σ_sq::Y = 1.0) where {Y<:AbstractFloat, T<:Integer}
    Σ = ones(P, P) 
    for i in 1:P
      for j in 1:P
        Σ[i,j] = ρ^(abs(i - j))
      end
    end
    Σ[diagind(Σ)] .= 1
    X = zeros(N, P)
    X .= rand(MultivariateNormal(zeros(P), Σ), N)'
    β = zeros(P)
    for i in 1:P
        if rand(Bernoulli(1 - sparsity)) == 1
          β[i] = (rand() * 3 + 1) * (-1)^i
        end
    end
    if sum(β) == 0
      β[1] = (rand() * 3 + 1) * (-1)^1
    end

    Y_obs = rand(MultivariateNormal(X * β, σ_sq * diagm(ones(N))))

    return X, Y_obs, β
end

Random.seed!(123)

N = 50
D = 25

X, Y, β = gen_data_AR1(N, D, ρ = 0.7, sparsity = 0.9, σ_sq = 1.0)
```

We visualize the correlation structure of the design matrix.

```@example Regression
heatmap(cor(X))
```

We can also view the values of $\boldsymbol{\beta}$.

```@example Regression
scatter(β,legend = false)
```

### Specification of the Log Posterior Density

When using the `AdaptEllipticalSliceSampler.jl`, it is crucial that we construct a function
that can efficiently evaluate the log posterior density. We will first start by constructing two
functions that evaluate the priors on $\boldsymbol{\beta}$ and $\boldsymbol{\lambda}$. Since
$\lambda_j$ has positive support, we will take a log transformation of the $\lambda_j$ parameters
($j = 1, \dots, D$). Similarly, we will take a log transformation of $\tau$ and $\sigma$.

```@example Regression
function prior_β(β::AbstractVector{Y}, τ::Y, λ::AbstractVector{Y}, 
                 σ::Y) where {Y<:AbstractFloat}
  lpdf::Float64 = 0.0
  for i in eachindex(β)
      @views lpdf += logpdf(Normal(0.0, exp(σ) * exp(τ) * exp(λ[i])), β[i])
  end
  return lpdf
end

function prior_λ(λ::AbstractVector{Y}) where {Y<:AbstractFloat}
  lpdf::Float64 = 0.0
  cauchy_d = Cauchy(0.0,1.0)
  for i in eachindex(λ)
      @views lpdf += logpdf(cauchy_d, exp(λ[i])) + λ[i]
  end
  return lpdf
end
```

Using these two functions, we can specify a function evaluating the log pdf of the target distribution
of interest. We will concatenate all the parameters into one vector $\text{Param} = [\boldsymbol{\beta}, \log(\boldsymbol{\lambda}), \log(\tau), \log(\sigma)]$.

```@example Regression
function log_posterior_HD(Param::AbstractVector{Y}, X::AbstractMatrix{Y}, 
                          y::AbstractVector{Y}, D::T) where {Y<:AbstractFloat, T<:Integer}
  lpdf::Float64 = 0.0
  ## Likelihood
  for i in eachindex(y)
    @views lpdf += logpdf(Normal(dot(X[i,:], Param[1:D]), exp(Param[2*D + 2])), y[i])
  end
  ## Prior 
  lpdf += prior_β(Param[1:D], Param[2*D + 1], Param[(D + 1):2*D], Param[2*D + 2]) + 
            prior_λ(Param[(D + 1):2*D]) + logpdf(Cauchy(0.0,1.0), exp(Param[2*D + 1])) + Param[2*D + 1]

  return lpdf
end
```

### Running AGESS

Now that we have specified a function to efficiently evaluate the log posterior density, we can 
use the `AGESS` function to generate samples from the posterior distribution. Similarly to before, 
we will specify an anonymous function with $\text{Param}$ as the only input to the function 
`log_posterior_HD`, using the global parameter values for $\mathbf{X}$, $\mathbf{y}$, and $D$.

```@example Regression
### Specify the dimension of the target distribution
P = 2*D + 2
### Specify the number of MCMC iterations
n_MCMC = 100000

### Run AGESS
results = AGESS(Param -> log_posterior_HD(Param, X, Y, D), n_MCMC, P)
```

After running `AGESS`, we can view the trace plots of the non-zero coefficients.

```@example Regression
plot(results.samps[25001:10:end, findall(β .!= 0)], legend = false)
hline!(β[findall(β .!= 0)], line = :dash, color =:black)
```

We can also view the trace plots of the coefficients that are equal to zero.

```@example Regression
plot(results.samps[25001:10:end, findall(β .== 0)], legend = false)
hline!(β[findall(β .== 0)], line = :dash, color =:black)
```

Next we will view the trace plot of $\sigma^2$.

```@example Regression
plot(exp.(results.samps[25001:10:end, 2*D + 2]).^2, legend = false)
hline!([1], line = :dash, color =:black)
```

Lastly, we can plot the log pdf of the posterior at every iteration of the Markov chain to potentially
detect convergence issues.

```@example Regression
plot(results.l_pdf[1:n_MCMC], legend = false)
```

## Generalize ReLU Regression

As discussed in the main manuscript[^1], AGESS can handle non-differentiable target functions.
In this section of the tutorial, we consider a generalized regression setting where the posterior
distribution is not differentiable everywhere. Therefore, alternative samplers such as HMC[^4] 
are not suitable for these types of target distributions. Let $\mathbf{Y} \in \mathbb{R}^N$ be
the response variable and $\mathbf{X} \in \mathbb{R}^{N \times D}$ be the covariates of interest.
Consider the following model, inspired by density discontinuity modeling[^5]:

$$Y_i \sim Bernoulli(\Phi(\mu_i)) \;\;\;\;\;\Phi(z) = \frac{e^z}{1 + e^z} \;\;\;\;\; \mu_i = \max(0, \mathbf{x}_i'\boldsymbol{\beta}),$$

for $i = 1, \dots, N$, where $\boldsymbol{\beta} \sim \mathcal{N}_{D}(\mathbf{0}, \mathbf{I})$.
We will first start by specifying a function to generate data from our model. We will consider
the simple case of when $D = 2$, allowing for easy visualization of the target distribution.

```@example Regression
function generate_data(N::T, P::T, ν::Y = 6.0) where {Y<:AbstractFloat, T<:Integer}
    β = randn(P) * (2 * log(P))^(1.0 / 4)
    β .*= sqrt(rand(Gamma(ν/2, 2/ ν)))
    x::Matrix{typeof(ν)} = randn(N, P) .+ randn() * 0.5
    μ = zeros(N)
    y::Vector{typeof(N)} = zeros(Int64, N)

    for i in 1:N
        μ[i] = max(0, dot(x[i,:], β))
        y[i] = rand(Binomial(1, LogExpFunctions.logistic(μ[i])), 1)[1]
    end


    return β, x, μ, y
end

Random.seed!(123)

D = 2
N = 1000
β, x, μ, y = generate_data(N, D)
```

### Specification of the Log Posterior Density

Now that we have generated the data, we can construct a function which evaluates the posterior 
log pdf. As in the previous sections, it is important that we specify a function that is efficiently
implemented. We will first start by constructing a function that evaluates the log likelihood,
and then a function that evaluates the log posterior density.

```@example Regression
function log_likelihood(β::AbstractVector{Y}, x::Matrix{Y}, 
                        y::Vector{T}) where {Y <:AbstractFloat, T<:Integer}
    log_lik::Float64 = 0.0
    z::Float64 = 0.0
    for i in eachindex(y)
        @views z = dot(x[i,:], β)
        if z < 0.0
            z = 0.0
        end
        log_lik -= log1p(exp(-(sign(y[i] - 0.5) * z)))
    end

    return log_lik
end

function log_posterior_ReLU(β::AbstractVector{Y}, x::Matrix{Y}, 
                            y::Vector{T}) where {Y <:AbstractFloat, T<:Integer}
    log_lik::Float64 = log_likelihood(β, x, y) - 0.5 * dot(β, β)
    return log_lik
end
```

### Running AGESS

Now that we have specified a function to efficiently evaluate the log posterior density, we can 
use the `AGESS` function to generate samples from the posterior distribution.

```@example Regression
### Specify the dimension of the target distribution
P = D
### Specify the number of MCMC iterations
n_MCMC = 10000

### Run AGESS
results = AGESS(β -> log_posterior_ReLU(β, x, y), n_MCMC, P)
```

Using the output of `AGESS`, we can visualize the samples from the target distribution.

```@example Regression
# Discard the initial 2500 samples to burnin
scatter(results.samps[2501:10000, 1], results.samps[2501:10000, 2], alpha = 0.1, 
        legend = false)
scatter!([β[1]], [β[2]], color = "red")
```

Additionally, we can plot the log pdf of the posterior at every iteration of the Markov chain to potentially
detect convergence issues.

```@example Regression
plot(results.l_pdf[1:n_MCMC], legend = false)
```


## Conclusion

Adaptive generalized elliptical slice sampling[^1] can be utilized in many generalized regression settings, 
including target distributions that are non-differentiable. While limited 
by computational resources in this tutorial, AGESS can be used to sample from relatively 
high-dimensional target distributions. Comparisons between AGESS and alternative samplers can 
be found in the main manuscript[^1].

[^1]: N. Marco and S. T. Tokdar. Adaptive generalized elliptical slice sampling. arXiv preprint arXiv:2605.21659, 2026.

[^2]: C. M. Carvalho, N. G. Polson, and J. G. Scott. Handling sparsity via the horseshoe. In Artificial intelligence and statistics, pages 73–80. PMLR, 2009.

[^3]: C. M. Carvalho, N. G. Polson, and J. G. Scott. The horseshoe estimator for sparse signals. Biometrika, pages 465–480, 2010.

[^4]: M. Betancourt. A conceptual introduction to hamiltonian monte carlo. arXiv preprint arXiv:1701.02434, 2017.

[^5]: S. T. Tokdar, R. Sen, H. Zheng, and S. Zhang. Density discontinuity regression. arXiv preprint arXiv:2507.05581, 2025.