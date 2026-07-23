# # Regression

# In this tutorial, we will illustrate how to use the `AdaptEllipticalSliceSampler.jl` package, to
# perform Bayesian computation using the adaptive generalized elliptical slice sampler[^1] in
# regression settings. We consider four settings in this tutorial: (1) Bayesian linear regression,
# (2) Bayesian generalized linear regression (3) high-dimensional linear sparse regression[^2] [^3],
# and (4) generalized ReLU regression.

#md # [Download this tutorial as a Jupyter notebook](notebooks/Regression.ipynb)

# ## Bayesian Linear Regression

# In this section, we consider using AGESS in the context of standard Bayesian linear regression.
# Letting $\mathbf{Y} \in \mathbb{R}^N$ be the response variable and $\mathbf{X} \in \mathbb{R}^{N \times D}$
# be the set of covariates, we can specify our model as follows:
#
# $$Y_i \sim \mathcal{N}(\mathbf{x}_i' \boldsymbol{\beta}, \sigma^2),$$
# $$\boldsymbol{\beta} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}),$$
# $$\sigma^2 \sim \text{Inv-Gamma}(1,1).$$
#
# Let's start by first generating data from our model.

import Random
import LogExpFunctions
using AdaptEllipticalSliceSampler
using Distributions
using Plots
using LinearAlgebra
using Turing, MCMCChains, StatsPlots

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

## Generate data with 1000 observations and 10 covariates
D = 10
β, X, y = generate_data(1000, D)

# We can see that we generated the data from our model under $\sigma = 0.5$ ($N = 1000$, $D = 10$).
# As of version 0.2.0 of `AdaptEllipticalSliceSampler.jl`, the package is now integrated into the
# `AbstractMCMC.jl` framework, allowing us to utilize the
# [Turing ecosystem](https://turinglang.org/docs/getting-started/index.html).
# Thus, we have two options: (1) we can write a function that calculates the log posterior density
# or (2) we can specify a `Turing.jl` model and directly use that to conduct Bayesian computation.
# While one can easily write a function for evaluating the log_posterior density (while making sure
# to transform $\sigma^2$ into an unconstrained space), let's look at how to use AGESS using a
# `Turing.jl` model. Luckily, `Turing.jl` (and the `AdaptEllipticalSliceSampler.jl` package) will
# automatically transform constrained parameters into an unconstrained space for sampling, then transform
# it back to the original space before returning the chain back to the user. Thus, in cases where we can use `Turing.jl`,
# we can not worry about remembering Jacobians!

@info "Starting Linear Regression"
@model function linear_regression(X::AbstractMatrix{Y}, y::AbstractVector{Y}) where {Y<:AbstractFloat}
    N, D = size(X)
    ## Make sure dimensions conform
    @assert length(y) == N

    ## Start with priors
    β ~ MvNormal(zeros(D), I)
    σ² ~ InverseGamma(1.0, 1.0)

    ## Specify Likelihood
    y ~ MvNormal(X * β,  σ² * I)

end;

model = linear_regression(X, y)

# Now that we have constructed the model, we can use AGESS to conduct Bayesian computation by
# simply calling two functions: `AGESSSampler()` and `sample()`.

n_MCMC = 10_000
sampler = AGESSSampler(model, n_MCMC)
results = sample(model, sampler, n_MCMC)

# We will start by discarding the initial part of the chain due to burn-in and will assess the
# results using `describe()`.

## Discard first 2500 iterations due to burn-in
results = results[2501:end,:,:]
describe(results)

# Notice how the parameter names that we specified also follow in the subsequent evaluations of the
# chain---this holds for calling `plot(results)`, `ess(results)`, `autocorplot(results)`, etc.
# instead of using the default plotting function, let's visualize only the positive coefficients
# along with the true value (due to the dimension).

## Plot trace plot
plot(results[:, findall(β .> 0),:])
## Plot true values, one per coefficient's own subplot
true_vals = β[findall(β .> 0)]
hline!(reshape(true_vals, 1, :), subplot = reshape(1:2:2*length(true_vals), 1, :),
       color = :red, linestyle = :dash)

# Similarly, we can view the trace plot of $\sigma^2$.

## Plot trace plot, don't forget to transform the transformed variables back
plot(results[:σ²], label = false)
## Plot true value
hline!([0.25], color = :red, label = false)

# Lastly, we can plot the log posterior density at every iteration of the Markov chain to potentially
# detect convergence issues.

plot(results[:lp], legend = false)

# ## High-Dimensional Sparse Linear Regression

# In many modern applications, the number of covariates may be larger than the sample size ($N < D$),
# often requiring regularization in order to achieve good predictive properties. Here we consider
# the sparse regression setting, where sparsity is induced via continuous shrinkage through global-local
# shrinkage priors. Here, we consider the horseshoe prior[^2] [^3]. Specifically, we consider the following
# model:
#
# $$Y_i \sim \mathcal{N}(\mathbf{x}_i'\boldsymbol{\beta}, \sigma^2)$$
#
# $$\beta_j \sim \mathcal{N}(0, \sigma^2\tau^2\lambda_j^2)  \;\;\;\;\; p(\sigma^2) \propto \frac{1}{\sigma^2},$$
#
# $$\tau \sim C^+(0,1) \;\;\;\;\; \lambda_j \sim C^+(0,1),$$
#
# for $i = 1, \dots, N$ and $j = 1, \dots, D$, where $C^+$ denotes the half-Cauchy distribution.
# We will first start by generating data, where the design matrix has correlated covariates. We
# will consider the case where we have 25 observations and 25 covariates. Here we will generate
# a dataset where the probability of $\beta_j = 0$ is $0.9$ ($1 \le j \le 25$).

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

X, y, β = gen_data_AR1(N, D, ρ = 0.7, sparsity = 0.9, σ_sq = 1.0)

# We visualize the correlation structure of the design matrix.

heatmap(cor(X))

# We can also view the values of $\boldsymbol{\beta}$.

scatter(β,legend = false)

# ### Specification of the Model

# While we could directly specify a function evaluating the log posterior density (remembering to
# log-transform $\tau$ and $\lambda$), we will instead simply specify the model using the `Turing.jl`
# framework.

@info "Starting Horse Shoe"
@model function HS_regression(X::AbstractMatrix{Y}, y::AbstractVector{Y}) where {Y<:AbstractFloat}
    N,D = size(X)
    @assert length(y) == N

    ## Define Half-Cauchy distribution
    half_cauchy = truncated(Cauchy(0, 1); lower=0)

    ## Priors
    τ ~ half_cauchy                 # Global Shrinkage param
    λ ~ filldist(half_cauchy, D)    # Local Shrinkage param
    log_σ² ~ Flat()                 # Flat prior to create Jeffrey's prior
    σ² = exp(log_σ²)
    β ~ MvNormal(zeros(D), σ² .* Diagonal((λ .* τ).^2))

    ## Likelihood
    y ~ MvNormal(X * β, σ² * I)
end

model = HS_regression(X, y)

# ### Running AGESS

# Similarly to before, once we have constructed the model, we can sample by simply calling two
# functions.

## Specify the number of MCMC iterations
n_MCMC = 100_000

## Run AGESS
sampler = AGESSSampler(model, n_MCMC)
results = sample(model, sampler, n_MCMC)

# After running `AGESS`, we can view the trace plots of the non-zero coefficients.

## Discard initial 25% of chain due to burn-in
results = results[25_001:end,:,:]

## Get β parameters
β_samps = get(results, :β)
p = plot()
for i in findall(β .!= 0)
    p = plot!(β_samps.β[i][1:10:end], legend = false)
end

p
hline!(β[findall(β .!= 0)], line = :dash, color =:black)

# We can also view the trace plots of the coefficients that are equal to zero.

p = plot()
for i in findall(β .== 0)
    p = plot!(β_samps.β[i][1:10:end], legend = false)
end

p
hline!([0.0], line = :dash, color =:black)

# Next we will view the trace plot of $\sigma^2$.

plot(exp.(results[:log_σ²][1:10:end]), legend = false)
hline!([1], line = :dash, color =:black)

# Lastly, we can plot the log pdf of the posterior at every iteration of the Markov chain to potentially
# detect convergence issues.

plot(results[:lp], legend = false)

# ## Generalized ReLU Regression

# As discussed in the main manuscript[^1], AGESS can handle non-differentiable target functions.
# In this section of the tutorial, we consider a generalized regression setting where the posterior
# distribution is not differentiable everywhere. Therefore, alternative samplers such as HMC[^4]
# are not suitable for these types of target distributions. Let $\mathbf{Y} \in \mathbb{R}^N$ be
# the response variable and $\mathbf{X} \in \mathbb{R}^{N \times D}$ be the covariates of interest.
# Consider the following model, inspired by density discontinuity modeling[^5]:
#
# $$Y_i \sim Bernoulli(\Phi(\mu_i)) \;\;\;\;\;\Phi(z) = \frac{e^z}{1 + e^z} \;\;\;\;\; \mu_i = \max(0, \mathbf{x}_i'\boldsymbol{\beta}),$$
#
# for $i = 1, \dots, N$, where $\boldsymbol{\beta} \sim \mathcal{N}_{D}(\mathbf{0}, \mathbf{I})$.
# We will first start by specifying a function to generate data from our model. We will consider
# the simple case of when $D = 2$, allowing for easy visualization of the target distribution.

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

# ### Specification of the Log Posterior Density

# Now that we have generated the data, we can construct a function which evaluates the posterior
# log pdf. As in the previous sections, it is important that we specify a function that is efficiently
# implemented. We will first start by constructing a function that evaluates the log likelihood,
# and then a function that evaluates the log posterior density.

@info "Starting ReLU"
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

# ### Running AGESS

# Now that we have specified a function to efficiently evaluate the log posterior density, we can
# use the `AGESS` function to generate samples from the posterior distribution.

## Specify the dimension of the target distribution
P = D
## Specify the number of MCMC iterations
n_MCMC = 10_000

## Let's specify the param_names this time
param_names = [string("β_",i) for i in 1:D]
## Run AGESS
results = AGESS(β -> log_posterior_ReLU(β, x, y), n_MCMC, P; param_names = param_names)

# Discard the initial 2500 samples to burnin
results = results[2501:end,:,:]
describe(results)

# Using the output of `AGESS`, we can visualize the samples from the target distribution.

scatter(results[:β_1], results[:β_2], alpha = 0.1,
        legend = false)
scatter!([β[1]], [β[2]], color = "red")

# Additionally, we can plot the log pdf of the posterior at every iteration of the Markov chain to potentially
# detect convergence issues.

plot(results[:lp], legend = false)
@info "Ending ReLU"

# ## Conclusion

# Adaptive generalized elliptical slice sampling[^1] can be utilized in many generalized regression settings,
# including target distributions that are non-differentiable. While limited
# by computational resources in this tutorial, AGESS can be used to sample from relatively
# high-dimensional target distributions. Comparisons between AGESS and alternative samplers can
# be found in the main manuscript[^1].

# [^1]: N. Marco and S. T. Tokdar. Adaptive generalized elliptical slice sampling. arXiv preprint arXiv:2605.21659, 2026.
#
# [^2]: C. M. Carvalho, N. G. Polson, and J. G. Scott. Handling sparsity via the horseshoe. In Artificial intelligence and statistics, pages 73–80. PMLR, 2009.
#
# [^3]: C. M. Carvalho, N. G. Polson, and J. G. Scott. The horseshoe estimator for sparse signals. Biometrika, pages 465–480, 2010.
#
# [^4]: M. Betancourt. A conceptual introduction to hamiltonian monte carlo. arXiv preprint arXiv:1701.02434, 2017.
#
# [^5]: S. T. Tokdar, R. Sen, H. Zheng, and S. Zhang. Density discontinuity regression. arXiv preprint arXiv:2507.05581, 2025.
