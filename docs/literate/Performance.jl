# # Performance Tips

# Julia is a dynamic programming language that allows for high performance computing. However,
# Julia's optional typing can lead to slow performance if left unspecified. Since the `AGESS` function essentially
# only requires the user to specify a function evaluating the log target distribution, it is
# paramount that the user specifies an efficient implementation of this function, as this function
# will constantly be called in the `AGESS` function. Here, we will give a quick example of 3
# evaluations of the same target distribution; each leading to different computational
# costs. While a full guide to writing performance oriented code in Julia is out of the scope
# of this documentation, here are some useful resources:
#
# * [Performance Tips from Julia](https://docs.julialang.org/en/v1/manual/performance-tips/)
#
# * [Learning Resources for Julia](https://julialang.org/learning/)
#
# * [JET.jl](https://aviatesk.github.io/JET.jl/dev/)
#
# * [BenchmarkTools.jl](https://juliaci.github.io/BenchmarkTools.jl/stable/)

#md # [Download this page as a Jupyter notebook](notebooks/Performance.ipynb)

# ## Use of Turing Models

# As of version 0.2.0, users are able to specify a [Turing.jl](https://turinglang.org/) model instead of supplying a
# function that efficiently evaluates the log posterior density. Thus, we refer users to the `Turing.jl`
# documentation for tips on improving performance.
#
#   * [Performance Tips for Turing.jl](https://turinglang.org/docs/usage/performance-tips/)

# ## Linear Regression
#
# Consider the standard model for linear regression (see the "Regression" tutorial for more details):
#
# $$Y_i \sim \mathcal{N}(\mathbf{x}_i' \boldsymbol{\beta}, \sigma^2),$$
#
# $$\boldsymbol{\beta} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}),$$
#
# $$\sigma^2 \sim \text{Inv-Gamma}(1,1).$$
#
# Let's consider a simple implementation of this function, where we do not specify any types:

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

# Let's generate some synthetic data and benchmark how long it takes to run `lm_log_posterior_1`.

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

## Benchmark function
Param = ones(D + 1)
@benchmark lm_log_posterior_1($Param, $X, $y)

# Let's see if we can improve on this by pre-allocating some of our variables and specifying the
# type of variables.

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

## Pre-allocate parameters
μ = zeros(1000)
Σ_I_N = diagm(ones(1000))
Σ_I_P = diagm(ones(D))
μ_0 = zeros(D)
@benchmark lm_log_posterior_2($Param, $X, $y, $μ, $μ_0, $Σ_I_N, $Σ_I_P)

# We can see that there was a modest improvement in performance. We can see that we are allocating
# less memory. However, it is slow to actually construct these multivariate distributions and
# evaluate the log pdf of these distributions. We can just perform the calculations ourselves and
# get significantly better performance. Since Julia is compiled just-in-time, we should feel free
# to use for-loops as we please! (Unlike R)

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

# We can see that we have a 1000-fold speed-up by just efficiently evaluating the log posterior
# density. This directly translates into a similar magnitude increase in the effective sample size
# per second achieved by AGESS.
#
# !!! tip "Key takeaway"
#     It is paramount to write efficient functions that evaluate the log posterior density when using AGESS.
#
# Tips:
#
# * Packages like `JET.jl` can help catch inefficiencies in coding.
#
# * Pre-allocate variables (especially for intermediate computations)
#
# * `@views` can help reduce allocating new arrays when doing computations on subarrays

# ## Block Updates

# Writing an efficient `log_posterior` is crucial, but for high-dimensional target distributions
# we still perform 1-d updates during the burn-in stage, as well as randomly throughout (as controlled
# by `single_step_prop`). However, if your model has structure -- such as a hierarchical model
# where we have conditional independence between blocks of parameters and do not need to compute the entire
# posterior to calculate the conditional density -- we can significantly
# reduce the computational burden of these 1-d updates by using `AGESSSampler`'s `blocks`. Here,
# we will provide an example of fitting a hierarchical model using these `blocks` (using a direct
# specification of the log target density, and separately using the Turing.jl ecosystem).

# Consider a simple hierarchical model: 
#
# $$Y_{ig} \sim \mathcal{N}(\theta_g, 1), \qquad \theta_g \sim \mathcal{N}(\mu, 1).$$
#
# We can simulate data as follows.

G = 10
n_g = 200
μ_true = 1.0
θ_true = μ_true .+ 0.5 .* randn(G)
data = [θ_true[g] .+ randn(n_g) for g in 1:G]

# We will first start by explicitly specifying the blocks and the log target density.

## Param layout: Param[1] = μ, Param[1+g] = θ_g for g in 1:G.
function hier_log_posterior(Param::AbstractVector{Y}, data) where {Y<:AbstractFloat}
    μ = Param[1]
    lpdf = -0.5 * μ^2
    for g in eachindex(data)
        θ_g = Param[1 + g]
        lpdf += -0.5 * (θ_g - μ)^2
        for y in data[g]
            lpdf += -0.5 * (y - θ_g)^2
        end
    end
    return lpdf
end

# The full `hier_log_posterior` above touches every group's data on every call, even when only
# `θ_g` for one group is changing. A block's `conditional` gets to see the same full parameter
# vector, but only needs to return the parts of the density that actually depend on its own
# block. Here, updating `θ_g` only needs group `g`'s own data:

function group_conditional(g::Integer, Param::AbstractVector{Y}, data) where {Y<:AbstractFloat}
    μ = Param[1]
    θ_g = Param[1 + g]
    lpdf = -0.5 * (θ_g - μ)^2
    for y in data[g]
        lpdf += -0.5 * (y - θ_g)^2
    end
    return lpdf
end

# For $G=10$ groups of 200 observations each, that's roughly a 10-fold reduction in the amount
# of data touched per call:

Param = vcat(μ_true, θ_true)
@benchmark hier_log_posterior($Param, $data)
#-
@benchmark group_conditional(3, $Param, $data)

# Wiring this into `AGESSSampler` just means building one `AGESSBlock` per group with its
# `conditional`, plus a block for `μ` left on the default (full `hier_log_posterior`) path:
# `μ`'s own prior doesn't touch any `θ_g`, but each `θ_g`'s prior depends on `μ`, so `μ`'s full
# conditional needs every group's `θ_g` and so isn't separable the same way.

blocks = [AGESSBlock([1 + g]; conditional = p -> group_conditional(g, p, data)) for g in 1:G]
push!(blocks, AGESSBlock([1]))

n_MCMC = 10_000
chain = AGESS(p -> hier_log_posterior(p, data), n_MCMC, 1 + G; blocks = blocks)

# For Turing.jl models we can also utilize `AGESSSampler`'s `blocks`. Using the same setup,
# we can set up our Turing.jl model as follows:

using Turing

@model function local_model(data_g, μ)
    θ ~ Normal(μ, 1)
    data_g .~ Normal(θ, 1.0)
end

@model function full_model(data)
    G = length(data)
    μ ~ Normal(0.0, 1.0)
    θ ~ filldist(Normal(μ, 1.0), G)
    for g in 1:G
        for i in eachindex(data[g])
            data[g][i] ~ Normal(θ[g], 1.0)
        end
    end
end

## Note: We can also write the model using the local model, but we will get warnings for growable
## arrays. Note that these warnings do not affect the correctness of the sampling scheme.
## @model function full_model(data, G)
##    μ ~ Normal(0, 1)
##    θ = Vector{Float64}(undef, G)
##    for g in 1:G
##        θ[g] ~ to_submodel(local_model(data[g], μ))
##    end
## end

## μ at index 1; group g's θ at index 1 + g
blocks = [AGESSBlock([1 + g], ctx -> local_model(data[g], ctx[1]), [1]) for g in 1:G]
push!(blocks, AGESSBlock([1]))  # μ stays on the default (full log_posterior) path

agessB = AGESSSampler(full_model(data), n_MCMC; blocks = blocks)
chain_turing = sample(full_model(data), agessB, n_MCMC)

# !!! tip "Key takeaway"
#     If your model has natural conditional independence structure where part of the likelihood does
#     not depend on a block of parameters, `AGESSSampler`'s `blocks` keyword can
#     significantly reduce computational costs -- particularly in high-dimensional settings.
#     However, correctness depends on `conditional` actually being a valid restriction of `log_posterior`.
#     A block left without a `conditional` always falls back to the full `log_posterior`, so
#     when in doubt, leave it out rather than risk a subtly wrong `conditional`.
