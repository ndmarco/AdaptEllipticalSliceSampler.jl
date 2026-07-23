# # Bayesian Neural Network

# Consider a Bayesian feedforward neural network[^1] with $L$ hidden layers. Letting $x_i$ be the input vector and $y_i$ be the observed output, we will assume the following model:

# $$y_i \mid x_i, \theta \sim \mathcal{N}(f(x_i; \theta), 1 / \beta),$$

# where 

# $$f(x;\theta) = W^{L+1}h\left(W^{L}h\left(W^{L-1} \dots h(W^{1}x + b^{1})\dots \right) + b^{L} \right) + b^{L+1}.$$

# Here, $h$ is a non-linear activation function applied element-wise (taken to be $\tanh$ in this tutorial), $W^{l}$ are the weights associated with the $l^{th}$ layer, and $b^{l}$ are the biases associated with the $l^{th}$ layer. Let the widths of each layer be $\{n_0, n_1, \dots, n_L, n_{L+1}\}$ where $n_0$ and $n_{L+1}$ correspond to the input dimension and output dimension, respectively. Following Neal, we will use the following prior specification

# $$w^{1}_{jk} \sim \mathcal{N}(0, 1 / \alpha_1), \qquad w^{l}_{ji} \sim \mathcal{N}(0, 1/(\alpha_l \times n_{l-1}))\quad (l \ge 2), \qquad b \sim \mathcal{N}(0, \sigma^2_b).$$

# We will assume that $\sigma^2_b = 1$ is fixed, and put hyperpriors on $\alpha_l$ and $\beta$ such that $\alpha_l \sim \text{Gamma}(a,b)$ and $\beta \sim \text{Gamma}(a,b)$ $(l = 1, \dots, L +1)$. To use AGESS[^2], we need only specify the log target density.

#md # [Download this tutorial as a Jupyter notebook](notebooks/BayesNN.ipynb)

# ### Specifying the log posterior density

# We start by defining a struct which contains all of the information of the network architecture. In addition, we will specify a function which returns the total number of parameters in our model (n_params) and a function which evaluates $f(x; \theta)$ (forward!). Using these functions, we will build a function that evaluates the transformed log posterior density; taking in a flattened vector of parameters (along with other variables used for intermediate calculations to save on memory allocation) and returning the transformed log posterior density. Here, the scale parameters are transformed to the log-scale so all variables are unbounded.

using LinearAlgebra
using Distributions
using Random
using SpecialFunctions: loggamma

@info "Starting Bayesian NN"
struct BNNArchitecture
    sizes::Vector{Int}     # [n_0, n_1, ..., n_L, n_out]
end

function n_params(arch::BNNArchitecture)
    params = 0
    ## Iterate over layers
    for l in 1:(length(arch.sizes) - 1)
        ## Number of weights
        params += arch.sizes[l + 1] * arch.sizes[l]
        ## Number of biases
        params += arch.sizes[l + 1]
    end
    params += (length(arch.sizes) - 1) # precision parameters
    params += 1 # precision in likelihood β

    return params
end

## Forward pass: tanh hidden layer
function forward!(y::AbstractMatrix{Y}, W::AbstractVector{<:AbstractMatrix{Y}}, 
                  b::AbstractVector{<:AbstractVector{Y}}, X::AbstractMatrix{Y}, 
                  H_ph::AbstractVector{<:AbstractMatrix{Y}}) where {Y<:AbstractFloat}
    ## X is n_in x n_samples
    H_ph[1] .= X
    for i in 1:(length(W) -1)
        H_ph[i+1] .= tanh.(W[i] * H_ph[i] .+ b[i])
    end
    y .= W[end] * H_ph[end] .+ b[end]           # n_out x n_samples
end

## ---- Hyperpriors on precisions: Gamma(shape, rate) on α1, α2, β ----
function log_gamma_hyperprior(log_α::Y, α::Y, shape::Y, rate::Y) where {Y<:AbstractFloat}
    lp = shape * log(rate) - loggamma(shape) +
         (shape - 1) * log_α - rate * α +  log_α   # Jacobian of α = exp(log_α)
    return lp
end

## ----------------------------------------------------------------------
## Calculation of the log-posterior; taking a flattened vector of parameters.
## ----------------------------------------------------------------------

function transformed_log_posterior(θ::AbstractVector{Y}, X::AbstractMatrix{Y}, y::AbstractMatrix{Y}, 
                                   W_ph::AbstractVector{<:AbstractMatrix{Y}},
                                   b_ph::AbstractVector{<:AbstractVector{Y}}, log_α::AbstractVector{Y}, 
                                   H_ph::AbstractVector{<:AbstractMatrix{Y}}, Y_ph::AbstractMatrix{Y}, 
                                   resid::AbstractMatrix{Y}, arch::BNNArchitecture, gamma_shape::Y, 
                                   gamma_rate::Y, σ2b::Y) where {Y<:AbstractFloat}

    ## ----------------------------------------------------------------------
    ## Unflatten
    ## ----------------------------------------------------------------------
    
    ## index
    idx = 1

    for l in 1:(length(arch.sizes)-1)
        ## get weights
        nW = arch.sizes[l+1] * arch.sizes[l]
        @views W_ph[l] .= reshape(θ[idx:idx+nW-1],  arch.sizes[l+1], arch.sizes[l]) 
        idx += nW

        ## get biases 
        @views b_ph[l] .= θ[idx:idx+arch.sizes[l+1]-1]
        idx += arch.sizes[l+1]
    end

    ## get precision for hyperpriors
    for l in 1:(length(arch.sizes)-1)
        log_α[l] = θ[idx]
        idx += 1
    end

    ## get precision for likelihood
    log_β  = θ[idx]
    idx += 1

    ## ----------------------------------------------------------------------
    ## Calculate log posterior density
    ## ----------------------------------------------------------------------
    forward!(Y_ph, W_ph, b_ph, X, H_ph)
    resid .= y .- Y_ph
    lpdf = 0.5 * (length(resid) * log_β - exp(log_β) * sum(abs2, resid))

    
    for l in 1:(length(arch.sizes)-1)
        ## prior on weights
        nW = arch.sizes[l+1] * arch.sizes[l]
        if l == 1
            lpdf += 0.5 * (nW * log_α[l] - exp(log_α[l]) * sum(abs2, W_ph[l]))
        else
            ## scale variance by the number of nodes in previous layer
            lpdf += 0.5 * (nW * (log_α[l] + log(arch.sizes[l])) - exp(log_α[l]) * arch.sizes[l] * sum(abs2, W_ph[l]))
        end
        
        ## prior on biases
        lpdf += 0.5 * (-sum(abs2, b_ph[l]) / σ2b)
        ## Prior on precision parameters
        lpdf += log_gamma_hyperprior(log_α[l], exp(log_α[l]), gamma_shape, gamma_rate)
    end

    ## Prior on precision parameter
     lpdf += log_gamma_hyperprior(log_β, exp(log_β), gamma_shape, gamma_rate)


    return lpdf
end
# ### Case Study: Yacht Hydrodynamics

# We will explore the utility of Bayesian neural networks by considering the Yacht Hydrodynamics dataset. The dataset contains 308 observations with 6 predictor variables and 1 response variable. We will standardize the dataset, and randomly split the dataset into a training set and a test set, so that we can evaluate the performance of the neural network.

using DelimitedFiles
using StatsBase
using AdaptEllipticalSliceSampler
using Plots

Random.seed!(123)
## ----------------------------------------------------------------------
## Read data
## ----------------------------------------------------------------------
## Data can be found at http://archive.ics.uci.edu/ml/machine-learning-databases/00243/yacht_hydrodynamics.data
data_path = joinpath(pkgdir(AdaptEllipticalSliceSampler), "docs", "literate", "data", "yacht_hydrodynamics.data")
data = readdlm(data_path)
X = Matrix(data[:, 1:6]')      # 6 x 308
y = reshape(data[:, 7], 1, :)  # 1 x 308

y_mean = mean(y); y_std = std(y)
y = (y .- y_mean) ./ y_std
## Standardize dataset
X = zscore(X, 2) 


## Split into training and testing 
perm = randperm(size(X, 2))
n_test = 50
test_idx  = perm[1:n_test]
train_idx = perm[n_test+1:end]
 
X_train = X[:, train_idx]
y_train = y[:, train_idx]
X_test = X[:, test_idx]
y_test = y[:, test_idx]

n_obs = size(X_train, 2)

## Specify architecture of neural network
## ----------------------------------------------------------------------
## Input = 6
## Hidden_1 = 8
## Hidden_2 = 10
## Output = 1
## ----------------------------------------------------------------------
arch = BNNArchitecture([6, 8, 10, 1]) 
P = n_params(arch)

## Specify intermediate values used in likelihood calculation
b_ph = Vector{Vector{Float64}}(undef, length(arch.sizes) - 1)
H_ph = Vector{Matrix{Float64}}(undef, length(arch.sizes) - 1)
W_ph = Vector{Matrix{Float64}}(undef, length(arch.sizes) - 1)
log_α = zeros(length(arch.sizes) - 1)
for l in 1:(length(arch.sizes) - 1)
    b_ph[l] = zeros(arch.sizes[l+1])
    H_ph[l] = zeros(arch.sizes[l], n_obs)
    W_ph[l] = zeros(arch.sizes[l+1], arch.sizes[l])
end
Y_ph = similar(y_train)
resid = similar(y_train)

## Hyperpriors
gamma_shape = 2.0
gamma_rate = 2.0

println("--- Number of Parameters = ", P, " ---- \n")

# Now that we have all of the necessary function and variables specified, we can use the AGESS function to conduct MCMC.

## ----------------------------------------------------------------------
## Run AGESS
## ----------------------------------------------------------------------
n_MCMC = 20000  # Should run for longer in realistic situations
results = AGESS(θ -> transformed_log_posterior(θ, X_train, y_train, W_ph, b_ph, log_α, 
                                               H_ph, Y_ph, resid, arch, 
                                               gamma_shape,  gamma_rate, 1.0), 
                n_MCMC, P)

# By plotting the log target density, we can see that our Markov chain appears to converge relatively fast. However, we will conservatively discard the first 50\% of the chain and use the remaining 50\% for inference.

plot(results[:lp], label = "log target density")
vline!([10000], color =:red, label = "burnin")

# ### Posterior Predictive Distributions

# One of the main benefits of using a Bayesian neural network over a traditional neural network is that we obtain a full posterior distribution (instead of just maximizing the weights as is done in traditional neural networks), which allows us to utilized the posterior predictive distribution to evaluate the uncertainty around a new data point. Let's first take a look at the posterior predictive mean. While harder to visualize when there is more than one covariate, we will look at the posterior predictive mean as a function of the Froude number (one of the predictors), while holding all other variables at their mean value.

## Function for generating posterior predictive draws
function posterior_predictive_draws(post_samples::AbstractMatrix{Y}, X_grid::AbstractMatrix{Y},
                                    W_ph::AbstractVector{<:AbstractMatrix{Y}},
                                    b_ph::AbstractVector{<:AbstractVector{Y}}, log_α::AbstractVector{Y}, 
                                    H_ph::AbstractVector{<:AbstractMatrix{Y}}, Y_ph::AbstractMatrix{Y},
                                    arch::BNNArchitecture) where {Y<: AbstractFloat}
    n_s = size(post_samples,1)
    n_g = size(X_grid, 2)
    draws = zeros(n_s, n_g)        # noisy draws
    f_means = zeros(n_s, n_g)      # noiseless function draws
    for s in 1:n_s
        ## ----------------------------------------------------------------------
        ## Unflatten
        ## ----------------------------------------------------------------------
        
        ## index
        idx = 1
    
        for l in 1:(length(arch.sizes)-1)
            ## get weights
            nW = arch.sizes[l+1] * arch.sizes[l]
            @views W_ph[l] .= reshape(post_samples[s, idx:idx+nW-1],  arch.sizes[l+1], arch.sizes[l]) 
            idx += nW
    
            ## get biases 
            @views b_ph[l] .= post_samples[s, idx:idx+arch.sizes[l+1]-1]
            idx += arch.sizes[l+1]
        end
    
        ## get precision for hyperpriors
        for l in 1:(length(arch.sizes)-1)
            log_α[l] = post_samples[s, idx]
            idx += 1
        end
    
        ## get precision for likelihood
        log_β  = post_samples[s, idx]
        idx += 1

        ## ----------------------------------------------------------------------
        ## Run forward Model
        ## ----------------------------------------------------------------------
        forward!(Y_ph, W_ph, b_ph, X_grid, H_ph)
        
        @views f_means[s, :] .= vec(Y_ph)
        σ = sqrt(1 / exp(log_β))
        
        ## Add error term
        @views draws[s, :] .= f_means[s, :] .+ σ .* randn(n_g)
    end
    return draws, f_means
end

froude_idx = 6
## Build a grid over Froude number for visualization, holding other
## features fixed at their mean
froude_grid_std = collect(range(minimum(X[froude_idx, :]),
                         maximum(X[froude_idx, :]), length=100))
n_grid = length(froude_grid_std)
X_grid = zeros(6, n_grid)
@views X_grid[froude_idx, :] .= froude_grid_std


### rest H and Y_ph based on number in grid
H_ph1 = Vector{Matrix{Float64}}(undef, length(arch.sizes) - 1)
for l in 1:(length(arch.sizes) - 1)
    H_ph1[l] = zeros(arch.sizes[l], n_grid)
end
Y_ph1 = zeros(1, n_grid)
## Use function to get posterior draws
@views draws, f_means = posterior_predictive_draws(results.value[10000:end,:,1], X_grid, W_ph, 
                                                   b_ph, log_α, H_ph1, Y_ph1, arch)

 
pred_mean   = vec(mean(f_means, dims=1))          # E[f(x)] over posterior
pred_lower  = [quantile(f_means[:, j], 0.025) for j in 1:n_grid]
pred_upper  = [quantile(f_means[:, j], 0.975) for j in 1:n_grid]
predictive_lower = [quantile(draws[:, j], 0.025) for j in 1:n_grid]   # includes noise
predictive_upper = [quantile(draws[:, j], 0.975) for j in 1:n_grid]

p1 = plot(froude_grid_std, predictive_upper, fillrange=predictive_lower,
          fillalpha=0.15, linealpha=0, color=:steelblue,
          label="95% predictive interval (incl. noise)")
plot!(p1, froude_grid_std, pred_upper, fillrange=pred_lower,
      fillalpha=0.55, linealpha=0, color=:steelblue,
      label="95% credible interval (f only)")
plot!(p1, froude_grid_std, pred_mean, color=:navy, linewidth=2,
      label="Posterior predictive mean")
scatter!(p1, X_train[froude_idx, :], y_train[1, :],
         color=:black, markersize=4, label="Training data")
scatter!(p1, X_test[froude_idx, :], y_test[1, :],
         color=:red, markersize=5, markershape=:diamond, label="Held-out test data")
xlabel!(p1, "Froude number")
ylabel!(p1, "Residuary resistance")
title!(p1, "Posterior predictive: BNN fit to Yacht data")

# By looking at this plot, we can see that there is very little variability in the residual resistance when the Froude number is small; however, the variability increases as the Froude number increases. Thus, the predictive intervals look too wide when the Froude number is small, and too narrow when the Froude number is big. However, it is important to recall that this is a view of the conditional posterior predictive distribution (conditional on all other covariates at their mean value). Thus, it is possible that perhaps the other variables can explain the increased variability observed at high Froude numbers. Therefore, let's look at the posterior predictive coverage on the test dataset to get a better sense of what is happening.

function predictive_interval_at(post_samples::AbstractMatrix{Y}, x_i::AbstractVector{Y},
                                 W_ph::AbstractVector{<:AbstractMatrix{Y}},
                                 b_ph::AbstractVector{<:AbstractVector{Y}},
                                 log_α::AbstractVector{Y}, H_ph_i::AbstractVector{<:AbstractMatrix{Y}},
                                 Y_ph::AbstractMatrix{Y},
                                 arch::BNNArchitecture;
                                 q::Vector{Float64} = [0.025, 0.975],
                                 n_draw::Int = 2000) where {Y<:AbstractFloat}
    n_s = size(post_samples, 1)
    draws_i = Vector{Float64}(undef, n_draw)
    x_mat = reshape(x_i, :, 1)   # reshape to n_in x 1 matrix for forward!
    for k in 1:n_draw
        s = rand(1:n_s)
        ## ----------------------------------------------------------------------
        ## Unflatten
        ## ----------------------------------------------------------------------
        
        ## index
        idx = 1
    
        for l in 1:(length(arch.sizes)-1)
            ## get weights
            nW = arch.sizes[l+1] * arch.sizes[l]
            @views W_ph[l] .= reshape(post_samples[s, idx:idx+nW-1],  arch.sizes[l+1], arch.sizes[l]) 
            idx += nW
    
            ## get biases 
            @views b_ph[l] .= post_samples[s, idx:idx+arch.sizes[l+1]-1]
            idx += arch.sizes[l+1]
        end
    
        ## get precision for hyperpriors
        for l in 1:(length(arch.sizes)-1)
            log_α[l] = post_samples[s, idx]
            idx += 1
        end
    
        ## get precision for likelihood
        log_β  = post_samples[s, idx]
        idx += 1

        forward!(Y_ph, W_ph, b_ph, x_mat, H_ph_i)
        σ_i = sqrt(1 / exp(log_β))
        draws_i[k] = Y_ph[1] + σ_i * randn()
    end
    return quantile(draws_i, q[1]), quantile(draws_i, q[2])
end

### rest H and Y_ph based on number in grid
H_ph_i = Vector{Matrix{Float64}}(undef, length(arch.sizes) - 1)
for l in 1:(length(arch.sizes) - 1)
    H_ph_i[l] = zeros(arch.sizes[l], 1)
end
Y_ph_i = zeros(1, 1)

lo_vec = zeros(n_test)
hi_vec = zeros(n_test)
covered = falses(n_test)

for i in 1:n_test
    lo, hi = predictive_interval_at(results.value[10000:end, :, 1], X_test[:, i], W_ph,
                                     b_ph, log_α, H_ph_i, Y_ph_i, arch)
    lo_vec[i] = lo
    hi_vec[i] = hi
    covered[i] = (lo <= y_test[1, i] <= hi)
end

println("Empirical coverage (target 0.95): ", mean(covered))

## Sort by Froude number for a readable x-axis
order = sortperm(X_test[froude_idx, :])

p_coverage = plot(legend=:topleft, xlabel="Froude number",
                   ylabel="Residuary resistance (standardized)",
                   title="Posterior predictive coverage on held-out data")

## Error bars: green if covered, red if missed
for (rank, i) in enumerate(order)
    color = covered[i] ? :green : :red
    plot!(p_coverage, [X_test[froude_idx, i], X_test[froude_idx, i]],
          [lo_vec[i], hi_vec[i]], color=color, linewidth=2, label=false, alpha=0.7)
end

scatter!(p_coverage, X_test[froude_idx, :], y_test[1, :],
         group=ifelse.(covered, "Covered", "Missed"),
         color=ifelse.(covered, :green, :red),
         markersize=4, markerstrokewidth=0.5)

##display(p_coverage)

nominal_levels = 0.5:0.05:0.99
empirical_coverage = zeros(length(nominal_levels))

for (li, level) in enumerate(nominal_levels)
    alpha = 1 - level
    q_lo, q_hi = alpha/2, 1 - alpha/2
    covered_at_level = falses(n_test)
    for i in 1:n_test
        lo, hi = predictive_interval_at(results.value[10000:end, :, 1], X_test[:, i], W_ph,
                                         b_ph, log_α, H_ph_i, Y_ph_i, arch;
                                         q=[q_lo, q_hi])
        covered_at_level[i] = (lo <= y_test[1, i] <= hi)
    end
    empirical_coverage[li] = mean(covered_at_level)
    println("Nominal $(round(level,digits=2)) -> empirical $(round(empirical_coverage[li],digits=3))")
end

p_calib = plot(nominal_levels, nominal_levels, linestyle=:dash, color=:black,
               label="Perfect calibration", xlabel="Nominal coverage",
               ylabel="Empirical coverage", title="Calibration curve",
               legend=:topleft, aspect_ratio=:equal, xlims=(0.45,1.0), ylims=(0.45,1.0))
plot!(p_calib, nominal_levels, empirical_coverage, marker=:circle, color=:navy,
      label="BNN predictive intervals", linewidth=2)

## ---- Combined figure -------------------------------------------------
p_combined = plot(p_coverage, p_calib, layout=(1,2), size=(1100,450),
                   left_margin=5Plots.mm, bottom_margin=5Plots.mm)
p_combined

# From the graphs above, we can see that we are overly conservative in our coverage, especially when looking at smaller credible intervals (i.e., 50 - 70% credible intervals). Recall that our model assumes a homoskedastic error term ($1/\beta$). Let's first check to see if a homoskedastic error term $\beta$ is appropriate in this setting by plotting the residual values against the Froude number.

H_ph_test = Vector{Matrix{Float64}}(undef, length(arch.sizes) - 1)
for l in 1:(length(arch.sizes) - 1)
    H_ph_test[l] = zeros(arch.sizes[l], n_test)
end
Y_ph_test = zeros(1, n_test)

@views draws_test, f_means_test = posterior_predictive_draws(
    results.value[10000:end, :, 1], X_test, W_ph, b_ph, log_α, H_ph_test, Y_ph_test, arch)

## draws_test is n_post_samples x n_test; compute per-point predictive
## mean/sd from the full (noise-included) posterior predictive draws
pred_mean_test = vec(mean(draws_test, dims=1))
pred_sd_test   = vec(std(draws_test, dims=1))

z_scores = (vec(y_test) .- pred_mean_test) ./ pred_sd_test

## ---- Plot: z-scores vs Froude number ----------
order = sortperm(X_test[froude_idx, :])

p_z = scatter(X_test[froude_idx, :], z_scores, color=:navy, markersize=4,
              label="Test points", xlabel="Froude number",
              ylabel="Standardized residual (z)",
              title="Calibration check: z-scores vs. Froude number",
              legend=:topleft)
hline!(p_z, [0], color=:black, linestyle=:dash, label=false)
hline!(p_z, [-2, 2], color=:red, linestyle=:dot, label="±2 SD")

# By plotting the standardized residuals against the Froude number, we can see that the assumption of a homoskedastic error may not be appropriate in this setting. Specifically, we can see that high Froude numbers lead to larger residual values. One possible solution is to allow for a more complex neural network (i.e., increasing the depth and the width of the neural network). Perhaps a more complex function could explain more of the heterogeneity observed, leading to residuals that are homoskedastic. However, increasing the depth and width did not help with this problem. Instead, let's consider a Bayesian neural network with heterogeneous error terms.

# ## Heteroskedastic Bayesian Neural Network

# Consider the following heteroskedastic Bayesian neural network, which uses two different neural networks to capture the mean function and variance function:

# $$y_i \mid x_i, \theta \sim \mathcal{N}(f(x_i; \theta), \exp(g(x_i; \theta))^{-1}),$$

# where

# $$f(x;\theta) = W_f^{L+1}h\left(W^{L_f}h\left(W_f^{L_f-1} \dots h(W^{1}x + b_f^{1})\dots \right) + b_f^{L_f} \right) + b_f^{L+1},$$

# and

# $$g(x;\theta) = W_g^{L+1}h\left(W^{L_g}h\left(W_g^{L_g-1} \dots h(W_g^{1}x + b_g^{1})\dots \right) + b_g^{L_g} \right) + b_g^{L+1}.$$

# Notice that each neural network is assumed to have its own network architecture. We will use a similar prior structure as before:

# $$(w^{1}_f)_{jk} \sim \mathcal{N}(0, 1 / \alpha_1), \qquad (w^{l}_f)_{ji} \sim \mathcal{N}(0, 1/ (\alpha_l \times n_{l-1}))\quad (l \ge 2), $$

# $$(w^{1}_g)_{jk} \sim \mathcal{N}(0, 1 / \alpha_1), \qquad (w^{l}_g)_{ji} \sim \mathcal{N}(0, 1/ (\alpha_l \times n_{l-1}))\quad (l \ge 2),$$

# $$b \sim \mathcal{N}(0, \sigma^2_b);$$

# As before, we will use relatively uninformative priors on $\alpha_l$ (i.e., $\alpha_l \sim \text{Gamma}(a,b)$). Let's start by constructing the necessary functions to evaluate the likelihood and calculate the total number of parameters.

### New struct allowing for heterogenous model
### Can also handle homogeneous model
struct HBNNArchitecture
    sizes_mean::Vector{Int}     # [n_0, n_1, ..., n_L, n_out]
    sizes_var::Vector{Int}     # [n_0, n_1, ..., n_L, n_out]
    heterogeneous::Bool    # true or false
end

### overload function for heterogeneous model
function n_params(arch::HBNNArchitecture)
    params = 0
    ## Iterate over layers for mean NN
    for l in 1:(length(arch.sizes_mean) - 1)
        ## Number of weights
        params += arch.sizes_mean[l + 1] * arch.sizes_mean[l]
        ## Number of biases
        params += arch.sizes_mean[l + 1]
    end
    ## precision parameters for mean
    params += length(arch.sizes_mean) - 1
    
    if arch.heterogeneous == true
        ## Iterate over layers for var NN
        for l in 1:(length(arch.sizes_var) - 1)
            ## Number of weights
            params += arch.sizes_var[l + 1] * arch.sizes_var[l]
            ## Number of biases
            params += arch.sizes_var[l + 1]
        end

        ## precision parameters for var
        params += length(arch.sizes_var) - 1
    else
        params += 1
    end

    return params
end

## ----------------------------------------------------------------------
## Calculation of the log-posterior of heterogeneous model
## ----------------------------------------------------------------------

function transformed_log_posterior(θ::AbstractVector{Y}, X::AbstractMatrix{Y}, y::AbstractMatrix{Y}, 
                                   W_f::AbstractVector{<:AbstractMatrix{Y}}, W_g::AbstractVector{<:AbstractMatrix{Y}},
                                   b_f::AbstractVector{<:AbstractVector{Y}}, b_g::AbstractVector{<:AbstractVector{Y}},
                                   log_α_f::AbstractVector{Y}, log_α_g::AbstractVector{Y},
                                   H_f::AbstractVector{<:AbstractMatrix{Y}}, H_g::AbstractVector{<:AbstractMatrix{Y}},
                                   f::AbstractMatrix{Y}, g::AbstractMatrix{Y}, resid::AbstractMatrix{Y}, 
                                   arch::HBNNArchitecture, gamma_shape::Y, gamma_rate::Y, 
                                   gamma_shape_g::Y, gamma_rate_g::Y, σ2b::Y) where {Y<:AbstractFloat}

    ## ----------------------------------------------------------------------
    ## Unflatten
    ## ----------------------------------------------------------------------
    
    ## index
    idx = 1
    log_β = 0.0

    for l in 1:(length(arch.sizes_mean)-1)
        ## get weights
        nW = arch.sizes_mean[l+1] * arch.sizes_mean[l]
        @views W_f[l] .= reshape(θ[idx:idx+nW-1],  arch.sizes_mean[l+1], arch.sizes_mean[l]) 
        idx += nW

        ## get biases 
        @views b_f[l] .= θ[idx:idx+arch.sizes_mean[l+1]-1]
        idx += arch.sizes_mean[l+1]
    end

    ## get precision for hyperpriors
    for l in 1:(length(arch.sizes_mean)-1)
        log_α_f[l] = θ[idx]
        idx += 1
    end

    if arch.heterogeneous == true
        for l in 1:(length(arch.sizes_var)-1)
            ## get weights
            nW = arch.sizes_var[l+1] * arch.sizes_var[l]
            @views W_g[l] .= reshape(θ[idx:idx+nW-1],  arch.sizes_var[l+1], arch.sizes_var[l]) 
            idx += nW
    
            ## get biases 
            @views b_g[l] .= θ[idx:idx+arch.sizes_var[l+1]-1]
            idx += arch.sizes_var[l+1]
        end
    
        ## get precision for hyperpriors
        for l in 1:(length(arch.sizes_var)-1)
            log_α_g[l] = θ[idx]
            idx += 1
        end
    else
        ## get precision for likelihood in homogeneous model
        log_β  = θ[idx]
        idx += 1
    end
    

    ## ----------------------------------------------------------------------
    ## Calculate log posterior density
    ## ----------------------------------------------------------------------
    forward!(f, W_f, b_f, X, H_f)
    if arch.heterogeneous == true
        forward!(g, W_g, b_g, X, H_g)
    end
    
    resid .= y .- f
    lpdf = 0.0
    if arch.heterogeneous == true
        for i in eachindex(resid)
            lpdf += 0.5 * (g[i] - exp(g[i]) * resid[i]^2)
        end
    else
        lpdf = 0.5 * (length(resid) * log_β - exp(log_β) * sum(abs2, resid))
    end

    
    for l in 1:(length(arch.sizes_mean)-1)
        ## prior on weights
        nW = arch.sizes_mean[l+1] * arch.sizes_mean[l]
        if l == 1
            lpdf += 0.5 * (nW * log_α_f[l] - exp(log_α_f[l]) * sum(abs2, W_f[l]))
        else
            ## scale variance by the number of nodes in previous layer
            lpdf += 0.5 * (nW * (log_α_f[l] + log(arch.sizes_mean[l])) - (exp(log_α_f[l]) * 
                            arch.sizes_mean[l] * sum(abs2, W_f[l])))
        end
        
        ## prior on biases
        lpdf += 0.5 * (-sum(abs2, b_f[l]) / σ2b)
        ## Prior on precision parameters
        lpdf += log_gamma_hyperprior(log_α_f[l], exp(log_α_f[l]), gamma_shape, gamma_rate)
    end

    if arch.heterogeneous == true
        for l in 1:(length(arch.sizes_var)-1)
            ## prior on weights
            nW = arch.sizes_var[l+1] * arch.sizes_var[l]
            if l == 1
                lpdf += 0.5 * (nW * log_α_g[l] - exp(log_α_g[l]) * sum(abs2, W_g[l]))
            else
                ## scale variance by the number of nodes in previous layer
                lpdf += 0.5 * (nW * (log_α_g[l] + log(arch.sizes_var[l])) - (exp(log_α_g[l]) * 
                                arch.sizes_var[l] * sum(abs2, W_g[l])))
            end
            
            ## prior on biases
            lpdf += 0.5 * (-sum(abs2, b_g[l]) / σ2b)
            ## Prior on precision parameters
            lpdf += log_gamma_hyperprior(log_α_g[l], exp(log_α_g[l]), gamma_shape_g, gamma_rate_g)
        end
    else
        ## Prior on precision parameter for homogeneous model
         lpdf += log_gamma_hyperprior(log_β, exp(log_β), gamma_shape, gamma_rate)
    end
    
    return lpdf
end

# ### Run AGESS

# Now that we have constructed the necessary functions, let's consider a heteroskedastic neural network with the same architecture for the mean function as before ($n_0 = 6,
# n_1 = 8, n_2 = 10, n_{\text{out}} = 1$) and a relatively shallow and narrow neural network to capture the heterogeneity of the error terms ($n_0 = 6, n_1 = 6, n_{\text{out}} = 1$). Since the variance is a second-order statistic, and thus requires more data in order to learn, it is reasonable to restrict the class of functions to less complex functions by having a more shallow and narrow neural network compared to the mean function. 

arch = HBNNArchitecture([6, 8, 10, 1], [6, 6, 1], true) 
P = n_params(arch)

## Specify intermediate values used in likelihood calculation
b_f = Vector{Vector{Float64}}(undef, length(arch.sizes_mean) - 1)
H_f = Vector{Matrix{Float64}}(undef, length(arch.sizes_mean) - 1)
W_f = Vector{Matrix{Float64}}(undef, length(arch.sizes_mean) - 1)
log_α_f = zeros(length(arch.sizes_mean) - 1)
for l in 1:(length(arch.sizes_mean) - 1)
    b_f[l] = zeros(arch.sizes_mean[l+1])
    H_f[l] = zeros(arch.sizes_mean[l], n_obs)
    W_f[l] = zeros(arch.sizes_mean[l+1], arch.sizes_mean[l])
end
b_g = Vector{Vector{Float64}}(undef, length(arch.sizes_var) - 1)
H_g = Vector{Matrix{Float64}}(undef, length(arch.sizes_var) - 1)
W_g = Vector{Matrix{Float64}}(undef, length(arch.sizes_var) - 1)
log_α_g = zeros(length(arch.sizes_var) - 1)
for l in 1:(length(arch.sizes_var) - 1)
    b_g[l] = zeros(arch.sizes_var[l+1])
    H_g[l] = zeros(arch.sizes_var[l], n_obs)
    W_g[l] = zeros(arch.sizes_var[l+1], arch.sizes_var[l])
end
f = similar(y_train)
g = similar(y_train)
resid = similar(y_train)

## Hyperpriors
gamma_shape = 2.0
gamma_rate = 2.0
gamma_shape_g = 2.0
gamma_rate_g = 2.0

println("--- Number of Parameters = ", P, " ---- \n")

# While convergence was relatively fast in the homoskedastic model, this is not the case in the heteroskedastic model. Since we have the same network structure for the mean, we can use a warm-start of the mean function, to help the markov chain converge faster (however, convergence in general is much slower as the posterior geometry seems to be much more challenging).

## ----------------------------------------------------------------------
## Run AGESS
## ----------------------------------------------------------------------
n_MCMC = 500000  # Should run for longer in realistic scenarios

@info "Starting Bayesian NN (Mean and Variance)"
### use warm start using the homogeneous model for W_f and b_f
θ_init = zeros(P)
index_f = let idx = 0
    for l in 1:(length(arch.sizes_mean)-1)
        idx += arch.sizes_mean[l+1] * arch.sizes_mean[l]
        idx += arch.sizes_mean[l+1]
    end
    idx
end
### set to posterior median of homogeneous model
@views θ_init[1:index_f] .= vec(median(Matrix(results.value[10000:end,1:index_f,1]), dims = 1))


## run AGESS
results2 = AGESS(θ -> transformed_log_posterior(θ, X_train, y_train, W_f, W_g, b_f, b_g, 
                                                log_α_f, log_α_g, H_f, H_g, f, g, resid, arch, gamma_shape, 
                                                gamma_rate, gamma_shape_g, gamma_rate_g, 1.0), 
                 n_MCMC, P, init_x = θ_init)

# As we can see, the Markov chain took significantly longer to converge under this more flexible model.

burnin = floor(Int64, 0.5 * n_MCMC)
index = collect(25:10:n_MCMC)
plot(index, results2[:lp][index], label = "log target density")
vline!([burnin], color =:red, label = "burnin")

# ### Posterior Predictive Distributions

# Let's compare the posterior predictive distributions under the heteroskedastic model to the homoskedastic model.

## Function for generating posterior predictive draws
function posterior_predictive_draws(post_samples::AbstractMatrix{Y}, X_grid::AbstractMatrix{Y},
                                    W_f::AbstractVector{<:AbstractMatrix{Y}}, W_g::AbstractVector{<:AbstractMatrix{Y}},
                                    b_f::AbstractVector{<:AbstractVector{Y}}, b_g::AbstractVector{<:AbstractVector{Y}},
                                    log_α_f::AbstractVector{Y}, log_α_g::AbstractVector{Y},
                                    H_f::AbstractVector{<:AbstractMatrix{Y}},
                                    H_g::AbstractVector{<:AbstractMatrix{Y}}, f::AbstractMatrix{Y}, 
                                    g::AbstractMatrix{Y}, arch::HBNNArchitecture) where {Y<: AbstractFloat}
    n_s = size(post_samples,1)
    n_g = size(X_grid, 2)
    draws = zeros(n_s, n_g)        # noisy draws
    f_means = zeros(n_s, n_g)      # noiseless function draws
    σ = zeros(n_g)
    log_β = 0.0
    for s in 1:n_s
        ## ----------------------------------------------------------------------
        ## Unflatten
        ## ----------------------------------------------------------------------
        
        ## index
        idx = 1
    
        for l in 1:(length(arch.sizes_mean)-1)
            ## get weights
            nW = arch.sizes_mean[l+1] * arch.sizes_mean[l]
            @views W_f[l] .= reshape(post_samples[s,idx:idx+nW-1],  arch.sizes_mean[l+1], arch.sizes_mean[l]) 
            idx += nW
    
            ## get biases 
            @views b_f[l] .= post_samples[s,idx:idx+arch.sizes_mean[l+1]-1]
            idx += arch.sizes_mean[l+1]
        end
    
        ## get precision for hyperpriors
        for l in 1:(length(arch.sizes_mean)-1)
            log_α_f[l] = post_samples[s,idx]
            idx += 1
        end
    
        if arch.heterogeneous == true
            for l in 1:(length(arch.sizes_var)-1)
                ## get weights
                nW = arch.sizes_var[l+1] * arch.sizes_var[l]
                @views W_g[l] .= reshape(post_samples[s,idx:idx+nW-1],  arch.sizes_var[l+1], arch.sizes_var[l]) 
                idx += nW
        
                ## get biases 
                @views b_g[l] .= post_samples[s,idx:idx+arch.sizes_var[l+1]-1]
                idx += arch.sizes_var[l+1]
            end
        
            ## get precision for hyperpriors
            for l in 1:(length(arch.sizes_var)-1)
                log_α_g[l] = post_samples[s,idx]
                idx += 1
            end
        else
            ## get precision for likelihood in homogeneous model
            log_β  = post_samples[s,idx]
            idx += 1
        end
    

        ## ----------------------------------------------------------------------
        ## Run forward Model
        ## ----------------------------------------------------------------------
        forward!(f, W_f, b_f, X_grid, H_f)
        if arch.heterogeneous == true
            forward!(g, W_g, b_g, X_grid, H_g)
        end
        
        @views f_means[s, :] .= vec(f)
        if arch.heterogeneous == true
            σ .= vec(sqrt.(1 ./ exp.(g)))
        else
            σ .= sqrt(1 / exp(log_β))
        end
        
        ## Add error term
        @views draws[s, :] .= f_means[s, :] .+ σ .* randn(n_g)
    end
    return draws, f_means
end

froude_idx = 6
## Build a grid over Froude number for visualization, holding other
## features fixed at their mean
froude_grid_std = collect(range(minimum(X[froude_idx, :]),
                         maximum(X[froude_idx, :]), length=100))
n_grid = length(froude_grid_std)
X_grid = zeros(6, n_grid)
@views X_grid[froude_idx, :] .= froude_grid_std


### reset H, f, and g based on number in grid
H_f1 = Vector{Matrix{Float64}}(undef, length(arch.sizes_mean) - 1)
for l in 1:(length(arch.sizes_mean) - 1)
    H_f1[l] = zeros(arch.sizes_mean[l], n_grid)
end
H_g1 = Vector{Matrix{Float64}}(undef, length(arch.sizes_var) - 1)
for l in 1:(length(arch.sizes_var) - 1)
    H_g1[l] = zeros(arch.sizes_var[l], n_grid)
end
f1 = zeros(1, n_grid)
g1 = zeros(1, n_grid)
## Use function to get posterior draws
@views draws, f_means = posterior_predictive_draws(results2.value[burnin:end,:,1], X_grid, W_f,
                                                   W_g, b_f, b_g, log_α_f, log_α_g, H_f1, H_g1,
                                                   f1, g1, arch)

 
pred_mean   = vec(mean(f_means, dims=1))          # E[f(x)] over posterior
pred_lower  = [quantile(f_means[:, j], 0.025) for j in 1:n_grid]
pred_upper  = [quantile(f_means[:, j], 0.975) for j in 1:n_grid]
predictive_lower = [quantile(draws[:, j], 0.025) for j in 1:n_grid]   # includes noise
predictive_upper = [quantile(draws[:, j], 0.975) for j in 1:n_grid]

p1 = plot(froude_grid_std, predictive_upper, fillrange=predictive_lower,
          fillalpha=0.15, linealpha=0, color=:steelblue,
          label="95% predictive interval (incl. noise)")
plot!(p1, froude_grid_std, pred_upper, fillrange=pred_lower,
      fillalpha=0.55, linealpha=0, color=:steelblue,
      label="95% credible interval (f only)")
plot!(p1, froude_grid_std, pred_mean, color=:navy, linewidth=2,
      label="Posterior predictive mean")
scatter!(p1, X_train[froude_idx, :], y_train[1, :],
         color=:black, markersize=4, label="Training data")
scatter!(p1, X_test[froude_idx, :], y_test[1, :],
         color=:red, markersize=5, markershape=:diamond, label="Held-out test data")
xlabel!(p1, "Froude number")
ylabel!(p1, "Residuary resistance")
title!(p1, "Posterior predictive: BNN fit to Yacht Hydrodynamics data")

# Compared to before, we can see that predictive intervals are much narrower when the Froude number is small, and grows with Froude number. However, we can see that even when the Froude number is larger ($ > 1.5$), the predictive interval is narrower compared to the homoskedastic model. Again, one has to remember that this is a view of the conditional predictive distribution, and that all predictor values can influence the width of the predictive intervals. To get a better sense of the calibration, let's again look at the coverage under the test set.

function predictive_interval_at(post_samples::AbstractMatrix{Y}, x_i::AbstractVector{Y},
                                W_f::AbstractVector{<:AbstractMatrix{Y}}, W_g::AbstractVector{<:AbstractMatrix{Y}},
                                b_f::AbstractVector{<:AbstractVector{Y}}, b_g::AbstractVector{<:AbstractVector{Y}},
                                log_α_f::AbstractVector{Y}, log_α_g::AbstractVector{Y},
                                H_f::AbstractVector{<:AbstractMatrix{Y}},
                                H_g::AbstractVector{<:AbstractMatrix{Y}},
                                f_i::AbstractMatrix{Y}, g_i::AbstractMatrix{Y},
                                arch::HBNNArchitecture;
                                q::Vector{Float64} = [0.025, 0.975],
                                n_draw::Int = 10000) where {Y<:AbstractFloat}
    n_s = size(post_samples, 1)
    draws_i = Vector{Float64}(undef, n_draw)
    x_mat = reshape(x_i, :, 1)   # reshape to n_in x 1 matrix for forward!
    σ_i = 0.0
    for k in 1:n_draw
        s = rand(1:n_s)
        ## ----------------------------------------------------------------------
        ## Unflatten
        ## ----------------------------------------------------------------------
        
        ## index
        idx = 1
    
        for l in 1:(length(arch.sizes_mean)-1)
            ## get weights
            nW = arch.sizes_mean[l+1] * arch.sizes_mean[l]
            @views W_f[l] .= reshape(post_samples[s,idx:idx+nW-1],  arch.sizes_mean[l+1], arch.sizes_mean[l]) 
            idx += nW
    
            ## get biases 
            @views b_f[l] .= post_samples[s,idx:idx+arch.sizes_mean[l+1]-1]
            idx += arch.sizes_mean[l+1]
        end
    
        ## get precision for hyperpriors
        for l in 1:(length(arch.sizes_mean)-1)
            log_α_f[l] = post_samples[s,idx]
            idx += 1
        end
    
        if arch.heterogeneous == true
            for l in 1:(length(arch.sizes_var)-1)
                ## get weights
                nW = arch.sizes_var[l+1] * arch.sizes_var[l]
                @views W_g[l] .= reshape(post_samples[s,idx:idx+nW-1],  arch.sizes_var[l+1], arch.sizes_var[l]) 
                idx += nW
        
                ## get biases 
                @views b_g[l] .= post_samples[s,idx:idx+arch.sizes_var[l+1]-1]
                idx += arch.sizes_var[l+1]
            end
        
            ## get precision for hyperpriors
            for l in 1:(length(arch.sizes_var)-1)
                log_α_g[l] = post_samples[s,idx]
                idx += 1
            end
        else
            ## get precision for likelihood in homogeneous model
            log_β  = post_samples[s,idx]
            idx += 1
        end
    

        ## ----------------------------------------------------------------------
        ## Run forward Model
        ## ----------------------------------------------------------------------
        forward!(f_i, W_f, b_f, x_mat, H_f)
        if arch.heterogeneous == true
            forward!(g_i, W_g, b_g, x_mat, H_g)
        end
        if arch.heterogeneous == true
            σ_i = sqrt(1 / exp(g_i[1,1]))
        else
            σ_i = sqrt(1 / exp(log_β))
        end
        draws_i[k] = f_i[1] + σ_i * randn()
    end
    return quantile(draws_i, q[1]), quantile(draws_i, q[2])
end

### rest H and Y_ph based on number in grid
H_f_i = Vector{Matrix{Float64}}(undef, length(arch.sizes_mean) - 1)
for l in 1:(length(arch.sizes_mean) - 1)
    H_f_i[l] = zeros(arch.sizes_mean[l], 1)
end

H_g_i = Vector{Matrix{Float64}}(undef, length(arch.sizes_var) - 1)
for l in 1:(length(arch.sizes_var) - 1)
    H_g_i[l] = zeros(arch.sizes_var[l], 1)
end

Y_ph_i = zeros(1, 1)

lo_vec = zeros(n_test)
hi_vec = zeros(n_test)
covered = falses(n_test)

for i in 1:n_test
    lo, hi = predictive_interval_at(results2.value[burnin:end, :, 1], X_test[:, i], W_f, W_g,
                                     b_f, b_g, log_α_f, log_α_g, H_f_i, H_g_i, f1, g1, arch)
    lo_vec[i] = lo
    hi_vec[i] = hi
    covered[i] = (lo <= y_test[1, i] <= hi)
end

println("Empirical coverage (target 0.95): ", mean(covered))

## Sort by Froude number for a readable x-axis
order = sortperm(X_test[froude_idx, :])

p_coverage = plot(legend=:topleft, xlabel="Froude number",
                   ylabel="Residuary resistance (standardized)",
                   title="Posterior predictive coverage on held-out data")

## Error bars: green if covered, red if missed
for (rank, i) in enumerate(order)
    color = covered[i] ? :green : :red
    plot!(p_coverage, [X_test[froude_idx, i], X_test[froude_idx, i]],
          [lo_vec[i], hi_vec[i]], color=color, linewidth=2, label=false, alpha=0.7)
end

scatter!(p_coverage, X_test[froude_idx, :], y_test[1, :],
         group=ifelse.(covered, "Covered", "Missed"),
         color=ifelse.(covered, :green, :red),
         markersize=4, markerstrokewidth=0.5)

##display(p_coverage)

nominal_levels = 0.5:0.05:0.99
empirical_coverage = zeros(length(nominal_levels))

for (li, level) in enumerate(nominal_levels)
    alpha = 1 - level
    q_lo, q_hi = alpha/2, 1 - alpha/2
    covered_at_level = falses(n_test)
    for i in 1:n_test
        lo, hi = predictive_interval_at(results2.value[burnin:end, :, 1], X_test[:, i], W_f, W_g,
                                        b_f, b_g, log_α_f, log_α_g, H_f_i, H_g_i, f1, g1, arch;
                                        q=[q_lo, q_hi])
        covered_at_level[i] = (lo <= y_test[1, i] <= hi)
    end
    empirical_coverage[li] = mean(covered_at_level)
    println("Nominal $(round(level,digits=2)) -> empirical $(round(empirical_coverage[li],digits=3))")
end

p_calib = plot(nominal_levels, nominal_levels, linestyle=:dash, color=:black,
               label="Perfect calibration", xlabel="Nominal coverage",
               ylabel="Empirical coverage", title="Calibration curve",
               legend=:topleft, aspect_ratio=:equal, xlims=(0.45,1.0), ylims=(0.45,1.0))
plot!(p_calib, nominal_levels, empirical_coverage, marker=:circle, color=:navy,
      label="BNN predictive intervals", linewidth=2)

## ---- Combined figure -------------------------------------------------
p_combined = plot(p_coverage, p_calib, layout=(1,2), size=(1100,450),
                   left_margin=5Plots.mm, bottom_margin=5Plots.mm)
p_combined
@info "Ending Bayesian NN (Mean and Var)"

# We can see that the predictive distributions are better calibrated under the heteroskedastic model. While there is a bit of undercoverage, one has to remember that this is coverage on a test dataset, and some undercoverage is perhaps expected.

# ## Summary

# Here we explored the capability of AGESS to fit Bayesian neural networks on a real dataset. We must emphasize this is not meant to be a tutorial on Bayesian neural networks (there are likely much better sources available), but is instead meant to demonstrate the performance of AGESS. Indeed, we illustrated efficient sampling performance when targeting the homoskedastic model, with relatively fast convergence of the Markov chain. However, the introduction of a heteroskedastic error term, led to significantly longer burn-in requirements, and a relatively slow-mixing Markov chain. It appears that the additional flexibility of the heteroskedastic model leads to complex posterior geometry, such as funnel or multimodal posteriors, which in general are challenging for many sampling algorithms. Indeed, funnels often lead to divergent transitions when using gradient-based methods (the common type of MCMC algorithms used in these settings). While additional studies are needed to compare the advantages and disadvantages of AGESS to gradient-based methods, this tutorial provides initial evidence that AGESS can be utilized in these settings. 


# [^1]: R. Neal. Bayesian learning for neural networks (Vol. 118). Springer Science & Business Media, 2012.

# [^2]: N. Marco and S. T. Tokdar. Adaptive generalized elliptical slice sampling. arXiv preprint arXiv:2605.21659, 2026.