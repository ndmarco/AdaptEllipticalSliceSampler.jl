# Deep Gaussian Process Surrogate Modeling

In many scientific domains, the use of complex computer simulations have become more prominent,
especially in fields where data collection is challenging[^1]. However, these high-fidelity computer
simulations are often expensive to evaluate, restricting scientists to evaluate the complex
computer simulation at a limited number of points in the (potentially high-dimensional) input space.
The use of Gaussian Processes as surrogate models allows for predictive inference across the 
input space given a limited set of evaluations; enabling scientists to use the model to inform
the next evaluation of the complex computer simulation. Recently, the use of deep Gaussian processes[^2]
has become a popular choice of surrogate models[^3] [^4] [^5] [^6].

### Model Specification

Deep GPs provide a flexible non-stationary model using a hierarchical representation of multiple
stationary Gaussian Processes. Here, we will focus on a two-layer deep GP. Let $\mathbf{Y} \in \mathbb{R}^N$ 
be the outputs of interest, and let $\mathbf{X} \in \mathbb{R}^{N \times D}$ be the inputs.Our goal is to estimate
the function $f: \mathbb{R}^D \rightarrow \mathbb{R}$ where $Y_i = f(\mathbf{x}_i)$. Letting $W(\mathbf{x})$ 
$(\mathbf{x} \in \mathbb{R}^D)$ be the augmented Gaussian process, with  
$\mathbf{W} := (W(\mathbf{x}_1), \dots, W(\mathbf{x}_N)) \in \mathbb{R}^N$, we can specify the 
deep GP surrogate model using the following hierarchical representation[^3]:

$$\mathbf{Y} \sim \mathcal{N}\left(\mathbf{0}, \tau\left(K_{\theta_y}(\mathbf{X}) + g_y\mathbf{I}_N \right)\right),$$

$$\mathbf{W}  \sim \mathcal{N}\left(\mathbf{0}, K_{\theta_w}(\mathbf{X}) + g_w \mathbf{I}_N\right),$$

$$\tau \sim \text{Inv-Gamma}(\nu/2, \nu/2),$$

where $K_{\theta_y}(\mathbf{x}_i, \mathbf{x}_j):= \exp\left( - \left[\sum_{d = 1}^D \frac{\left\lVert x_{id} - x_{jd}\right\rVert ^2}{\theta_{y_d}} +  \frac{\left\lVert W_{i} - W_{j}\right\rVert ^2}{\theta_{y_{D+1}}}\right]\right)$, $K_{\theta_y}(\mathbf{x}_i, \mathbf{x}_j):= \exp\left( - \sum_{d = 1}^D \frac{\left\lVert x_{id} - x_{jd}\right\rVert ^2}{\theta_{w_d}}\right)$, and $g_y, g_w \in \mathbb{R}_{+}$.
In this tutorial, we will consider the case where the following parameters are fixed (as is often done
when the outputs of the complex computer simulations are deterministic): $g_w = 10^{-8}$, $g_y  = 10^{-8}$,
 and $\nu = 6$.

### Simulating the Data

We will consider the following one-dimensional function[^3]:

$$f(x) = \sin(x) + 2\exp(-30x^2),$$

observed at $N = 15$ uniformly evaluated points on $\Omega = [-2.5,2.5]$. We can generate the data
using the following code.

```@example DeepGP
import Random
using AdaptEllipticalSliceSampler
using Distributions
using Plots
using LinearAlgebra

Random.seed!(123)

function gen_data(N::T, min_eval::Y, max_eval::Y) where {Y<:AbstractFloat, T<:Integer}
    X = collect(LinRange(min_eval, max_eval, N))
    Y_N = sin.(X) .+ 2 * exp.(-30 * X.^2)
    return X, Y_N
end

### Generate Data
N_obs = 15
X_N, Y_N = gen_data(N_obs, -2.5, 2.5)

### Generate the truth
time_points, Y_truth = gen_data(1000, -2.5, 2.5)

### Plot observed data and true function
### observed data
p = plot(X_N, Y_N, seriestype=:scatter, color = "red", label = "Observed Data")
### True function
plot!(p, time_points, Y_truth, color = "blue", label ="Truth")
```

### Specifying the Log Posterior Density

Using `AdaptEllitpicalSliceSampler.jl`, we essentially only need to specify a function that efficiently
evaluates the log posterior density. Notice that we can integrate out the parameter $\tau$, leading to $\mathbf{Y}$
being multivariate T distributed. We will start by constructing functions that generate $K_{\theta_w}(\mathbf{X})$
and $K_{\theta_y}(\mathbf{X})$.

```@example DeepGP
### Covariance matrix for W
function construct_Kernel_Mat!(Σ::AbstractMatrix{Y}, X::AbstractVector{Y}, 
                               θ::Y) where {Y<:AbstractFloat}
    for j in 1:size(Σ)[1],k in 1:size(Σ)[2]
        Σ[j,k] = exp(-(((X[j] - X[k])^2) / θ))
    end
    Σ .= Symmetric(Σ)
end

### Covariance matrix for Y
function construct_Kernel_Mat_y!(Σ::AbstractMatrix{Y}, X::AbstractVector{Y},
                                 W::AbstractVector{Y}, θ_y_x::Y, 
                                 θ_y_w::Y) where {Y<:AbstractFloat}
    for j in 1:size(Σ)[1],k in 1:size(Σ)[2]
        Σ[j,k] = -(((X[j] - X[k])^2) / θ_y_x)
        Σ[j,k] -= (((W[j] - W[k])^2) / θ_y_w)
    end
    Σ .= exp.(Σ)
    Σ .= Symmetric(Σ)
end
```
We can see that both of these functions take $\Sigma$ as an input, ensuring that we are not needlessly 
allocating memory. Using these functions, we can now specify the pdf of $\mathbf{W}$ and $\mathbf{Y}$.

```@example DeepGP
### Y is multivariate T distributed
function likelihood_Y(W::AbstractVector{Y}, X::AbstractVector{Y}, Y_N::AbstractVector{Y}, 
                      g::Y, θ_y_x::Y, θ_y_w::Y, ph::AbstractVector{Y}, 
                      Σ::AbstractMatrix{Y}, ν_y::Y)::Float64 where {Y<:AbstractFloat}
    construct_Kernel_Mat_y!(Σ, X, W, θ_y_x, θ_y_w)
    Σ[diagind(Σ)] .+=  g
    cholesky!(Σ)
    ph .= UpperTriangular(Σ)' \ Y_N
    lpdf =  -sum(log.(diag(Σ)))  - (0.5 * (ν_y + length(ph)) * log1p(dot(ph, ph) / ν_y))

    return lpdf
end

### W is multivariate Gaussian distributed
function likelihood_W(W::AbstractVector{Y}, X_N::AbstractVector{Y}, θ_w::Y, g::Y, 
                      ph::AbstractVector{Y}, 
                      Σ::AbstractMatrix{Y})::Float64 where {Y<:AbstractFloat}
    construct_Kernel_Mat!(Σ, X_N, θ_w)
    Σ[diagind(Σ)] .+= g
    cholesky!(Σ)

    ph .= UpperTriangular(Σ)' \ W
    lpdf = -sum(log.(diag(Σ)))  - 0.5 * dot(ph, ph)

    return lpdf
end
```

Notice that in the evaluation of the likelihoods, we have "place-holding" variables `ph` which are used for intermediate
calculations, allowing us to reallocate that memory for efficient computation. We will concatenate 
all the parameters into one big vector of parameters, 
$\mathbf{p} = [\mathbf{W}, \theta_{y_1}, \theta_{y_{2}}, \theta_{w_1}]$. We will transform the scale 
parameters ($\theta_{y_1}, \theta_{y_{2}}, \theta_{w_1}$) using a log transformation to remove
the positivity constraints (don't forget the Jacobian term when calculating the log pdfs).

```@example DeepGP
function log_posterior(p::AbstractVector{Y}, X_N::AbstractVector{Y}, Y_N::AbstractVector{Y},
                       Σ1::AbstractMatrix{Y}, Σ2::AbstractMatrix{Y}, ph1::AbstractVector{Y},
                       ph2::AbstractVector{Y}, g_w::Y, g_y::Y, ν::Y, 
                       N_obs::T) where {Y<:AbstractFloat, T<:Integer}
    lpdf = likelihood_Y(p[1:N_obs], X_N, Y_N, g_y, exp(p[N_obs+1]), 
                        exp(p[N_obs+2]), ph1, Σ1, ν)
    lpdf += likelihood_W(p[1:N_obs], X_N, exp(p[N_obs+3]), g_w, ph2, Σ2)
    ### prior on transformed theta_y_1
    lpdf += logpdf(Gamma(1.0,2.0), exp(p[N_obs+1])) + p[N_obs+1]
    ### prior on transformed theta_y_2
    lpdf += logpdf(Gamma(1.0,2.0), exp(p[N_obs+2])) + p[N_obs+2]
    ### prior on transformed theta_w_1
    lpdf += logpdf(Gamma(1.0,2.0), exp(p[N_obs+3])) + p[N_obs+3]

    return lpdf
end
```

### Running AGESS

Now that we have specified a function to efficiently evaluate the log posterior density, we can 
simply specify all of our pre-allocated variables and use the `AGESS` function. We will use
the standard prior mean $\boldsymbol{\mu}_0$ centered at $\mathbf{0}$, and prior variance 
$\boldsymbol{\Sigma}_0 = \mathbf{I}$. When calling the function `AGESS`, we will specify an 
anonymous function with $\mathbf{p}$ as the only input to the function log_posterior, using
the global parameter values for all the other variables. 

```@example DeepGP
### Specify pre-allocated variables
ph1 = ones(N_obs)
Σ1 = diagm(ones(N_obs))
ph2 = ones(N_obs)
Σ2 = diagm(ones(N_obs))

### Specify constants
ν = 6.0
g_y = 1e-8 
g_w = 1e-8

### MCMC parameters
P = N_obs + 3
n_MCMC = 50000


### Run Agess
results = AGESS(p -> log_posterior(p, X_N, Y_N, Σ1, Σ2, ph1, ph2, g_w, g_y, ν, N_obs), 
                n_MCMC, P)
```

We can plot the log pdf of the posterior at every iteration of the Markov chain to potentially
detect convergence issues.

```@example DeepGP
plot(results.l_pdf[1:n_MCMC], legend = false)
```

**Note: The Markov chain should likely be run longer, but due to limited computational resources in compiling the documentation, we are not able to run it sufficiently long.**

### Posterior Predictive Distribution

Now that we have fit our model, we often wish to study the posterior predictive distribution; 
allowing us to identify sections of the input space with high uncertainty and infer values
at inputs not observed. Suppose that we want to evaluate the posterior predictive distribution at
a set of new of inputs $\mathbf{X^*}$. We can do this by first sampling $\mathbf{W}^*$ and then 
sampling $\mathbf{Y}^*$. We know that $W(\mathbf{x})$ is a Gaussian process, thus we have that

$$\boldsymbol{\mu}_{\mathbf{W}^*}^{(i)} = K_{\theta_w^{(i)}}(\mathbf{X}^*, \mathbf{X})\left[K_{\theta_w^{(i)}}(\mathbf{X}) + \mathbf{I}g_w\right]^{-1} \mathbf{W}^{(i)},$$

$$\boldsymbol{\Sigma}_{\mathbf{W}^*}^{(i)} = K_{\theta_w^{(i)}}(\mathbf{X}^*) + \mathbf{I}g_w - K_{\theta_w^{(i)}}(\mathbf{X}^*, \mathbf{X})\left[K_{\theta_w^{(i)}}(\mathbf{X}) + \mathbf{I}g_w\right]^{-1}K_{\theta_w^{(i)}}(\mathbf{X}, \mathbf{X}^*),$$

where the notation $x^{(i)}$ denotes the $i^{th}$ sample of $x$ from our Markov chain. Thus we 
can draw samples of $(\mathbf{W}^*)^{(i)}$ from $\mathcal{N}(\boldsymbol{\mu}_{\mathbf{W}^*}^{(i)}, \boldsymbol{\Sigma}_{\mathbf{W}^*}^{(i)})$. Similarly, we can sample $(\mathbf{Y}^*)^{(i)}$ conditionally on our sample
of $(\mathbf{W}^*)^{(i)}$ (i.e., using conditional multivariate T distributions). Below is a function
that implements sampling from the posterior predictive distribution using the Markov chain constructed
through the `AGESS` function. 


```@example DeepGP
### Function for generative samples from the posterior predictive distribution
function predictive_draws(time_points::AbstractVector{Y}, W::AbstractMatrix{Y}, 
                          θ_w::AbstractVector{Y}, θ_y_x::AbstractVector{Y}, 
                          θ_y_w::AbstractVector{Y}, g::Y, g_x::Y, Y_N::AbstractVector{Y}, 
                          X_N::AbstractVector{Y}, ν_y::Y;  burnin::Y=0.5, 
                          thinning::T = 10) where {Y<:AbstractFloat, T<:Integer}
    n_MCMC = size(W)[1]
    burnin_num = floor(Int64, (burnin * n_MCMC))
    P_out = length(time_points)
    P = length(X_N)
    steps = collect(range(burnin_num + 1, n_MCMC, step = thinning))
    Y_out = zeros(length(steps), P_out)
    W_out = zeros(P_out)
    k_out_X_const = zeros(P_out, P)
    k_out_X = zeros(P_out, P)
    k_out_Y = zeros(P_out, P)
    Σ = zeros(P,P)
    ph = zeros(P_out, P)
    ph1 = zeros(P)

    μ_out = zeros(P_out)
    Σ_out = zeros(P_out, P_out) 

    for j in 1:P_out
        for k in 1:P
            k_out_X_const[j,k] = -(time_points[j] - X_N[k])^2
        end
    end

    iter = 1
    for i in steps
        ## get distribution of W_out
        k_out_X .= exp.(k_out_X_const ./ θ_w[i])
        construct_Kernel_Mat!(Σ, X_N, θ_w[i])
        Σ[diagind(Σ)] .+=  g_x
        ph .= (Σ \ k_out_X')'
        @views μ_out .= ph * W[i,:]
        construct_Kernel_Mat!(Σ_out, time_points, θ_w[i])
        Σ_out .-= ph * k_out_X'
        Σ_out[diagind(Σ_out)] .+= g_x
        Σ_out .= Hermitian(Σ_out)
        cholesky!(Σ_out)

        ## Generate sample of W_out
        W_out .= UpperTriangular(Σ_out)' * randn(P_out) 
        W_out .+= μ_out

        ## Generate sample of Y_out
        for j in 1:P_out
            for k in 1:P
                k_out_Y[j,k] = exp(-(((W_out[j] - W[i,k])^2 / θ_y_w[i]) + 
                                ((time_points[j] - X_N[k])^2 / θ_y_x[i])))
            end
        end


        @views construct_Kernel_Mat_y!(Σ, X_N, W[i,:], θ_y_x[i], θ_y_w[i])
        Σ[diagind(Σ)] .+=  g
        ph .= (Σ \ k_out_Y')'
        μ_out .= ph * Y_N

        cholesky!(Σ)
        ph1 .= UpperTriangular(Σ)' \ Y_N
        construct_Kernel_Mat_y!(Σ_out, time_points, W_out, θ_y_x[i], θ_y_w[i])
        Σ_out[diagind(Σ_out)] .+= g
        Σ_out .-= ph * k_out_Y'
        Σ_out .= Hermitian(Σ_out)

        d = dot(ph1, ph1)
        Σ_out .*= (ν_y + d) / (ν_y + P)
        cholesky!(Σ_out)

        
        Y_out[iter,:] .= UpperTriangular(Σ_out)' * randn(P_out)
        Y_out[iter,:] .*= 1 / sqrt(rand(Gamma((ν_y + P) / 2, 2 / (ν_y + P))))
        Y_out[iter,:] .+= μ_out
        iter = iter + 1
    end

    return Y_out
end

### Time points of interest
time_points = collect(collect(LinRange(-2.5, 2.5, 500)))
time_points = setdiff(time_points, X_N)

### Run function (remember to transform your log-transformed variables back)
Y_pred = predictive_draws(time_points, results.samps[:,1:N_obs], 
                          exp.(results.samps[:,N_obs + 3]), 
                          exp.(results.samps[:,N_obs + 1]), 
                          exp.(results.samps[:,N_obs + 2]), g_y, 
                          g_w, Y_N, X_N, ν, burnin = results.params.burnin)


### Plot results
p = plot(X_N, Y_N, seriestype=:scatter, color = "black", label = "Observed Data")
P_out = length(time_points)

### Get Credible intervals
Upper_CI = zeros(P_out)
Lower_CI = zeros(P_out)
median_est = zeros(P_out)
for i in 1:P_out
    median_est[i] = median(Y_pred[:,i])
    Lower_CI[i] = quantile(Y_pred[:,i], 0.025)
    Upper_CI[i] = quantile(Y_pred[:,i], 0.975)
end
p = plot!(p, time_points, median_est, color = "blue", label ="Posterior Median")
truth = sin.(time_points) .+ 2 * exp.(-30 * time_points.^2)
p = plot!(p, time_points, truth, color = "red", label ="Truth")
plot!(p, time_points, Lower_CI, fillrange = Upper_CI, fillalpha = 0.3, alpha = 0.3, 
      label = "95% CI")
```

### Conclusion

Deep Gp models are highly flexible models, that can lead to multimodal posterior distributions.
Adaptive generalized elliptical slice sampling[^7] offers an efficient and reliable sampling method
for conducting inference in these settings. For a detailed comparison between AGESS and other
sampling schemes, please refer to the main manuscript[^7].

[^1]: R. B. Gramacy. Surrogates: Gaussian process modeling, design, and optimization for the applied sciences. Chapman and Hall/CRC, 2020.
[^2]: A. Damianou and N. D. Lawrence. Deep gaussian processes. In Artificial intelligence and statistics, pages 207–215. PMLR, 2013.
[^3]: S. Montagna and S. T. Tokdar. Computer emulation with nonstationary gaussian processes. SIAM/ASA Journal on Uncertainty Quantification, 4(1):26–47, 2016.
[^4]: M. I. Radaideh and T. Kozlowski. Surrogate modeling of advanced computer simulations using deep gaussian processes. Reliability Engineering & System Safety, 195:106731, 2020.
[^5]: A. Sauer, A. Cooper, and R. B. Gramacy. Non-stationary gaussian process surrogates. arXiv preprint arXiv:2305.19242, 2023.
[^6]: A. Sauer, R. B. Gramacy, and D. Higdon. Active learning for deep gaussian process surrogates. Technometrics, 65(1):4–18, 2023.
[^7]: N. Marco and S. T. Tokdar. Adaptive Generalized Elliptical Slice Sampling