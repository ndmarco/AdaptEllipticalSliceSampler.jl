# Factor Models

In this tutorial, we will cover a more advanced use-case of `AdaptEllipticalSliceSampler.JL`, 
where we use AGESS[^1] to sample a subset of parameters, and use Gibbs sampling to 
conditionally update the rest of the samplers. In this tutorial, we consider the following
factor model using the multiplicative gamma process shrinkage prior[^2]:

$$\mathbf{y}_i = \Lambda \eta_i + \epsilon_i,$$

$$\epsilon_i \sim \mathcal{N}(0, \mathbf{D}),$$

$$\Lambda_{jh} \mid \phi_{jh}, \tau_h \sim \mathcal{N}(0, \phi_{jh}^{-1} \tau_h^{-1}),$$

$$\phi_{jh} \sim \text{Gamma}(\nu / 2, \nu / 2),$$

$$\tau_h = \prod_{k=1}^h \delta_k \;\;\; (h = 1, \dots, K)$$

$$\delta_1 \sim \text{Gamma}(a_1, 1),$$

$$\delta_{h} \sim \text{Gamma}(a_2, 1) \;\;\; (h \ge 2),$$

$$\mathbf{D}_{jj} \sim \text{Inv-Gamma}(a, b),$$

where $\mathbf{y}_i \in \mathbb{R}^P$, $\Lambda \in \mathbb{R}^{P \times K}$, $\eta_i \in \mathbb{R}^{K}$,
$\mathbf{D}$ is a Diagonal matrix, and $a_2 > 1$ (to promote shrinkage). In these settings, it is 
often the case that $K$ is much smaller than $P$. Under this setup, a Gibbs sampler is available[^2],
making sampling relatively straightforward. Consider the case where we want to use AGESS to sample
$\Lambda$, $\eta_i$, and $\mathbf{D}$, while using Gibbs updates for $\delta_h$ and
$\phi_{jh}$. We will start by constructing Gibbs samplers for $\delta_h$ and $\phi_{jh}$.

### Gibbs Updates

The conditional posterior distribution of $\delta_h$ and $\phi_{jh}$ are as follows[^2]:

$$\phi_{jh}\mid \Theta_{-\phi_{jh}} \sim \text{Gamma}\left(0.5(\nu + 1), 0.5(\nu + \tau_h \Lambda_{jh}^2)\right)$$

$$\delta_1 \mid \Theta_{-\delta_1} \sim \text{Gamma}\left(a_1 + 0.5(P\times K), 1 + 0.5\left(\sum_{l=1}^K \tau_{l}^{(-1)} \sum_{j=1}^P \phi_{jl}\Lambda_{jl}^2\right)\right)$$

$$\delta_h \mid \Theta_{-\delta_h} \sim \text{Gamma}\left(a_2 + 0.5(P\times (K - h +1)), 1 + 0.5\left(\sum_{l=h}^K \tau_{l}^{(-h)} \sum_{j=1}^P \phi_{jl}\Lambda_{jl}^2\right)\right)\;\;\; (2 \le h \le K)$$

We can construct Julia functions to update these parameters as follows.

```@example Factor
import Random, BenchmarkTools
using AdaptEllipticalSliceSampler
using Distributions
using Plots
using LinearAlgebra

Random.seed!(123)

function delta_sampler!(δ::AbstractVector{Y}, ϕ::AbstractMatrix{Y}, Λ::AbstractMatrix{Y}, 
                        a1_δ::Y, a2_δ::Y, ph_δ::AbstractVector{Y}) where {Y<:AbstractFloat}
    K = length(δ)
    P = size(Λ)[2]
    ## Sample δ1
    α = a1_δ + 0.5 * K * P
    @views ph_δ .= ϕ[1,:] .* Λ[1,:]
    @views β = 1.0 + 0.5 * dot(ph_δ, Λ[1,:])
    for i in 2:K
        @views τ = prod(δ[2:i])
        @views ph_δ .= ϕ[i,:] .* Λ[i,:]
        @views β += 0.5 * τ * dot(ph_δ, Λ[i,:])
    end
    δ[1] = rand(Gamma(α, 1 / β))

    ## Sample rest of δ
    for i in 2:K
        α = a2_δ + 0.5 * (K - i + 1) * P
        β = 1.0 
       @views τ = prod(δ[1:(i-1)])
        for j in i:K
            if i != j
                τ *= δ[j]
            end
            @views ph_δ .= ϕ[j,:] .* Λ[j,:]
            @views β += 0.5 * τ * dot(ph_δ, Λ[j,:])
        end
        δ[i] = rand(Gamma(α, 1 / β))
    end
 
    return nothing
end

function phi_sampler!(ϕ::AbstractMatrix{Y}, δ::AbstractVector{Y}, Λ::AbstractMatrix{Y}, 
                       ν::Y) where {Y<:AbstractFloat}
    K = length(δ)
    P =  size(Λ)[2]
    for i in 1:K
        @views τ = prod(δ[1:i])
        for j in 1:P
            α = (ν + 1) * 0.5
            β = 0.5 * (ν  + τ * Λ[i,j]^2)
            ϕ[i,j] = rand(Gamma(α, 1 / β))
        end
    end

    return nothing
end
```

### AGESS Updates

Now that we have our functions to update the $\phi$ and $\tau$ parameters via Gibbs updates, we 
can write a function that evaluates the conditional posterior of the rest of the parameters so
that we can update the state of the Markov chain for these parameters via AGESS[^1]. Here we will
construct two functions. The first one evaluates the likelihood, and the second one reshapes and
makes necessary transformations to our variables of interest.

```@example Factor
function posterior(Λ::AbstractMatrix{Y}, η::AbstractMatrix{Y}, Y_obs::AbstractMatrix{Y}, 
                   D::AbstractMatrix{Y}, ϕ::AbstractMatrix{Y}, δ::AbstractVector{Y}, 
                   τ_ph::AbstractVector{Y}, a::Y, b::Y,
                   ph::AbstractVector{Y}, ph1::AbstractVector{Y})::Float64 where {Y<:AbstractFloat}
    
    lpdf::Float64 = 0.0
    ## Likelihood
    for i in 1:size(Y_obs)[1]
        @views mul!(ph, Λ', η[i,:])
        @views ph .-= Y_obs[i,:]
        @views ph1 .= ph ./ D[diagind(D)]
        @views lpdf += (-0.5* sum(log.(D[diagind(D)])) - 0.5 * dot(ph1, ph))
    end

    ig_d = InverseGamma(a,b)
    for i in 1:size(D)[1]
       lpdf += logpdf(ig_d, D[i,i])
    end

    τ_ph .= δ
    for i in 2:length(δ)
        τ_ph[i] *= τ_ph[i-1]
    end

    for i in 1:size(Λ)[1], j in 1:size(Λ)[2]
        lpdf += 0.5 * log(ϕ[i,j] * τ_ph[i]) - 0.5 * (Λ[i,j]^2 * ϕ[i,j] * τ_ph[i])
    end

    for i in 1:size(η)[1]
        @views lpdf += - 0.5 * dot(η[i,:], η[i,:])
    end

    return lpdf
end

function transform_posterior(x::AbstractVector{Y}, Y_obs::AbstractMatrix{Y}, 
                             ϕ::AbstractMatrix{Y}, δ::AbstractVector{Y}, 
                             τ_ph::AbstractVector{Y}, a::Y, b::Y,
                             N::T, P::T, K::T, D_ph::AbstractMatrix{Y}, 
                             ph::AbstractVector{Y}, 
                             ph1::AbstractVector{Y})::Float64 where {Y<:AbstractFloat, T<:Integer}
    @views Λ_ph = reshape(x[1:K*P], (K, P))
    @views η_ph = reshape(x[(K*P + 1):(N*K + K*P)], (N, K))
    @views D_ph[diagind(D_ph)] .= exp.(x[(N*K + K*P + 1):(N*K + K*P + P)])
    lpdf = posterior(Λ_ph, η_ph, Y_obs, D_ph, ϕ, δ, τ_ph, a, b, ph, ph1)
    ### Jacobian for transformation
    @views lpdf += sum(x[(N*K + K*P + 1):(N*K + K*P + P)])

    return lpdf
end
```


### Construct Custom MCMC Sampling Scheme

Now that we have set up the functions necessary to use the `AdaptEllipticalSliceSampler.jl` package,
and have constructed the functions necessary to perform Gibbs updates, we can use the following
two functions to create a custom MCMC sampling method: `AGESS_single_step_1d!` and `AGESS_single_step!`.
Here we will use a t-distribution with 6 degrees of freedom as our marginal distribution of the
auxiliary variable in AGESS.

```@example Factor

function custom_MCMC(Y_obs::AbstractMatrix{Y}, K::T, n_MCMC::T, a_1::Y, a_2::Y, a::Y, b::Y, 
                     ϵ::Y, single_step_prop::Y, burnin::Y, ν::Y, β::Y) where {Y<:AbstractFloat, T<:Integer}
    N = size(Y_obs)[1]
    P = size(Y_obs)[2]
    n_params = (N * K) + (K * P) + P

    ### Set up variables
    x = zeros(n_MCMC, n_params)
    ϕ = ones(n_MCMC, K, P)
    δ = ones(n_MCMC, K)


    ## Allocate variables for AGESS
    ph_AGESS = zeros(n_params)
    z = similar(ph_AGESS)
    w_const = max(2/3, ((cbrt(n_params) - 1) / cbrt(n_params)))
    N_J = 2
    n_j = 2
    ph_cholesky_update = ones(n_params)


    ### Allocate variables for intermediate calculations
    τ_ph = zeros(K)
    D_ph = diagm(ones(P))
    ph = zeros(P)
    ph1 = similar(ph)
    ph_δ = similar(ph)


    ### Setup adaptive parameters
    μ_adapt = zeros(n_params)
    μ_adapt_ph = zeros(n_params)
    μ_0 = zeros(n_params)

    Σ_adapt = diagm(ones(n_params))
    Σ_chol_0 = cholesky(Σ_adapt)
    Σ_chol_adapt = deepcopy(Σ_chol_0)
    Σ_chol_adapt_ph = deepcopy(Σ_chol_0)

    ### variable for storing log posterior pdf
    lpdf = zeros(n_MCMC)
    @views lpdf[1] = transform_posterior(x[1,:], Y_obs, ϕ[1,:,:], δ[1,:], τ_ph, a, 
                                         b, N, P, K, D_ph, ph, ph1)
    perm = randperm(n_params)

    ### Start MCMC
    for i in 2:n_MCMC
        ### AGESS update
        if i < (n_MCMC * burnin)
            ### 1-d AGESS updates for fast burn-in
            @views lpdf[i] = AGESS_single_step_1d!(x, y -> transform_posterior(y, Y_obs, 
                                                   ϕ[i,:,:], δ[i,:], τ_ph, a, 
                                                   b, N, P, K, D_ph, ph, ph1), true, 6.0, 
                                                   n_params, μ_adapt, 
                                                   Σ_chol_adapt.L, lpdf[i-1], perm, i)
        else
            if rand() > (ϵ + single_step_prop)
                ### standard AGESS step
                @views lpdf[i] = AGESS_single_step!(x, z, y -> transform_posterior(y, Y_obs, 
                                                    ϕ[i,:,:], δ[i,:], τ_ph, a, 
                                                    b, N, P, K, D_ph, ph, ph1), true, 6.0, 
                                                    n_params, ph_AGESS, μ_adapt, 
                                                    Σ_chol_adapt.L, lpdf[i-1], i)
            elseif rand() < (single_step_prop / (ϵ + single_step_prop))
                ### AGESS updates with only 1-d updates
                @views lpdf[i] = AGESS_single_step_1d!(x, y -> transform_posterior(y, Y_obs, 
                                                       ϕ[i,:,:], δ[i,:], τ_ph, a, 
                                                       b, N, P, K, D_ph, ph, ph1), true, 6.0, 
                                                       n_params, μ_adapt, 
                                                       Σ_chol_adapt.L, lpdf[i-1], perm, i)
            else
                ### Non-adaptive update
                @views lpdf[i] = AGESS_single_step!(x, z, y -> transform_posterior(y, Y_obs, 
                                                    ϕ[i,:,:], δ[i,:], τ_ph, a, 
                                                    b, N, P, K, D_ph, ph, ph1), true, 6.0, 
                                                    n_params, ph_AGESS, μ_0, 
                                                    Σ_chol_0.L, lpdf[i-1], i)
            end
        end

        ### Update variables
        @views Λ_ph = reshape(x[i, 1:K*P], (K, P))
        @views η_ph = reshape(x[(K*P + 1):(N*K + K*P)], (N, K))
        @views D_ph[diagind(D_ph)] .= exp.(x[(N*K + K*P + 1):(N*K + K*P + P)])

        ### Gibbs Updates
        @views phi_sampler!(ϕ[i,:,:], δ[i,:], Λ_ph, ν)
        @views delta_sampler!(δ[i,:], ϕ[i,:,:], Λ_ph, a_1, a_2, ph_δ)

        ### Update Adaptive Scheme
        w_i = i^(-w_const)
        Σ_chol_adapt_ph.U .= sqrt((1 - w_i)) .*  Σ_chol_adapt_ph.U
        @views ph_cholesky_update .= sqrt(w_i) .* (x[i,:] .- μ_adapt_ph)
        lowrankupdate!(Σ_chol_adapt_ph, ph_cholesky_update)
        @views μ_adapt_ph .= (1 - w_i) * μ_adapt_ph +  w_i * x[i,:]
            
        ## Adapt mean and covariance according to AIRMCMC
        if i == N_J
            Σ_chol_adapt.U .= Σ_chol_adapt_ph.U
            @views μ_adapt .= μ_adapt_ph
            n_j += 1
            N_J += floor(n_j^β)
        end

        ### Update next state
        if i < n_MCMC
            @views x[i+1,:] .= x[i,:]
            @views ϕ[i+1,:,:] .= ϕ[i,:,:]
            @views δ[i+1,:] .= δ[i,:]
        end

        ### Print statement
        if (i % 100) == 0
            @views println("MCMC iter: ", i, "  lpdf = ", mean(lpdf[i-99:i]))
        end
    end


    Λ_out = zeros(n_MCMC, K, P) 
    η_out = zeros(n_MCMC, N, K)
    D_out = zeros(n_MCMC, P, P)
    for i in 1:n_MCMC
        @views Λ_out[i,:,:] .= reshape(x[i, 1:K*P], (K, P))
        @views η_out[i,:,:] .= reshape(x[i, (K*P + 1):(N*K + K*P)], (N, K))
        for j in 1:P
            D_out[i,j,j] = exp(x[i, (N*K + K*P + j)])
        end
    end

    return Λ_out, η_out, D_out, ϕ, δ
end
```

We have constructed our custom MCMC sampler. We can now generate some synthetic data and
test the performance of our sampler.

```@example Factor
### Generate Data
K = 4
N = 200
P = 50

Σ = diagm(ones(P))
ph = zeros(P)
for i in 1:(K - 3)
    ph = randn(P)
    Σ .+= i .* (ph * ph')
end

μ_0 = zeros(P)
Y_obs = rand(MvNormal(μ_0, Σ), N)'
Σ_truth = deepcopy(Σ)


### Run custom MCMC
a_1 = 2.0
a_2 = 2.0
ν = 2.0
a = 1.0
b = 1.0
n_MCMC = 10000
ϵ = 0.05
single_step_prop = 0.001 
β = 0.5
burnin = 0.01


@time Λ_samp, η_samp, D_samp, ϕ_samp, δ_samp  = custom_MCMC(Y_obs, K, n_MCMC, a_1, a_2, a,
                                                      b, ϵ, single_step_prop, burnin,
                                                      ν, β)
```

We can look at estimates of $\Lambda'\Lambda + \mathbf{D}$ to see how well we recovered the covariance
structure. First, we will look at the true covariance matrix.

```@example Factor
heatmap(Σ_truth)
```

Now, we can look at the posterior element-wise mean of $\Lambda'\Lambda + \mathbf{D}$.
```@example Factor
function posterior_Σ(Λ_samp::AbstractArray{Y,3}, D_samp::AbstractArray{Y,3}, P::T; 
                     burnin = 0.5) where {Y<:AbstractFloat, T<:Integer}
    n_MCMC = size(Λ_samp)[1]
    burnin_num = floor(Int64, burnin * n_MCMC)
    posterior_samps = zeros(n_MCMC - burnin_num, P, P)
    for i in (burnin_num +1):n_MCMC
        @views posterior_samps[i - burnin_num,:,:] .= Λ_samp[i,:,:]' * Λ_samp[i,:,:]
        posterior_samps[i - burnin_num,:,:] .+= D_samp[i,:,:]
    end
    return posterior_samps
end

Σ_samp = posterior_Σ(Λ_samp, D_samp, P, burnin = 0.5)
Σ_mean = zeros(P,P)
for i in 1:P
    for j in 1:P
        Σ_mean[i,j] = mean(Σ_samp[:,i,j])
    end
end

heatmap(Σ_mean)
```

We can also look at the trace plots of individual elements of $\Lambda'\Lambda + \mathbf{D}$,
along with the true value (represented by the horizontal line).

```@example Factor
plot(Σ_samp[:,1,1])
hline!([Σ_truth[1,1]])
```

### Conclusion

As illustrated in this tutorial, the `AdaptEllipticalSliceSampler.jl` package can be used
to construct custom MCMC schemes. When considering factor analysis in this setting, one may notice
that we can simply use AGESS for sampling all parameters. Although this tutorial primarily serves 
as a guide for how to construct custom MCMC sampling schemes, we can compare the results obtained
when using AGESS to sample all parameters.

```@example Factor
function posterior2(Λ::AbstractMatrix{Y}, η::AbstractMatrix{Y}, Y_obs::AbstractMatrix{Y}, 
                   D::AbstractMatrix{Y}, ϕ::AbstractMatrix{Y}, δ::AbstractVector{Y}, 
                   τ_ph::AbstractVector{Y}, a_1::Y, a_2::Y, ν::Y, a::Y, b::Y,
                   ph::AbstractVector{Y}, ph1::AbstractVector{Y})::Float64 where {Y<:AbstractFloat}
    
    lpdf::Float64 = 0.0
    ## Likelihood
    for i in 1:size(Y_obs)[1]
        @views mul!(ph, Λ', η[i,:])
        @views ph .-= Y_obs[i,:]
        @views ph1 .= ph ./ D[diagind(D)]
        @views lpdf += (-0.5* sum(log.(D[diagind(D)])) - 0.5 * dot(ph1, ph))
    end

     ##Priors 
    for i in eachindex(δ)
        if i == 1
            lpdf += logpdf(Gamma(a_1, 1), exp(δ[1])) + δ[1]
        else 
            lpdf += logpdf(Gamma(a_2, 1), exp(δ[i])) + δ[i] 
        end
    end

    ig_d = InverseGamma(a,b)
    for i in 1:size(D)[1]
       lpdf += logpdf(ig_d, D[i,i]) + log(D[i,i])
    end

    τ_ph .= exp.(δ)
    for i in 2:length(δ)
        τ_ph[i] *= τ_ph[i-1]
    end

    gamma_d = Gamma(0.5 * ν, 2 / ν)
    for i in 1:size(Λ)[1], j in 1:size(Λ)[2]
        lpdf += 0.5 * log(exp(ϕ[i,j]) * τ_ph[i]) - 0.5 * (Λ[i,j]^2 * exp(ϕ[i,j]) * τ_ph[i])
        lpdf += logpdf(gamma_d, exp(ϕ[i,j])) + ϕ[i,j]
    end

    for i in 1:size(η)[1]
        @views lpdf += - 0.5 * dot(η[i,:], η[i,:])
    end

    return lpdf
end

function transform_posterior2(x::AbstractVector{Y}, Y_obs::AbstractMatrix{Y}, 
                              δ_ph::AbstractVector{Y}, τ_ph::AbstractVector{Y}, 
                              a_1::Y, a_2::Y, ν::Y, a::Y, b::Y,
                              N::T, P::T, K::T, D_ph::AbstractMatrix{Y}, 
                              ph::AbstractVector{Y}, 
                              ph1::AbstractVector{Y})::Float64 where {Y<:AbstractFloat, T<:Integer}
    @views Λ_ph = reshape(x[1:K*P], (K, P))
    @views η_ph = reshape(x[(K*P + 1):(N*K + K*P)], (N, K))
    @views D_ph[diagind(D_ph)] .= exp.(x[(N*K + K*P + 1):(N*K + K*P + P)])
    @views ϕ_ph = reshape(x[(N*K + K*P + P + 1):(N*K + 2*K*P + P)], (K, P))
    @views δ_ph .= x[(N*K + 2*K*P + P + 1):(N*K + 2*K*P + P + K)]
    lpdf = posterior2(Λ_ph, η_ph, Y_obs, D_ph, ϕ_ph, δ_ph, τ_ph, a_1, a_2, ν, a, b, ph, ph1)

    return lpdf
end
```

Now that we have specified a function evaluating the posterior log pdf, we can call the function
`AGESS`.

```@example Factor
# Set variables
τ_ph = zeros(K)
δ_ph = zeros(K)
D_ph = zeros(P,P)
ph = zeros(P)
ph1 = similar(ph)

n_params = N*K + 2*K*P + P + K

results = AGESS(x -> transform_posterior2(x, Y_obs, δ_ph, τ_ph, a_1, a_2, ν, a, b, N, P, K,
                                          D_ph, ph, ph1), n_MCMC, n_params, 
                                          single_step_prop = 0.01)
```
We can extract the parameters using the following code.

```@example Factor
Λ_samp2 = zeros(n_MCMC, K, P) 
η_samp2 = zeros(n_MCMC, N, K)
D_samp2 = zeros(n_MCMC, P, P)
δ_samp2 = zeros(n_MCMC, K)
ϕ_samp2 = zeros(n_MCMC, K, P)
for i in 1:n_MCMC
    @views Λ_samp2[i,:,:] .= reshape(results.samps[i, 1:K*P], (K, P))
    @views η_samp2[i,:,:] .= reshape(results.samps[i, (K*P + 1):(N*K + K*P)], (N, K))
    for j in 1:P
        D_samp2[i,j,j] = exp(results.samps[i, (N*K + K*P + j)])
    end
    @views ϕ_samp2[i,:,:] .= reshape(exp.(results.samps[i, (N*K + K*P + P + 1):(N*K + 2*K*P + P)]), (K, P))
    @views δ_samp2[i,:] .= exp.(results.samps[i, (N*K + 2*K*P + P + 1):(N*K + 2*K*P + P + K)])
end
```

Similarly to before, we can look at the posterior element-wise mean of $\Lambda'\Lambda + \mathbf{D}$.

```@example Factor
Σ_samp2 = posterior_Σ(Λ_samp2, D_samp2, P, burnin = 0.5)
Σ_mean2 = zeros(P,P)
for i in 1:P
    for j in 1:P
        Σ_mean2[i,j] = mean(Σ_samp2[:,i,j])
    end
end

heatmap(Σ_mean2)
```
Lastly, we can look at the trace plots of individual elements of $\Lambda'\Lambda + \mathbf{D}$,
along with the true value (represented by the horizontal line).

```@example Factor
plot(Σ_samp2[:,1,1])
hline!([Σ_truth[1,1]])
```


[^1]: N. Marco and S. T. Tokdar. Adaptive generalized elliptical slice sampling. arXiv preprint arXiv:2605.21659, 2026.

[^2]: A. Bhattacharya and D. B. Dunson. Sparse Bayesian infinite factor models. Biometrika, 98(2):291-306, 2011.