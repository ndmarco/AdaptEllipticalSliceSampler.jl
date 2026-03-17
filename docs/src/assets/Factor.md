# Factor Models

In this tutorial, we will cover a more advanced use-case of `AdaptEllipticalSliceSampler.JL`, 
where we use AGESS[^1] to sample a subset of parameters, and use Gibbs sampling to 
conditionally update the rest of the samplers. In this tutorial, we consider the following
factor model using the multiplicative gamma shrinkage prior[^2]:

$$\mathbf{y}_i = \Lambda \eta_i + \epsilon_i,$$

$$\epsilon_i \sim \mathcal{N}(0, \mathbf{D}),$$

$$\Lambda_{jh} \mid \phi_{jh}, \tau_h \sim \mathcal{N}(0, \phi_{jh}^{-1} \tau_h^{-1}),$$

$$\phi_{jh} \sim \text{Gamma}(\nu / 2, \nu / 2),$$

$$\tau_h = \prod_{k=1}^h \delta_j \;\;\; (h = 1, \dots, K)$$

$$\delta_1 \sim \text{Gamma}(a_1, 1),$$

$$\delta_{h} \sim \text{Gamma}(a_2, 1) \;\;\; (h \ge 2),$$

$$\mathbf{D}_{jj} \sim \text{Gamma}(a, b),$$

where $\mathbf{y}_i \in \mathbb{R}^P$, $\Lambda \in \mathbb{R}^{P \times K}$, $\eta_i \in \mathbb{R}^{K}$,
$\mathbf{D}$ is a Diagonal matrix, and $a_2 > 1$ (to promote shrinkage). In these settings, it is 
often the case that $K$ is much smaller than $P$. Under this setup, a Gibbs sampler is available[^2],
making sampling relatively straightforward. Consider the case where we want to use AGESS to sample
$\Lambda$, $\eta_i$, and $\mathbf{D}$, while using Gibbs updates for $\delta_h$ and
$\phi_{jh}$. We will start by constructing Gibbs samplers for $\delta_h$ and $\phi_{jh}$.

### Gibbs Updates

The conditional posterior distribution of $\delta_h$ and $\phi_{jh}$ are as follows[^2]:

$$\phi_{jh}\mid \Theta_{-\phi_{jh}} \sim \text{Gamma}\left(0.5(\nu + 1), 0.5(\nu + \tau_h \Lambda_{jh}^2)\right)$$

$$\delta_1 \mid \Theta_{-\delta_1} \sim \text{Gamma}\left(a_1 + 0.5(P\times K), 1 + 0.5(\sum_{l=1}^H \tau_{l}^{(-1)} \sum_{j=1}^P \phi_{jl}\Lambda_{jl}^2)\right)$$

$$\delta_h \mid \Theta_{-\delta_h} \sim \text{Gamma}\left(a_2 + 0.5(P\times (K - h +1)), 1 + 0.5(\sum_{l=h}^H \tau_{l}^{(-h)} \sum_{j=1}^P \phi_{jl}\Lambda_{jl}^2)\right)\;\;\; (2 \le h \le K)$$

We can construct Julia functions to update these parameters as follows.

```@example Factor
import Random, BenchmarkTools
using AdaptEllipticalSliceSampler
using Distributions
using Plots
using LinearAlgebra

Random.seed!(123)

function delta_sampler!(δ::AbstractVector{Y}, ϕ::AbstractMatrix{Y}, Λ::AbstractMatrix{Y}, 
                        a1_δ::Y, a2_δ::Y) where {Y<:AbstractFloat}
    factor_rank = length(δ)
    N = size(Λ)[1]
    ## Sample δ1
    α = a1_δ + 0.5 * factor_rank * N
    @views β = 1.0 + 0.5 * dot(ϕ[:,1] .* Λ[:,1], Λ[:,1])
    for i in 2:factor_rank
        @views τ = prod(δ[2:i])
        @views β += 0.5 * τ * dot(ϕ[:,i] .* Λ[:,i], Λ[:,i])
    end
    δ[1] = rand(Gamma(α, 1 / β))

    ## Sample rest of δ
    for i in 2:factor_rank
        α = a2_δ + 0.5 * (factor_rank - i + 1) * N
        β = 1.0 
       @views τ = prod(δ[1:(i-1)])
        for j in i:factor_rank
            if i != j
                τ *= δ[j]
            end
            @views β += 0.5 * τ * dot(ϕ[:,j] .* Λ[:,j], Λ[:,j])
        end
        δ[i] = rand(Gamma(α, 1 / β))
    end
 
    return nothing
end

function phi_sampler!(ϕ::AbstractMatrix{Y}, δ::AbstractVector{Y}, Λ::AbstractMatrix{Y}, 
                       ν::Y) where {Y<:AbstractFloat}
    factor_rank = length(δ)
    N =  size(Λ)[1]
    for i in 1:factor_rank
        @views τ = prod(δ[1:i])
        for j in 1:N
            α = (ν + 1) * 0.5
            β = 0.5 * (ν  + τ * Λ[j,i]^2)
            ϕ[j,i] = rand(Gamma(α, 1 / β))
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
                   τ_ph::AbstractVector{Y}, a_1::Y, a_2::Y, ν::Y, a::Y, b::Y,
                   ph::AbstractVector{Y})::Float64 where {Y<:AbstractFloat}
    
    lpdf::Float64 = 0.0
    ## Likelihood
    for i in 1:size(Y_obs)[1]
        @views mul!(ph, Λ', η[i,:])
        @views ph .-= Y_obs[i,:]
        @views lpdf += (-0.5* sum(log.(D[diagind(D)])) - 0.5 * dot(ph ./ D[diagind(D)], ph))
    end

    ig_d = InverseGamma(a,b)
    for i in 1:size(D)[1]
        @views lpdf += logpdf(ig_d, D[i,i])
    end

    τ_ph .= exp.(δ)
    for i in 2:length(δ)
        τ_ph[i] *= τ_ph[i-1]
    end

    for i in 1:size(Λ)[1], j in 1:size(Λ)[2]
        lpdf -= 0.5 * log(ϕ[i,j] * τ_ph[i]) + 0.5 * ((Λ[i,j]^2) / (ϕ[i,j] * τ_ph[i])^2)
    end

    for i in 1:size(η)[1]
        @views lpdf += - 0.5 * dot(η[i,:], η[i,:])
    end

    return lpdf
end

function transform_posterior(x::AbstractVector{Y}, Y_obs::AbstractMatrix{Y}, 
                             ϕ::AbstractMatrix{Y}, δ::AbstractVector{Y}, 
                             τ_ph::AbstractVector{Y}, a_1::Y, a_2::Y, ν::Y, a::Y, b::Y,
                             N::T, P::T, K::T, D_ph::AbstractMatrix{Y}, 
                             ph::AbstractVector{Y})::Float64 where {Y<:AbstractFloat, T<:Integer}
    @views Λ_ph = reshape(x[1:K*P], (K, P))
    @views η_ph = reshape(x[(K*P + 1):(N*K + K*P)], (N, K))
    @views D_ph[diagind(D_ph)] .= exp.(x[(N*K + K*P + 1):(N*K + K*P + P)])
    lpdf = posterior(Λ_ph, η_ph, Y_obs, D_ph, ϕ, δ, τ_ph, a_1, a_2, ν, a, b, ph)
    ### Jacobian for transformation
    lpdf += sum(x[(N*K + K*P + 1):(N*K + K*P + P)])

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

    ### Start MCMC
    for i in 2:n_MCMC
        ### AGESS update
        if i < (n_MCMC * burnin)
            ### 1-d AGESS updates for fast burn-in
            @views lpdf[i] = AGESS_single_step_1d!(x, y -> transform_posterior(y, Y_obs, 
                                                   ϕ[i,:,:], δ[i,:], τ_ph, a_1, a_2, ν, a, 
                                                   b, N, P, K, D_ph, ph), true, 6.0, 
                                                   n_params, μ_adapt, 
                                                   Σ_chol_adapt.L, i)
        else
            if rand() > (ϵ + single_step_prop)
                ### standard AGESS step
                @views lpdf[i] = AGESS_single_step!(x, z, y -> transform_posterior(y, Y_obs, 
                                                    ϕ[i,:,:], δ[i,:], τ_ph, a_1, a_2, ν, a, 
                                                    b, N, P, K, D_ph, ph), true, 6.0, 
                                                    n_params, ph_AGESS, μ_adapt, 
                                                    Σ_chol_adapt.L, i)
            elseif rand() < (single_step_prop / (ϵ + single_step_prop))
                ### AGESS updates with only 1-d updates
                @views lpdf[i] = AGESS_single_step_1d!(x, y -> transform_posterior(y, Y_obs, 
                                                       ϕ[i,:,:], δ[i,:], τ_ph, a_1, a_2, ν, a, 
                                                       b, N, P, K, D_ph, ph), true, 6.0, 
                                                       n_params, μ_adapt, 
                                                       Σ_chol_adapt.L, i)
            else
                ### Non-adaptive update
                @views lpdf[i] = AGESS_single_step!(x, z, y -> transform_posterior(y, Y_obs, 
                                                    ϕ[i,:,:], δ[i,:], τ_ph, a_1, a_2, ν, a, 
                                                    b, N, P, K, D_ph, ph), true, 6.0, 
                                                    n_params, ph_AGESS, μ_0, 
                                                    Σ_chol_0.L, i)
            end
        end

        ### Update variables
        @views Λ_ph = reshape(x[i, 1:K*P], (K, P))
        @views η_ph = reshape(x[(K*P + 1):(N*K + K*P)], (N, K))
        @views D_ph[diagind(D_ph)] .= exp.(x[(N*K + K*P + 1):(N*K + K*P + P)])

        ### Gibbs Updates
        @views phi_sampler!(ϕ[i,:,:], δ[i,:], Λ_ph, ν)
        @views delta_sampler!(δ[i,:], ϕ[i,:,:], Λ_ph, a_1, a_2)

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
        println("MCMC iter: ", i, "  lpdf = ", mean(lpdf[i]))

        ### Print statement
        if (i % 100) == 0
            println("MCMC iter: ", i, "  lpdf = ", mean(lpdf[i-99:i]))
        end
    end


    Λ_out = zeros(n_MCMC, K, P) 
    η_out = zeros(n_MCMC, N, K)
    D_out = zeros(n_MCMC, P, P)
    for i in 1:n_MCMC
        @views Λ_out[i,:,:] .= reshape(x[i, 1:K*P], (K, P))
        @views η_out = reshape(x[i, (K*P + 1):(N*K + K*P)], (N, K))
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
P = 100

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


Λ_out_samp, η_out_samp, D_samp, ϕ_samp, δ_samp  = custom_MCMC(Y_obs, K, n_MCMC, a_1, a_2, a,
                                                              b, ϵ, single_step_prop, burnin,
                                                              ν, β)

```


 [^1]: N. Marco and S. T. Tokdar. Adaptive Generalized Elliptical Slice Sampling
 [^2]: A. Bhattacharya and D. B. Dunson. Sparse Bayesian infinite factor models. Biometrika, 98(2):291-306, 2011.