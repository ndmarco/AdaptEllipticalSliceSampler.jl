using AdaptEllipticalSliceSampler
using Test, LinearAlgebra, Distributions, Turing, DynamicPPL, Random

@testset "AdaptEllipticalSliceSampler.jl" begin
    
    function generate_data(N::T, P::T) where {T<:Integer}
        β = randn(P) * (2 * log(P))^(1.0 / 4)
        x = randn(N, P)
        y = zeros(Float64, N)
        for i in 1:N
            y[i] = randn() * 0.1 + dot(x[i,:], β)
        end
    
        return β, x, y
    end

    function log_posterior(β::AbstractVector{Y}, X::AbstractMatrix{Y}, y::AbstractVector{Y}) where {Y<: AbstractFloat}
        P = length(β)
        ## Normal Likelihood
        lpdf = -0.5 * (1 / exp(β[P])) *  norm(X * β[1:P-1] - y)^2 - (0.5 * length(y) * β[P])
 
        ## Priors
        ## Std Normal prior on coefficients
        lpdf += -0.5 * norm(β[1:P-1])^2

        ## IG(1,1) prior on scale parameter (log-transformed)
        lpdf += -1 * β[P]  -  (1 / exp(β[P]))
        
        return lpdf
    end

    ####################################
    ## Test direct way of using AGESS ##
    ####################################
    β, X, y = generate_data(1000, 10)
    mcmc_out = AGESS(β -> log_posterior(β, X, y), 1000, 11)
    mcmc_out = mcmc_out[500:1000,:,:]

    ## Test recovery of β coefficients
    @test maximum(abs.(mean(mcmc_out)[1:10,2] .- β)) < 0.05
    ## Test recovery of scale parameter
    @test abs(mean(exp.(mcmc_out.value[:, 11,:])) - 0.01) < 0.05

    ####################################
    ## Test using Turing.jl framework ##
    ####################################
    @model function mv_linear_regression(X, Y)
        n, p = size(X)                          # n observations, p predictors
        d = size(Y, 2)                          # d response variables
        # Priors
        σ² ~ InverseGamma(1.0, 1.0)
        B ~ filldist(Normal(0, 1.0), p, d)     # coefficient matrix, p × d

        # Likelihood
        μ = X * B
        for i in 1:n
            Y[i, :] ~ MvNormal(μ[i, :], σ² * I)
        end
    end

    sampler = AGESSSampler(mv_linear_regression(X, y), 1000)
    mcmc_out = sample(mv_linear_regression(X, y), sampler, 1000)
    mcmc_out = mcmc_out[500:1000,:,:]

    ## Test recovery of β coefficients
    @test maximum(abs.(mean(mcmc_out)[2:11,2] .- β)) < 0.05
    ## Test recovery of scale parameter
    @test abs(mean(mcmc_out.value[:, 1,:]) - 0.01) < 0.05

    #############################################
    ## Test the `blocks` (1-d/block) framework ##
    #############################################
    log_posterior_fn(b) = log_posterior(b, X, y)

    ## Degenerate case: a block per coordinate with no `conditional` performs exactly the
    ## same per-coordinate scalar updates as the unblocked sampler, so with the same seed the
    ## resulting chains must be bit-identical.
    scalar_blocks = [AGESSBlock([j]) for j in 1:11]
    sampler_unblocked = AGESSSampler(11, 500)
    sampler_scalar_blocks = AGESSSampler(11, 500; blocks = scalar_blocks)
    chain_unblocked = sample(Xoshiro(11), AGESSModel(log_posterior_fn, 11), sampler_unblocked, 500)
    chain_scalar_blocks = sample(Xoshiro(11), AGESSModel(log_posterior_fn, 11), sampler_scalar_blocks, 500)
    @test chain_unblocked.value == chain_scalar_blocks.value

    ## A genuine multi-dimensional block (the 10 coefficients together) with a hand-written
    ## `conditional` (the IG(1,1) term on the log-scale parameter is constant w.r.t. this
    ## block and may be omitted) must still recover the true coefficients and scale.
    function coef_conditional(b::AbstractVector{Y}) where {Y<:AbstractFloat}
        Pl = length(b)
        lpdf = -0.5 * (1 / exp(b[Pl])) * norm(X * b[1:Pl-1] - y)^2
        lpdf += -0.5 * norm(b[1:Pl-1])^2
        return lpdf
    end
    mixed_blocks = [AGESSBlock(collect(1:10); conditional = coef_conditional), AGESSBlock([11])]
    sampler_mixed_blocks = AGESSSampler(11, 1000; blocks = mixed_blocks)
    chain_mixed_blocks = sample(Xoshiro(11), AGESSModel(log_posterior_fn, 11), sampler_mixed_blocks, 1000)
    chain_mixed_blocks = chain_mixed_blocks[500:1000, :, :]
    @test maximum(abs.(mean(chain_mixed_blocks)[1:10, 2] .- β)) < 0.05
    @test abs(mean(exp.(chain_mixed_blocks.value[:, 11, :])) - 0.01) < 0.05

    ## `blocks` must partition 1:P exactly; overlapping/incomplete coverage should error.
    @test_throws ArgumentError AGESSSampler(11, 500; blocks = [AGESSBlock([1, 2]), AGESSBlock([2, 3])])

    #################################################################
    ## Test (Turing) DynamicPPL submodel-based blocks (AGESSBlock) ##
    #################################################################
    G_h = 8
    n_g_h = 200
    σ_obs_h = 1.0
    μ_true_h = 1.5
    τ_true_h = 0.6
    θ_true_h = μ_true_h .+ τ_true_h .* randn(G_h)
    data_h = [θ_true_h[g] .+ σ_obs_h .* randn(n_g_h) for g in 1:G_h]

    @model function local_model(g, data_g, μ, τ)
        θ ~ Normal(μ, τ)
        for i in eachindex(data_g)
            data_g[i] ~ Normal(θ, σ_obs_h)
        end
    end

    @model function hierarchical_model(data)
        μ ~ Normal(0, 10)
        τ ~ InverseGamma(2.0, 1.0)
        groups = Vector{Any}(undef, G_h)
        for g in 1:G_h
            groups[g] ~ to_submodel(local_model(g, data[g], μ, τ))
        end
    end

    hmodel = hierarchical_model(data_h)
    n_MCMC_h = 2000
    burn_h = 1000

    ## Unblocked baseline, for cross-checking against the blocked run below.
    sampler_h_default = AGESSSampler(hmodel, n_MCMC_h)
    chain_h_default = sample(Xoshiro(13), hmodel, sampler_h_default, n_MCMC_h)
    chain_h_default = chain_h_default[burn_h:n_MCMC_h, :, :]

    ## One `AGESSBlock` per group, using a `submodel_constructor` conditional built from
    ## `local_model` (τ has an `InverseGamma` prior, so its linked/unconstrained value is
    ## `log(τ)`); μ and τ themselves are left on the default (full log_posterior) path.
    blocks_h = [
        AGESSBlock([2 + g], ctx -> local_model(g, data_h[g], ctx[1], exp(ctx[2])), [1, 2])
        for g in 1:G_h
    ]
    push!(blocks_h, AGESSBlock([1, 2]))
    sampler_h_blocked = AGESSSampler(hmodel, n_MCMC_h; blocks = blocks_h)
    chain_h_blocked = sample(Xoshiro(13), hmodel, sampler_h_blocked, n_MCMC_h)
    chain_h_blocked = chain_h_blocked[burn_h:n_MCMC_h, :, :]

    μ_default_h = mean(chain_h_default[:μ])
    μ_blocked_h = mean(chain_h_blocked[:μ])
    θ_default_h = [mean(chain_h_default[Symbol("groups[$(g)].θ")]) for g in 1:G_h]
    θ_blocked_h = [mean(chain_h_blocked[Symbol("groups[$(g)].θ")]) for g in 1:G_h]

    ## The blocked sampler recovers the true generative parameters...
    @test abs(μ_blocked_h - μ_true_h) < 0.5
    @test maximum(abs.(θ_blocked_h .- θ_true_h)) < 0.3
    ## ...and agrees closely with the unblocked sampler run from the same seed.
    @test abs(μ_blocked_h - μ_default_h) < 0.3
    @test maximum(abs.(θ_blocked_h .- θ_default_h)) < 0.2
end
