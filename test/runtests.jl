using AdaptEllipticalSliceSampler
using Test, LinearAlgebra, Distributions, Turing

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
    @test maximum(mean(mcmc_out)[1:10,2] .- β) < 0.05
    ## Test recovery of scale parameter
    @test abs(mean(exp.(mcmc_out.value[:, 11,:])) - 0.1) < 0.2

    ####################################
    ## Test using Turing.jl framework ##
    ####################################
    @model function mv_linear_regression(X, Y)
        n, p = size(X)      # n observations, p predictors
        d = size(Y, 2)       # d response variables
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
    @test maximum(mean(mcmc_out)[2:11,2] .- β) < 0.05
    ## Test recovery of scale parameter
    @test abs(mean(exp.(mcmc_out.value[:, 1,:])) - 0.1) < 0.2
end
