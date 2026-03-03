using AdaptEllipticalSliceSampler
using Test 

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

    β, X, y = generate_data(10000, 2)

    mcmc_out = AGESS(β -> log_posterior(β, X, y), 10000, 3)

end
