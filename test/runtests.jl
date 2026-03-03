using AdaptEllipticalSliceSampler
using Test, LinearAlgebra, Distributions

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

    β, X, y = generate_data(1000, 10)
    mcmc_out = AGESS(β -> log_posterior(β, X, y), 1000, 11)

    ## Test recovery of β coefficients
    for i in 1:10
        @test abs(mean(mcmc_out.samps[500:1000, i]) - β[i]) < 0.05
    end

    ## Test recovery of scale parameter
    @test abs(mean(exp.(mcmc_out.samps[500:1000, 11])) - 0.1) < 0.2


    ## Test type stability
    X_32 = convert(Matrix{Float32}, X)
    y_32 = convert(Vector{Float32}, y)
    β_32 = convert(Vector{Float32}, β)

    burnin = 0.25

    mcmc_out = AGESS(β -> log_posterior(β, X_32, y_32), Int32(1000), Int32(11), μ_0 = Float32(0), 
                                        Σ_0 = Float32(1), init_x = Float32(0), burnin = Float32(0.25),
                                        ν = Float32(6), ϵ = Float32(0.1), single_step_prop = Float32(0.05), 
                                        β = Float32(0.5))
    
    @test eltype(mcmc_out.samps) == Float32
    @test eltype(mcmc_out.l_pdf) == Float32
    @test eltype(mcmc_out.adapted_μ) == Float32
    @test eltype(mcmc_out.adapted_Σ) == Float32
end
