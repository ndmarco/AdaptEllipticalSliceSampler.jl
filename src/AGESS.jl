"""
    AGESS_MCMC_params(log_posterior, μ_0, Σ_0, t_dist, ν, β, single_step_prop, burnin, ϵ, n_MCMC, P)

A struct containing user-specified values.

# Arguments
- `log_posterior::Function`: the user-specified function
- `μ_0::AbstractVector{<:AbstractFloat}`: the user-specified initial mean 
- `Σ_0::AbstractMatrix{<:AbstractFloat}`: the user-specified initial scale
- `t_dist::Bool`: the user-specified value
- `ν::AbstractFloat`: the user-specified degrees of freedom
- `β::AbstractFloat`: the user-specified β
- `single_step_prop::AbstractFloat`: the user-specified value
- `burnin::AbstractFloat`: the user-specified value
- `ϵ::AbstractFloat`: the user-specified value
- `n_MCMC::Integer`: the user-specified number of iterations
- `P::Integer`: the user-specified dimension of target distribution
"""
struct AGESS_MCMC_params
    log_posterior::Function
    μ_0::AbstractVector{<:AbstractFloat}
    Σ_0::AbstractMatrix{<:AbstractFloat}
    t_dist::Bool
    ν::AbstractFloat
    β::AbstractFloat
    single_step_prop::AbstractFloat
    burnin::AbstractFloat
    ϵ::AbstractFloat
    n_MCMC::Integer
    P::Integer
end

struct MCMC_output
    samps::AbstractMatrix{<:AbstractFloat}
    l_pdf::AbstractVector{<:AbstractFloat}
    params::AGESS_MCMC_params
    adapted_Σ::AbstractMatrix{<:AbstractFloat}
    adapted_μ::AbstractVector{<:AbstractFloat}
end

"""
    AGESS_single_step!(x, z, params, ph, μ_adapt, Σ_chol_adapt, i)

Performs one iteration of adaptive generalized elliptical slice sampling.

Performs an in-place update of the Markov chain using AGESS. This function should only be used 
for custom (advanced) sampling schemes (see `AGESS` for general use). The matrix `x` contains the Markov 
chain, while `z` is an auxiliary variable for constructing the ellipse. `ph` is a vector used 
for intermediate calculations of same dimension of `z` (i.e. dimension of the target distribution). 
`params` contains all the information of the Markov chain, including log pdf of target distribution. 
`μ_adapt` and `Σ_chol_adapt` contain information about the mean and scale parameters of the adapted 
distribution, while `i` contains the current state of the Markov chain. The current state of 
X should be populated with the last state of the Markov chain.

# Arguments
- `x::AbstractMatrix{<:AbstractFloat}`: a matrix containing the Markov chain (n_MCMC x P)
- `z::AbstractVector{<:AbstractFloat}`: a vector used to create the ellipse (dim = P)
- `log_posterior::Function`: a function evaluating the log posterior pdf with a vector of parameters as the only input
- `t_dist::Bool`: a Boolean containing whether to use the T-distribution to generate ellipses
- `ν::AbstractFloat`: the user-specified degrees of freedom
- `P::Integer`: the dimension of the target distribution
- `ph::AbstractVector{<:AbstractFloat}`: a vector used for intermediate calculations (dim = P)
- `μ_adapt::AbstractVector{<:AbstractFloat}`: a vector containing the mean parameter of adapted distribution (dim = P)
- `Σ_chol_adapt::LowerTriangular{<:AbstractFloat, <:AbstractMatrix{<:AbstractFloat}}`: a lower triangular matrix containing the cholesky decomposition of the scale parameter of the adapted matrix
- `i::Integer`: the iteration of the Markov chain 

# Examples
For examples, please view the `Tutorials` section of the documentation.

# References
N. Marco and S. T. Tokdar. Adaptive generalized elliptical slice sampling. arXiv preprint arXiv:2605.21659, 2026.
"""
function AGESS_single_step!(x::AbstractMatrix{Y}, z::AbstractVector{Y}, log_posterior::Function, 
                            t_dist::Bool, ν::Y, P::T, ph::AbstractVector{Y}, 
                            μ_adapt::AbstractVector{Y}, Σ_chol_adapt::LowerTriangular{Y, <:AbstractMatrix{Y}}, 
                            i::T) where {Y<:AbstractFloat, T<:Integer}
    l_pdf::eltype(x) = 0.0
    y::eltype(x) = 0.0
    L_star::eltype(x) = 0.0
    ## Propose new z
    if t_dist == true
        @views cond_rMvT!(z, x[i,:], μ_adapt, Σ_chol_adapt, ν, ph, P)
    else
        z .= Σ_chol_adapt * randn(P) .+ μ_adapt
    end

    @views y = log_posterior(x[i,:]) + log(rand())
    if t_dist == true
        @views y -= dMvT(x[i,:], μ_adapt, Σ_chol_adapt, ph, ν, P)
    else
        @views y -= dMvN(x[i,:], μ_adapt, Σ_chol_adapt, ph)
    end

    ## Propose Initial Angle
    θ = rand(eltype(x)) * 2 * π
    θ_min = θ - 2 * π
    θ_max = θ

    ## Propose initial first move
    @views @. x[i,:] = ((x[i-1,:] - μ_adapt) * cos(θ) +  (z - μ_adapt) * sin(θ)) + μ_adapt
    @views L_star = log_posterior(x[i,:])
    l_pdf = L_star
    if t_dist == true
        @views L_star -= dMvT(x[i,:], μ_adapt, Σ_chol_adapt, ph, ν, P)
    else
        @views L_star -= dMvN(x[i,:], μ_adapt, Σ_chol_adapt, ph)
    end

    ## Check to make sure that posterior pdfs are computable
    if isnan(L_star)
        L_star = y - 1.0
    end
    if !isfinite(L_star)
        L_star = y - 1.0
    end

    while L_star <= y
        if θ < 0
            θ_min = θ
        else
            θ_max = θ
        end

        ## Propose new angle
        θ = θ_min + rand(eltype(x)) * (θ_max - θ_min)
        @views @. x[i,:] = ((x[i-1,:] - μ_adapt) * cos(θ) +  (z - μ_adapt) * sin(θ)) + μ_adapt
        @views L_star = log_posterior(x[i,:])
        l_pdf = L_star
        if t_dist == true
            @views L_star -= dMvT(x[i,:], μ_adapt, Σ_chol_adapt, ph, ν, P)
        else
            @views L_star -= dMvN(x[i,:], μ_adapt, Σ_chol_adapt, ph)
        end

        ## Check to make sure that posterior pdfs are computable
        if isnan(L_star)
            L_star = y - 1.0
        end
        if !isfinite(L_star)
            L_star = y - 1.0
        end
    end

    return l_pdf
end

"""
    AGESS_single_step_1d!(x, params, μ_adapt, Σ_chol_adapt, i)

Performs one iteration of adaptive generalized elliptical slice sampling in each dimension.

Performs an in-place update of the Markov chain using one-dimensional AGESS updates. This 
function should only be used for custom (advanced) sampling schemes (see `AGESS` for general use). 
The matrix `x` contains the Markov chain. `params` contains all the information of the Markov 
chain, including log pdf of target distribution. `μ_adapt` and `Σ_chol_adapt` contain 
information about the mean and scale parameters of the adapted 
distribution, while `i` contains the current state of the Markov chain. The current state of 
X should be populated with the last state of the Markov chain.

# Arguments
- `x::AbstractMatrix{<:AbstractFloat}`: a matrix containing the Markov chain (n_MCMC x P)
- `log_posterior::Function`: a function evaluating the log posterior pdf with a vector of parameters as the only input
- `t_dist::Bool`: a Boolean containing whether to use the T-distribution to generate ellipses
- `ν::AbstractFloat`: the user-specified degrees of freedom
- `P::Integer`: the dimension of the target distribution
- `μ_adapt::AbstractVector{<:AbstractFloat}`: a vector containing the mean parameter of adapted distribution (dim = P)
- `Σ_chol_adapt::LowerTriangular{<:AbstractFloat, <:AbstractMatrix{<:AbstractFloat}}`: a lower triangular matrix containing the cholesky decomposition of the scale parameter of the adapted matrix
- `i::Integer`: the iteration of the Markov chain 

# Examples
For examples, please view the `Tutorials` section of the documentation.

# References
N. Marco and S. T. Tokdar. Adaptive generalized elliptical slice sampling. arXiv preprint arXiv:2605.21659, 2026.
"""
function AGESS_single_step_1d!(x::AbstractMatrix{Y}, log_posterior::Function, 
                               t_dist::Bool, ν::Y,  P::T, μ_adapt::AbstractVector{Y}, 
                               Σ_chol_adapt::LowerTriangular{Y, <:AbstractMatrix{Y}}, 
                               i::T) where {Y<:AbstractFloat, T<:Integer}
    l_pdf::eltype(x) = 0.0
    z::eltype(x) = 0.0
    y::eltype(x) = 0.0
    L_star::eltype(x) = 0.0
    for j in randperm(P)
        
        ## Propose new z from N(0, Σ)
        if t_dist == true
            z = cond_rMvT_1d(x[i,j], μ_adapt[j], Σ_chol_adapt[j,j], ν)
        else
            z = Σ_chol_adapt[j,j] * randn() + μ_adapt[j]
        end

        @views y = log_posterior(x[i,:]) + log(rand())
        if t_dist == true
            y -= dMvT_1d(x[i,j], μ_adapt[j], Σ_chol_adapt[j,j], ν)
        else
            y -= dMvN_1d(x[i,j], μ_adapt[j], Σ_chol_adapt[j,j])
        end

        ## Propose Initial Angle
        θ = rand(eltype(x)) * 2 * π
        θ_min = θ - 2 * π
        θ_max = θ

        ## Propose initial first move
        x[i,j] = ((x[i-1,j] - μ_adapt[j]) * cos(θ) +  (z - μ_adapt[j]) * sin(θ)) + μ_adapt[j]
        @views L_star = log_posterior(x[i,:])
        l_pdf = L_star
        if t_dist == true
            L_star -= dMvT_1d(x[i,j], μ_adapt[j], Σ_chol_adapt[j,j], ν)
        else
            L_star -= dMvN_1d(x[i,j], μ_adapt[j], Σ_chol_adapt[j,j])
        end

        ## Check to make sure that posterior pdfs are computable
        if isnan(L_star)
            L_star = y - 1.0
        end
        if !isfinite(L_star)
            L_star = y - 1.0
        end

        while L_star <= y
            if θ < 0
                θ_min = θ
            else
                θ_max = θ
            end

            ## Propose new angle
            θ = θ_min + rand(eltype(x)) * (θ_max - θ_min)
            x[i,j] = ((x[i-1,j] - μ_adapt[j]) * cos(θ) +  (z - μ_adapt[j]) * sin(θ)) + μ_adapt[j]
            @views L_star = log_posterior(x[i,:])
            l_pdf = L_star
            if t_dist == true
                L_star -= dMvT_1d(x[i,j], μ_adapt[j], Σ_chol_adapt[j,j], ν)
            else
                L_star -= dMvN_1d(x[i,j], μ_adapt[j], Σ_chol_adapt[j,j])
            end

            ## Check to make sure that posterior pdfs are computable
            if isnan(L_star)
                L_star = y - 1.0
            end   
            if !isfinite(L_star)
                L_star = y - 1.0
            end
        end
    end

    return l_pdf
end

"""
    AGESS(log_posterior, n_MCMC, P; μ_0, Σ_0, init_x, t_dist, ν, burnin, ϵ, single_step_prop, β)

Performs adaptive generalized elliptical slice sampling.

Performs AGESS on a target distribution specified by `log_posterior`. The target distribution 
is of dimension `P` and `n_MCMC` iterations of Markov chain Monte Carlo will be performed using 
the AGESS transition scheme.

# Arguments
- `log_posterior::Function`: a function evaluating the posterior log pdf with only the state (or variables) as the input
- `n_MCMC::Integer`: the number of iterations to run the Markov chain for
- `P::Integer`: the dimension of the target distribution 

# Keyword Arguments
- `μ_0::Union{AbstractVector{<:AbstractFloat}, AbstractFloat} = 0.0`: a vector (or number which will be multiplied by the one vector) containing the initial (or prior) mean of adaptive distribution
- `Σ_0::Union{AbstractMatrix{<:AbstractFloat}, AbstractFloat} = 1.0`: a matrix (or number which will be multiplied by I) containing the initial (or prior) scale of adaptive distribution
- `init_x::Union{AbstractVector{<:AbstractFloat}, AbstractFloat} = 0.0`: a vector (or number which will be multiplied by the one vector) containing the initial starting location of the Markov chain
- `t_dist::Bool = true`: a Boolean variable indicating whether to use a t-distribution (true) or normal distribution (false) for elliptical slice sampling (Note: should almost always use t-distribution)
- `ν::AbstractFloat = 6.0`: the degrees of freedom of the t-distribution
- `burnin::AbstractFloat = 0.25`: the proportion of chain used for burnin
- `ϵ::AbstractFloat = 0.1`: the proportion of non-adaptive transitions
- `single_step_prop::AbstractFloat = 0.05`: the proportion of transitions where we perform one-dimensional updated (P >= 10)
- `β::AbstractFloat = 0.5`: the rate at which the adaptation diminishes

# Returns
`output`: a struct containing the following:
- `samps::AbstractMatrix{<:AbstractFloat}`: a matrix containing the states of the Markov chain (n_MCMC x P)
- `l_pdf::AbstractVector{<:AbstractFloat}`: a vector containing the posterior log pdf evaluated at each state of the Markov chain
- `params::struct`: a struct containing the following:
    * `log_posterior::Function`: the user-specified function
    * `μ_0::AbstractVector{<:AbstractFloat}`: the user-specified initial mean 
    * `Σ_0::AbstractMatrix{<:AbstractFloat}`: the user-specified initial scale
    * `t_dist::Bool`: the user-specified value
    * `ν::AbstractFloat`: the user-specified degrees of freedom
    * `β::AbstractFloat`: the user-specified β
    * `single_step_prop::AbstractFloat`: the user-specified value
    * `burnin::AbstractFloat`: the user-specified value
    * `ϵ::AbstractFloat`: the user-specified value
    * `n_MCMC::Integer`: the user-specified number of iterations
    * `P::Integer`: the user-specified dimension of target distribution
- `adapted_Σ::AbstractMatrix{<:AbstractFloat}`: the adapted (learned) scale matrix
- `adapted_μ::AbstractVector{<:AbstractFloat}`: the adapted (learned) mean vector


# Examples
For examples, please view the `Tutorials` section of the documentation.

# References
N. Marco and S. T. Tokdar. Adaptive generalized elliptical slice sampling. arXiv preprint arXiv:2605.21659, 2026.
"""
function AGESS(log_posterior::Function, n_MCMC::T, P::T;
               μ_0::Union{<:AbstractVector{Y},Y} = 0.0, Σ_0::Union{<:AbstractMatrix{Y},Y} = 1.0,
               init_x::Union{<:AbstractVector{Y},Y} = 0.0, t_dist::Bool = true, ν::Y = 6.0, burnin::Y = 0.25,
               ϵ::Y = 0.1, single_step_prop::Y = 0.05, β::Y = 0.5) where {Y<:AbstractFloat, T<:Integer}
   # Get prior mean parameter
   if typeof(μ_0) <: AbstractFloat
        μ_0 = ones(typeof(μ_0), P) .* μ_0
    end
    @argcheck length(μ_0) == P

    # Get prior for variance parameter
    if typeof(Σ_0) <: AbstractFloat
        @argcheck Σ_0 > 0.0
        Σ_0 = diagm(ones(typeof(Σ_0), P)) .* Σ_0
    end

    # Initial starting point of markov chain
    if typeof(init_x) <: AbstractFloat
        init_x = ones(typeof(init_x), P) .* init_x
    end

    ## Check rest of arguments
    @argcheck P >= 1
    @argcheck burnin >= 0.0
    @argcheck burnin < 1.0
    @argcheck ϵ >= 0.0
    @argcheck ϵ < 1.0
    @argcheck single_step_prop >= 0.0
    @argcheck single_step_prop < 1.0
    @argcheck single_step_prop + ϵ <= 1.0
    
    ## Initialize MCMC iterations
    x = zeros(eltype(μ_0), n_MCMC, P)
    x[1,:] .= init_x
    x[2,:] .= init_x

    ## Auxiliary variable
    z = zeros(eltype(μ_0), P)

    ## Storage of log_posterior evaluations
    l_pdf = zeros(eltype(μ_0), n_MCMC) 

    ## Construct AGESS_MCMC_params
    params = AGESS_MCMC_params(log_posterior, μ_0, Σ_0, t_dist, ν, β, single_step_prop,
                               burnin, ϵ, n_MCMC, P)

    l_pdf[1] = params.log_posterior(init_x)
    @argcheck isfinite(l_pdf[1]) "Initial starting position of Markov chain must have finite posterior density above 0"


    burnin_num = floor(typeof(n_MCMC), params.burnin * params.n_MCMC)

    μ_adapt = deepcopy(params.μ_0)
    μ_adapt_ph = deepcopy(params.μ_0)
    ph = similar(μ_adapt)

    Σ_chol = cholesky(params.Σ_0)
    Σ_chol_adapt = deepcopy(Σ_chol)
    Σ_chol_adapt_ph = deepcopy(Σ_chol)

    ph_cholesky_update = ones(eltype(μ_0), P)
    w_const = max(2/3, ((cbrt(P) - 1) / cbrt(P)))
    N_J = 2
    n_j = 2

    prog = Progress(n_MCMC)

    ind_range = 2:n_MCMC
    ind_range = convert(UnitRange{eltype(P)}, ind_range)
    for i in ind_range
        if P >= 10
            if i < (burnin_num * params.single_step_prop)
                l_pdf[i] = AGESS_single_step_1d!(x, params.log_posterior, params.t_dist, 
                                                 params.ν, params.P, μ_adapt, Σ_chol_adapt.L, i)
            else
                if rand() > (params.ϵ + params.single_step_prop)
                    l_pdf[i] = AGESS_single_step!(x, z, params.log_posterior, params.t_dist, 
                                                  params.ν, params.P, ph, μ_adapt, Σ_chol_adapt.L, i)
                elseif rand() < (params.single_step_prop / (params.ϵ + params.single_step_prop))
                    l_pdf[i] = AGESS_single_step_1d!(x, params.log_posterior, params.t_dist, 
                                                     params.ν, params.P, μ_adapt, Σ_chol_adapt.L, i)
                else
                    l_pdf[i] = AGESS_single_step!(x, z, params.log_posterior, params.t_dist, 
                                                  params.ν, params.P, ph, μ_0, Σ_chol.L, i)
                end
            end
        else
            if rand() > params.ϵ
                l_pdf[i] = AGESS_single_step!(x, z, params.log_posterior, params.t_dist, 
                                              params.ν, params.P, ph, μ_adapt, Σ_chol_adapt.L, i)
            else
                l_pdf[i] = AGESS_single_step!(x, z, params.log_posterior, params.t_dist, 
                                              params.ν, params.P, ph, μ_0, Σ_chol.L, i)
            end
        end
        
        w_i = i^(-w_const)
        Σ_chol_adapt_ph.U .= sqrt((1 - w_i)) .*  Σ_chol_adapt_ph.U
        @views ph_cholesky_update .= sqrt(w_i) .* (x[i,:] .- μ_adapt_ph)
        lowrankupdate!(Σ_chol_adapt_ph, ph_cholesky_update)
        @views μ_adapt_ph .= (1 - w_i) * μ_adapt_ph +  w_i * x[i,:]
        
        ## Adapt mean and covariance
        if i == N_J
            Σ_chol_adapt.U .= Σ_chol_adapt_ph.U
            μ_adapt .= μ_adapt_ph
            n_j += 1
            N_J += floor(n_j^β)
        end

        ## Populate next value in Markov Chain
        if i < n_MCMC
            @views x[i+1,:] .= x[i,:]
        end

        # Update User
        if (i % 100) == 0
            update!(prog, Int64(i); showvalues = [("average lpdf", mean(l_pdf[(i-99):i]))])
        end
        
    end

    output = MCMC_output(x, l_pdf, params, Σ_chol_adapt.L * Σ_chol_adapt.U, μ_adapt)

    return output
end
