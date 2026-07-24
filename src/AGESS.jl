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
- `x_current::AbstractVector{<:AbstractFloat}`: a vector containing the current state of the Markov Chain
- `x_next::AbstractVector{<:AbstractFloat}`: a vector which will contain the next state of the Markov Chain
- `z::AbstractVector{<:AbstractFloat}`: a vector used to create the ellipse (dim = P)
- `log_posterior::Function`: a function evaluating the log posterior pdf with a vector of parameters as the only input
- `t_dist::Bool`: a Boolean containing whether to use the T-distribution to generate ellipses
- `ν::AbstractFloat`: the user-specified degrees of freedom
- `P::Integer`: the dimension of the target distribution
- `ph::AbstractVector{<:AbstractFloat}`: a vector used for intermediate calculations (dim = P)
- `μ_adapt::AbstractVector{<:AbstractFloat}`: a vector containing the mean parameter of adapted distribution (dim = P)
- `Σ_chol_adapt::LowerTriangular{<:AbstractFloat, <:AbstractMatrix{<:AbstractFloat}}`: a lower triangular matrix containing the cholesky decomposition of the scale parameter of the adapted matrix
- `l_pdf::AbstractFloat`: the log posterior density of current state
- `rng::Random.AbstractRNG = Random.default_rng()`: random number generator

# Examples
For examples, please view the `Tutorials` section of the documentation.

# References
N. Marco and S. T. Tokdar. Adaptive generalized elliptical slice sampling. arXiv preprint arXiv:2605.21659, 2026.
"""
function AGESS_single_step!(x_current::AbstractVector{Y}, x_next::AbstractVector{Y},
                            z::AbstractVector{Y}, log_posterior::Function,
                            t_dist::Bool, ν::Y, P::T, ph::AbstractVector{Y},
                            μ_adapt::AbstractVector{Y}, Σ_chol_adapt::LowerTriangular{Y, <:AbstractMatrix{Y}},
                            l_pdf::Y; rng::Random.AbstractRNG = Random.default_rng()) where {Y<:AbstractFloat, T<:Integer}
    y::eltype(x_current) = 0.0
    L_star::eltype(x_current) = 0.0
    ## Propose new z
    if t_dist == true
        @views cond_rMvT!(rng, z, x_current, μ_adapt, Σ_chol_adapt, ν, ph, P)
    else
        randn!(rng, z)
        lmul!(Σ_chol_adapt, z)
        z .+= μ_adapt
    end

    y = l_pdf + log(rand(rng))
    if t_dist == true
        @views y -= dMvT(x_current, μ_adapt, Σ_chol_adapt, ph, ν, P)
    else
        @views y -= dMvN(x_current, μ_adapt, Σ_chol_adapt, ph)
    end

    ## Propose Initial Angle
    θ = rand(rng, eltype(x_next)) * 2 * π
    θ_min = θ - 2 * π
    θ_max = θ

    ## Propose initial first move
    @. x_next = ((x_current - μ_adapt) * cos(θ) +  (z - μ_adapt) * sin(θ)) + μ_adapt
    @views L_star = log_posterior(x_next)
    l_pdf = L_star
    if t_dist == true
        L_star -= dMvT(x_next, μ_adapt, Σ_chol_adapt, ph, ν, P)
    else
        L_star -= dMvN(x_next, μ_adapt, Σ_chol_adapt, ph)
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
        θ = θ_min + rand(rng, eltype(x_next)) * (θ_max - θ_min)
        @. x_next = ((x_current - μ_adapt) * cos(θ) +  (z - μ_adapt) * sin(θ)) + μ_adapt
        @views L_star = log_posterior(x_next)
        l_pdf = L_star
        if t_dist == true
            @views L_star -= dMvT(x_next, μ_adapt, Σ_chol_adapt, ph, ν, P)
        else
            @views L_star -= dMvN(x_next, μ_adapt, Σ_chol_adapt, ph)
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
- `x_current::AbstractVector{<:AbstractFloat}`: a vector containing the current state of the Markov Chain
- `x_next::AbstractVector{<:AbstractFloat}`: a vector which will contain the next state of the Markov Chain
- `log_posterior::Function`: a function evaluating the log posterior pdf with a vector of parameters as the only input
- `t_dist::Bool`: a Boolean containing whether to use the T-distribution to generate ellipses
- `ν::AbstractFloat`: the user-specified degrees of freedom
- `μ_adapt::AbstractVector{<:AbstractFloat}`: a vector containing the mean parameter of adapted distribution (dim = P)
- `Σ_chol_adapt::LowerTriangular{<:AbstractFloat, <:AbstractMatrix{<:AbstractFloat}}`: a lower triangular matrix containing the cholesky decomposition of the scale parameter of the adapted matrix
- `l_pdf::AbstractFloat`: the log posterior density of current state
- `perm::AbstractVector{<:Integer}`: a vector containing a placeholder for the permutation of indices
- `rng::Random.AbstractRNG = Random.default_rng()`: random number generator

# Examples
For examples, please view the `Tutorials` section of the documentation.

# References
N. Marco and S. T. Tokdar. Adaptive generalized elliptical slice sampling. arXiv preprint arXiv:2605.21659, 2026.
"""
function AGESS_single_step_1d!(x_current::AbstractVector{Y}, x_next::AbstractVector{Y},
                               log_posterior::Function, t_dist::Bool, ν::Y,
                               μ_adapt::AbstractVector{Y},
                               Σ_chol_adapt::LowerTriangular{Y, <:AbstractMatrix{Y}},
                               l_pdf::Y, perm::AbstractVector{T};
                               rng::Random.AbstractRNG = Random.default_rng()) where {Y<:AbstractFloat, T<:Integer}
    z::eltype(x_current) = 0.0
    y::eltype(x_current) = 0.0
    L_star::eltype(x_current) = 0.0

    x_next .= x_current
    randperm!(rng, perm)
    for j in perm
        ## Propose new z from N(0, Σ)
        if t_dist == true
            z = cond_rMvT_1d(rng, x_current[j], μ_adapt[j], Σ_chol_adapt[j,j], ν)
        else
            z = Σ_chol_adapt[j,j] * randn(rng) + μ_adapt[j]
        end

        y = l_pdf + log(rand(rng))
        if t_dist == true
            y -= dMvT_1d(x_current[j], μ_adapt[j], Σ_chol_adapt[j,j], ν)
        else
            y -= dMvN_1d(x_current[j], μ_adapt[j], Σ_chol_adapt[j,j])
        end

        ## Propose Initial Angle
        θ = rand(rng, eltype(x_current)) * 2 * π
        θ_min = θ - 2 * π
        θ_max = θ

        ## Propose initial first move
        x_next[j] = ((x_current[j] - μ_adapt[j]) * cos(θ) +  (z - μ_adapt[j]) * sin(θ)) + μ_adapt[j]
        @views L_star = log_posterior(x_next)
        l_pdf = L_star
        if t_dist == true
            L_star -= dMvT_1d(x_next[j], μ_adapt[j], Σ_chol_adapt[j,j], ν)
        else
            L_star -= dMvN_1d(x_next[j], μ_adapt[j], Σ_chol_adapt[j,j])
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
            θ = θ_min + rand(rng, eltype(x_current)) * (θ_max - θ_min)
            x_next[j] = ((x_current[j] - μ_adapt[j]) * cos(θ) +  (z - μ_adapt[j]) * sin(θ)) + μ_adapt[j]
            @views L_star = log_posterior(x_next)
            l_pdf = L_star
            if t_dist == true
                L_star -= dMvT_1d(x_next[j], μ_adapt[j], Σ_chol_adapt[j,j], ν)
            else
                L_star -= dMvN_1d(x_next[j], μ_adapt[j], Σ_chol_adapt[j,j])
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
- `ϵ::AbstractFloat = 0.05`: the proportion of non-adaptive transitions
- `single_step_prop::AbstractFloat = 0.05`: the proportion of transitions where we perform one-dimensional updated (P >= 10)
- `β::AbstractFloat = 0.5`: the rate at which the adaptation diminishes
- `param_names`: optional vector of parameter names (dimension P)

# Returns
`output`: a struct of type MCMCChains.Chains:
- `value`: An `AxisArray` object with axes `iter` × `var` × `chains`
- `logevidence` : A field containing the logevidence.
- `name_map` : A `NamedTuple` mapping each variable to a section.
- `info` : A `NamedTuple` containing miscellaneous information relevant to the chain.

# Examples
For examples, please view the `Tutorials` section of the documentation.

# References
N. Marco and S. T. Tokdar. Adaptive generalized elliptical slice sampling. arXiv preprint arXiv:2605.21659, 2026.
"""
function AGESS(log_posterior::Function, n_MCMC::T, P::T;
               rng::Random.AbstractRNG = Random.default_rng(),
               μ_0::Union{<:AbstractVector{Y},Y} = 0.0, Σ_0::Union{<:AbstractMatrix{Y},Y} = 1.0,
               init_x::Union{<:AbstractVector{Y},Y} = 0.0, t_dist::Bool = true, ν::Y = 6.0, burnin::Y = 0.5,
               ϵ::Y = 0.05, single_step_prop::Y = 0.05, β::Y = 0.5, param_names=missing) where {Y<:AbstractFloat, T<:Integer}
    ### Construct Sampler
    sampler = AGESSSampler(P, n_MCMC, μ_0 = μ_0, Σ_0 = Σ_0, init_x = init_x, t_dist = t_dist, ν = ν, 
                           burnin = burnin, ϵ = ϵ, single_step_prop = single_step_prop, β = β)
    
    ### Construct Model
    model = AGESSModel(log_posterior, P)

    if ismissing(param_names)
        param_names = [Symbol(:param_, i) for i in 1:P]
    else
        @argcheck length(param_names) == P "param_names must have length $(P), got $(length(param_names))"
        param_names = Symbol.(param_names)
    end

    ### Run MCMC
    chain = AbstractMCMC.sample(rng, model, sampler, n_MCMC; chain_type = MCMCChains.Chains,
                                param_names = param_names)
    return chain
end
