"""
    AGESSModel(log_posterior, P)

A lightweight `AbstractMCMC.AbstractModel` wrapping a log-posterior function directly, for users
who do not want to depend on the `LogDensityProblems` interface. Also supported: `AbstractMCMC.LogDensityModel`
wrapping any `LogDensityProblems`-compliant object, for interoperability with the wider Turing ecosystem.

# Arguments
- `log_posterior::Function`: a function evaluating the log posterior pdf with a vector of parameters as the only input
- `P::Integer`: the dimension of the target distribution
"""
struct AGESSModel{F<:Function, T<:Integer} <: AbstractMCMC.AbstractModel
    log_posterior::F
    P::T
end

_logdensity(model::AGESSModel, x::AbstractVector) = model.log_posterior(x)
_dimension(model::AGESSModel) = model.P

_logdensity(model::AbstractMCMC.LogDensityModel, x::AbstractVector) = LogDensityProblems.logdensity(model.logdensity, x)
_dimension(model::AbstractMCMC.LogDensityModel) = LogDensityProblems.dimension(model.logdensity)
struct AGESSSampler{Y<:AbstractFloat, V<:AbstractVector{Y}, M<:AbstractMatrix{Y}} <: AbstractMCMC.AbstractSampler
    μ_0::V
    Σ_0::M
    Σ_0_chol::Cholesky{Y, M}
    init_x::V
    P::Int
    n_MCMC::Int
    ν::Y
    burnin::Y
    ϵ::Y
    single_step_prop::Y
    β::Y
    w_const::Y
    t_dist::Bool
end

"""
    AGESSSampler(P, n_MCMC; μ_0, Σ_0, init_x, t_dist, ν, burnin, ϵ, single_step_prop, β)

An `AbstractMCMC.AbstractSampler` implementing adaptive generalized elliptical slice sampling (AGESS).

# Arguments
- `P::Integer`: the dimension of the target distribution
- `n_MCMC::Integer`: the total number of iterations the sampler is planned to run for (used to schedule burn-in)

# Keyword Arguments
- `μ_0::Union{AbstractVector{<:AbstractFloat}, AbstractFloat} = 0.0`: a vector (or number which will be multiplied by the one vector) containing the initial (or prior) mean of adaptive distribution
- `Σ_0::Union{AbstractMatrix{<:AbstractFloat}, AbstractFloat} = 1.0`: a matrix (or number which will be multiplied by I) containing the initial (or prior) scale of adaptive distribution
- `init_x::Union{AbstractVector{<:AbstractFloat}, AbstractFloat} = 0.0`: a vector (or number which will be multiplied by the one vector) containing the initial starting location of the Markov chain
- `t_dist::Bool = true`: a Boolean variable indicating whether to use a t-distribution (true) or normal distribution (false) for elliptical slice sampling (Note: should almost always use t-distribution)
- `ν::AbstractFloat = 6.0`: the degrees of freedom of the t-distribution
- `burnin::AbstractFloat = 0.5`: the proportion of chain used for burnin
- `ϵ::AbstractFloat = 0.05`: the proportion of non-adaptive transitions
- `single_step_prop::AbstractFloat = 0.05`: the proportion of transitions where we perform one-dimensional updates (P >= 10)
- `β::AbstractFloat = 0.5`: the rate at which the adaptation diminishes

# References
N. Marco and S. T. Tokdar. Adaptive generalized elliptical slice sampling. arXiv preprint arXiv:2605.21659, 2026.
"""
function AGESSSampler(P::Integer, n_MCMC::Integer;
                      μ_0::Union{AbstractVector{Y}, Y} = 0.0, Σ_0::Union{AbstractMatrix{Y}, Y} = 1.0,
                      init_x::Union{AbstractVector{Y}, Y} = 0.0, t_dist::Bool = true, ν::Y = 6.0, burnin::Y = 0.5,
                      ϵ::Y = 0.05, single_step_prop::Y = 0.05, β::Y = 0.5) where {Y<:AbstractFloat}

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

    @argcheck length(μ_0) == P

    ## Check rest of arguments
    @argcheck P >= 1
    @argcheck n_MCMC >= 1
    @argcheck burnin >= 0.0
    @argcheck burnin < 1.0
    @argcheck ϵ >= 0.0
    @argcheck ϵ < 1.0
    @argcheck single_step_prop >= 0.0
    @argcheck single_step_prop < 1.0
    @argcheck single_step_prop + ϵ <= 1.0

    Σ_0_chol = cholesky(Σ_0)
    w_const = max(2/3, ((cbrt(P) - 1) / cbrt(P)))

    return AGESSSampler(μ_0, Σ_0, Σ_0_chol, init_x, Int(P), Int(n_MCMC), ν, burnin, ϵ,
                        single_step_prop, β, w_const, t_dist)
end


function AGESSSampler(model::AbstractMCMC.AbstractModel, n_MCMC::Integer;
                      μ_0::Union{AbstractVector{Y}, Y} = 0.0, Σ_0::Union{AbstractMatrix{Y}, Y} = 1.0,
                      init_x::Union{AbstractVector{Y}, Y} = 0.0, t_dist::Bool = true, ν::Y = 6.0, burnin::Y = 0.5,
                      ϵ::Y = 0.05, single_step_prop::Y = 0.05, β::Y = 0.5) where {Y<:AbstractFloat}

    P = _dimension(model)
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

    @argcheck length(μ_0) == P

    ## Check rest of arguments
    @argcheck P >= 1
    @argcheck n_MCMC >= 1
    @argcheck burnin >= 0.0
    @argcheck burnin < 1.0
    @argcheck ϵ >= 0.0
    @argcheck ϵ < 1.0
    @argcheck single_step_prop >= 0.0
    @argcheck single_step_prop < 1.0
    @argcheck single_step_prop + ϵ <= 1.0

    Σ_0_chol = cholesky(Σ_0)
    w_const = max(2/3, ((cbrt(P) - 1) / cbrt(P)))

    return AGESSSampler(μ_0, Σ_0, Σ_0_chol, init_x, Int(P), Int(n_MCMC), ν, burnin, ϵ,
                        single_step_prop, β, w_const, t_dist)
end

"""
    AGESSState

Holds the mutable state of an AGESS Markov chain between calls to `AbstractMCMC.step`.
"""
mutable struct AGESSState{Y<:AbstractFloat, V<:AbstractVector{Y}, M<:AbstractMatrix{Y}}
    x_current::V             # current state of Markov chain
    x_next::V                # used for potential new state and will eventually be the new state
    lpdf_current::Y
    μ_adapt::V
    Σ_chol_adapt::Cholesky{Y, M}
    μ_adapt_ph::V
    Σ_chol_adapt_ph::Cholesky{Y, M}
    ph_AGESS::V
    z::V
    perm::Vector{Int}
    ph_cholesky_update::V
    iteration::Int
    n_j::Int
    N_J::Int
end

"""
    AGESSTransition

The sample returned by `AbstractMCMC.step` for AGESS: the state of the Markov chain and its log
posterior density.
"""
struct AGESSTransition{V<:AbstractVector{<:AbstractFloat}, Y<:AbstractFloat}
    x::V
    lpdf::Y
end

### Initial step
function _initial_step(rng::Random.AbstractRNG, model::AbstractMCMC.AbstractModel, sampler::AGESSSampler; kwargs...)
    @argcheck _dimension(model) == sampler.P "Sampler was constructed for dimension $(sampler.P) but model has dimension $(_dimension(model))"

    x_current = deepcopy(sampler.init_x)
    lpdf_current = _logdensity(model, x_current)
    @argcheck isfinite(lpdf_current) "Initial starting position of Markov chain must have finite posterior density"

    state = AGESSState(
        x_current,                  # x_current
        deepcopy(x_current),        # x_next
        lpdf_current,               # lpdf_current
        deepcopy(sampler.μ_0),      # μ_adapt
        cholesky(sampler.Σ_0),      # Σ_chol_adapt
        deepcopy(sampler.μ_0),      # μ_adapt_ph
        cholesky(sampler.Σ_0),      # Σ_chol_adapt_ph
        similar(x_current),         # ph_AGESS
        similar(x_current),         # z
        randperm(rng, sampler.P),   # perm
        similar(x_current),         # ph_cholesky_update
        1,                          # iteration
        2,                          # n_j
        2,                          # N_j
    )

    return AGESSTransition(copy(x_current), lpdf_current), state
end

function AbstractMCMC.step(rng::Random.AbstractRNG, model::AbstractMCMC.AbstractModel, sampler::AGESSSampler; kwargs...)
    @argcheck _dimension(model) == sampler.P "Sampler was constructed for dimension $(sampler.P) but model has dimension $(_dimension(model))"

    x_current = deepcopy(sampler.init_x)
    lpdf_current = _logdensity(model, x_current)
    @argcheck isfinite(lpdf_current) "Initial starting position of Markov chain must have finite posterior density"

    state = AGESSState(
        x_current,                  # x_current
        deepcopy(x_current),        # x_next
        lpdf_current,               # lpdf_current
        deepcopy(sampler.μ_0),      # μ_adapt
        cholesky(sampler.Σ_0),      # Σ_chol_adapt
        deepcopy(sampler.μ_0),      # μ_adapt_ph
        cholesky(sampler.Σ_0),      # Σ_chol_adapt_ph
        similar(x_current),         # ph_AGESS
        similar(x_current),         # z
        randperm(rng, sampler.P),   # perm
        similar(x_current),         # ph_cholesky_update
        1,                          # iteration
        2,                          # n_j
        2,                          # N_j
    )

    return AGESSTransition(copy(x_current), lpdf_current), state
end

function AbstractMCMC.step(rng::Random.AbstractRNG, model::AbstractMCMC.AbstractModel, sampler::AGESSSampler,
                           state::AGESSState; kwargs...)

    i = state.iteration + 1
    P = sampler.P
    burnin_num = floor(Int, sampler.burnin * sampler.n_MCMC)
    log_posterior(x) = _logdensity(model, x)

    if P >= 10
        if i < (burnin_num * sampler.single_step_prop)
            # if high-dimensional: conduct 1-d updates for faster convergence at the beginning of the chain
            state.lpdf_current = AGESS_single_step_1d!(state.x_current, state.x_next, log_posterior, sampler.t_dist,
                                                        sampler.ν, state.μ_adapt, state.Σ_chol_adapt.L,
                                                        state.lpdf_current, state.perm; rng = rng)
        else
            # Conduct transition using adaptive kernel
            if rand(rng) > (sampler.ϵ + sampler.single_step_prop)
                state.lpdf_current = AGESS_single_step!(state.x_current, state.x_next, state.z, log_posterior, sampler.t_dist,
                                                         sampler.ν, P, state.ph_AGESS, state.μ_adapt, state.Σ_chol_adapt.L,
                                                         state.lpdf_current; rng = rng)
            # Conduct transition using 1-d update
            elseif rand(rng) < (sampler.single_step_prop / (sampler.ϵ + sampler.single_step_prop))
                state.lpdf_current = AGESS_single_step_1d!(state.x_current, state.x_next, log_posterior, sampler.t_dist,
                                                            sampler.ν, state.μ_adapt, state.Σ_chol_adapt.L,
                                                            state.lpdf_current, state.perm; rng = rng)
            # Conduct transition using non-adaptive kernel (standard GESS)
            else
                state.lpdf_current = AGESS_single_step!(state.x_current, state.x_next, state.z, log_posterior, sampler.t_dist,
                                                         sampler.ν, P, state.ph_AGESS, sampler.μ_0, sampler.Σ_0_chol.L,
                                                         state.lpdf_current; rng = rng)
            end
        end
    else
        ## In low dimensions
        ## Conduct transition using adaptive kernel
        if rand(rng) > sampler.ϵ
            state.lpdf_current = AGESS_single_step!(state.x_current, state.x_next, state.z, log_posterior, sampler.t_dist,
                                                     sampler.ν, P, state.ph_AGESS, state.μ_adapt, state.Σ_chol_adapt.L,
                                                     state.lpdf_current; rng = rng)
        ## COnduct transition using non-adaptive kernel (standard )
        else
            state.lpdf_current = AGESS_single_step!(state.x_current, state.x_next, state.z, log_posterior, sampler.t_dist,
                                                     sampler.ν, P, state.ph_AGESS, sampler.μ_0, sampler.Σ_0_chol.L,
                                                     state.lpdf_current; rng = rng)
        end
    end

    state.x_current, state.x_next = state.x_next, state.x_current

    ## Background (diminishing) adaptation of mean and covariance
    w_i = i^(-sampler.w_const)
    state.Σ_chol_adapt_ph.U .= sqrt(1 - w_i) .* state.Σ_chol_adapt_ph.U
    state.ph_cholesky_update .= sqrt(w_i) .* (state.x_current .- state.μ_adapt_ph)
    lowrankupdate!(state.Σ_chol_adapt_ph, state.ph_cholesky_update)
    state.μ_adapt_ph .= (1 - w_i) .* state.μ_adapt_ph .+ w_i .* state.x_current

    ## Update according to AirMCMC
    if i == state.N_J
        state.Σ_chol_adapt.U .= state.Σ_chol_adapt_ph.U
        state.μ_adapt .= state.μ_adapt_ph
        state.n_j += 1
        state.N_J += floor(Int, state.n_j ^ sampler.β)
    end

    state.iteration = i

    return AGESSTransition(copy(state.x_current), state.lpdf_current), state
end
