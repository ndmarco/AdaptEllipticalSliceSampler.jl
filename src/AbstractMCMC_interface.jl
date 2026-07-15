struct AGESSSampler{T<:AbstractFloat} <: AbstractMCMC.AbstractSampler
    ν::T                    # df for t-distribution
    burnin::T               # proportion of chain used for burnin
    ϵ::T                    # proportion of iterations that are non-adaptive
    single_step_prop::T     # proportion of iterations that are 1-d updates
    β::T                    # constant used in adaption scheme (AIRMCMC) - how often it updates
    w_const::T              # constant used in adaption schems (AIRMCMC) - how fast background adaptation diminishes
    t_dist::Bool            # whether or not to use a t-distribution for elliptical distribution
end


mutable struct AGESSState{V<:AbstractVector{AbstractFloat}}
    x_current::V            # current state of Markov chain
    x_proposal::V           # used for potential new state and will eventually be the new state
    lpdf_current::Float64
    μ_adapt::V
    Σ_chol_adapt::LowerTriangular{AbstractFloat, <:AbstractMatrix{AbstractFloat}}
    μ_adapt_ph::V
    Σ_chol_adapt_ph::LowerTriangular{AbstractFloat, <:AbstractMatrix{AbstractFloat}}
    ph_AGESS::V
    z::V
    perm::Vector{Int}
    ph_cholesky_update::V
    iteration::Int
    n_j::Int
    N_J::Int
end