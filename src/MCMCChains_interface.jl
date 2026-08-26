"""
    _build_chains(samples, param_names, stats, varname_to_symbol, state, save_state, thinning)

Shared array/`Chains`-construction logic used by both the generic `bundle_samples` below and
the `DynamicPPL.Model`-specific override (which needs to unlink each sample's vector to natural
scale first, but otherwise builds the `Chains` object the same way). If `save_state` is true,
`state` (the final `AGESSState` of the Markov chain) is stashed in `chain.info.samplerstate`,
retrievable via `AGESS_loadstate` -- see its docstring for how to resume sampling from it.

The returned `Chains`' iteration numbering starts at `state.iteration - thinning*(n-1)` (where
`n = length(samples)`) rather than always at 1: `state.iteration` is the *cumulative* raw-step
count of the Markov chain (carried across `AGESS_loadstate`-based resumes, unlike a fresh
`AGESSState` which starts at 1), so this makes a resumed run's chain continue the previous
chain's numbering instead of colliding with it -- which is what `vcat(chain1, chain2)` (to
combine a chain with its continuation) requires, since `MCMCChains.Chains` demands strictly
increasing iteration numbers across a concatenation.
"""
function _build_chains(samples::Vector{<:AGESSTransition}, param_names, stats, varname_to_symbol,
                       state, save_state::Bool, thinning::Integer)
    P = length(samples[1].x)
    names = param_names === missing ? [Symbol("param_", i) for i in 1:P] : collect(param_names)
    @argcheck length(names) == P "param_names must have length $(P), got $(length(names))"

    n = length(samples)
    arr = Array{Float64}(undef, n, P + 1, 1)
    for (i, s) in enumerate(samples)
        arr[i, 1:P, 1] .= s.x
        arr[i, P + 1, 1] = s.lpdf
    end

    info = NamedTuple()
    if stats !== missing
        info = merge(info, (start_time = stats.start, stop_time = stats.stop))
    end
    if varname_to_symbol !== missing
        info = merge(info, (varname_to_symbol = varname_to_symbol,))
    end
    if save_state
        info = merge(info, (samplerstate = state,))
    end

    start = state.iteration - thinning * (n - 1)
    return MCMCChains.Chains(arr, vcat(names, :lp), (internals = [:lp],);
                             start = start, thin = Int(thinning), info = info)
end

"""
    bundle_samples(samples, model, sampler, state, ::Type{MCMCChains.Chains}; param_names, save_state, kwargs...)

Bundles a vector of `AGESSTransition` into an `MCMCChains.Chains` object, for use with
`AbstractMCMC.sample(model, sampler, n_MCMC; chain_type = MCMCChains.Chains)`. This lets AGESS
chains plug directly into the wider Turing ecosystem (MCMCDiagnosticTools, StatsPlots, etc.).

# Keyword Arguments
- `param_names::Union{AbstractVector{Symbol}, Missing} = missing`: names for each dimension of the target distribution. Defaults to `:param_1, :param_2, ...`.
- `save_state::Bool = true`: if `true` (the default), stash the chain's final `AGESSState`
  (position, adaptive covariance, iteration count -- everything needed to resume) in
  `chain.info.samplerstate`. The cost is O(P^2) (dominated by the two adaptive-covariance
  Cholesky factors), paid once at the end of sampling, not per iteration -- negligible next to
  the O(N*P) sample array for any chain where N isn't tiny relative to P. Use
  `AGESS_loadstate(chain)` to retrieve it and `sample(model, sampler, n_more; initial_state = ...)`
  to continue the same Markov chain rather than starting a fresh one. Pass `false` to opt out.
"""
function AbstractMCMC.bundle_samples(
    samples::Vector{<:AGESSTransition},
    model::AbstractMCMC.AbstractModel,
    sampler::AGESSSampler,
    state,
    ::Type{MCMCChains.Chains};
    param_names = missing,
    stats = missing,
    varname_to_symbol = missing,
    save_state::Bool = true,
    thinning::Integer = 1,
    kwargs...,
)
    return _build_chains(samples, param_names, stats, varname_to_symbol, state, save_state, thinning)
end

"""
    AGESS_loadstate(chain::MCMCChains.Chains)

Extracts the final `AGESSState` stashed in `chain` by a previous `sample(...; save_state=true)`
call. Pass the result as `initial_state` to resume sampling exactly where `chain` left off (same
position, same adaptive covariance, same iteration count) instead of starting a fresh chain from
`sampler.init_x`:

```julia
chain1 = sample(model, sampler, n1; save_state = true)
state = AGESS_loadstate(chain1)
chain2 = sample(model, sampler, n2; initial_state = state)  # continues chain1
```

The returned state is marked so that the "beginning of chain" 1-d/block update phase (see
`AGESSSampler`'s `single_step_prop`/`burnin`) never re-triggers on the resumed run, regardless
of the `sampler`'s own `n_MCMC`/`burnin` settings -- once a chain has been resumed, sampling
proceeds straight into the mature adaptive-kernel mixture, exactly as if the phase had already
finished (which, from the chain's perspective, it has).

Throws an `ArgumentError` if `chain` was not produced with `save_state=true`.
"""
function AGESS_loadstate(chain::MCMCChains.Chains)
    haskey(chain.info, :samplerstate) || throw(ArgumentError(
        "chain was not produced with `save_state=true`; no sampler state available to resume from"))
    state = chain.info.samplerstate
    state.single_step_phase_done = true
    return state
end

### Overload function to have MCMCChains.Chains the default chain type
function AbstractMCMC.sample(
    rng::Random.AbstractRNG,
    model::AbstractMCMC.AbstractModel,
    sampler::AGESSSampler,
    N_or_isdone;
    chain_type::Type = MCMCChains.Chains,
    kwargs...,
)
    return AbstractMCMC.mcmcsample(rng, model, sampler, N_or_isdone; chain_type = chain_type, kwargs...)
end

### Overload function to use default rng
function AbstractMCMC.sample(
    model::AbstractMCMC.AbstractModel,
    sampler::AGESSSampler,
    N_or_isdone;
    kwargs...,
)
    return AbstractMCMC.sample(Random.default_rng(), model, sampler, N_or_isdone; kwargs...)
end
