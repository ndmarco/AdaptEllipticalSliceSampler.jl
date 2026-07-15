"""
    bundle_samples(samples, model, sampler, state, ::Type{MCMCChains.Chains}; param_names, kwargs...)

Bundles a vector of `AGESSTransition` into an `MCMCChains.Chains` object, for use with
`AbstractMCMC.sample(model, sampler, n_MCMC; chain_type = MCMCChains.Chains)`. This lets AGESS
chains plug directly into the wider Turing ecosystem (MCMCDiagnosticTools, StatsPlots, etc.).

# Keyword Arguments
- `param_names::Union{AbstractVector{Symbol}, Missing} = missing`: names for each dimension of the target distribution. Defaults to `:param_1, :param_2, ...`.
"""
function AbstractMCMC.bundle_samples(
    samples::Vector{<:AGESSTransition},
    model::AbstractMCMC.AbstractModel,
    sampler::AGESSSampler,
    state,
    ::Type{MCMCChains.Chains};
    param_names = missing,
    stats = missing,
    kwargs...,
)
    P = length(samples[1].x)
    names = param_names === missing ? [Symbol("param_", i) for i in 1:P] : collect(param_names)
    @argcheck length(names) == P "param_names must have length $(P), got $(length(names))"

    arr = Array{Float64}(undef, length(samples), P + 1, 1)
    for (i, s) in enumerate(samples)
        arr[i, 1:P, 1] .= s.x
        arr[i, P + 1, 1] = s.lpdf
    end

    info = stats === missing ? NamedTuple() : (start_time = stats.start, stop_time = stats.stop)

    return MCMCChains.Chains(arr, vcat(names, :lp), (internals = [:lp],); info = info)
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
