module AdaptEllipticalSliceSamplerDynamicPPLExt

using AdaptEllipticalSliceSampler: AdaptEllipticalSliceSampler, AGESSSampler, AGESSTransition
using DynamicPPL: DynamicPPL
using LogDensityProblems: LogDensityProblems
using AbstractMCMC: AbstractMCMC
using MCMCChains: MCMCChains
using Random: Random

### Functions to help users use Turing.jl type models with the AGESS framework

## optional dependence on Turing.jl allows this functionality if users have installed Turing.jl
## but is the package can still be used without the large dependencies

"""
Caches the `LogDensityFunction` built for a given `DynamicPPL.Model`, keyed by object
identity, so that `AGESSSampler`/`AGESS` can accept a raw Turing model directly without
rebuilding (and re-evaluating the model to determine variable ranges/transforms) on every
single log-posterior evaluation.
"""
const _LDF_CACHE = IdDict{DynamicPPL.Model, DynamicPPL.LogDensityFunction}()

function _get_ldf(model::DynamicPPL.Model)
    return get!(_LDF_CACHE, model) do
        DynamicPPL.LogDensityFunction(model, DynamicPPL.getlogjoint_internal, DynamicPPL.LinkAll())
    end
end

function AdaptEllipticalSliceSampler._logdensity(model::DynamicPPL.Model, x::AbstractVector)
    return LogDensityProblems.logdensity(_get_ldf(model), x)
end

function AdaptEllipticalSliceSampler._dimension(model::DynamicPPL.Model)
    return LogDensityProblems.dimension(_get_ldf(model))
end

"""
Builds default `param_names` (bare Turing variable names) and a `varname_to_symbol` map
(`VarName => column Symbol`) for a `DynamicPPL.Model`, for `chain.info` — this is what makes
`chain[@varname(...)]`-style access (via DynamicPPL's internal helpers), `DynamicPPL.returned`,
`predict`, etc. work on the resulting chain. Only scalar (single-element) variables get a
`varname_to_symbol` entry: constructing the correct *leaf* `VarName` for a multi-element
variable (e.g. `x[1]`) requires AbstractPPL's optic/lens API, which has changed shape across
versions in ways that make it too unstable to rely on here — those variables still get
`Symbol`-suffixed columns (`x[1]`, `x[2]`, ...), just not `VarName`-indexable ones.
"""
function _default_param_names_and_varname_map(model::DynamicPPL.Model)
    ranges_and_transforms = DynamicPPL.get_all_ranges_and_transforms(_get_ldf(model))

    names = Symbol[]
    varname_to_symbol = Dict{DynamicPPL.VarName, Symbol}()
    for (vn, rt) in pairs(ranges_and_transforms)
        r = rt.range
        base = string(vn)
        if length(r) == 1
            sym = Symbol(base)
            push!(names, sym)
            varname_to_symbol[vn] = sym
        else
            for k in 1:length(r)
                push!(names, Symbol(base, "[", k, "]"))
            end
        end
    end
    return names, varname_to_symbol
end

"""
Maps a sample's flat *linked* (unconstrained) vector back to its natural scale, using
`DynamicPPL.InitFromVector` + `UnlinkAll()` — the same machinery Turing itself uses so that
user-facing chains always hold natural-scale values, even though `AGESSSampler` walked
unconstrained space internally (e.g. `log(s2)`, not `s2`).
"""
function _unlink_vector(model::DynamicPPL.Model, x_linked::AbstractVector)
    ldf_linked = _get_ldf(model)
    init_strategy = DynamicPPL.InitFromVector(x_linked, ldf_linked)
    oavi = DynamicPPL.OnlyAccsVarInfo(DynamicPPL.VectorValueAccumulator())
    _, oavi_natural = DynamicPPL.init!!(model, oavi, init_strategy, DynamicPPL.UnlinkAll())
    ldf_natural = DynamicPPL.LogDensityFunction(model, DynamicPPL.getlogjoint, oavi_natural)
    return DynamicPPL.get_sample_input_vector(ldf_natural)
end

"""
Bundles a `DynamicPPL.Model` run into an `MCMCChains.Chains` object with natural-scale values
and (for scalar variables) `VarName`-indexable metadata — unlinking each `AGESSTransition`'s
vector before handing off to the shared `_build_chains` logic. `param_names`/`varname_to_symbol`
are auto-derived from the model unless the caller supplies their own `param_names`, in which
case we can't safely correlate custom names back to `VarName`s, so that metadata is skipped.
"""
function AbstractMCMC.bundle_samples(
    samples::Vector{<:AGESSTransition},
    model::DynamicPPL.Model,
    sampler::AGESSSampler,
    state,
    ::Type{MCMCChains.Chains};
    param_names = nothing,
    stats = missing,
    kwargs...,
)
    unlinked_samples = [AGESSTransition(_unlink_vector(model, s.x), s.lpdf) for s in samples]

    if param_names === nothing
        names, varname_to_symbol = _default_param_names_and_varname_map(model)
        return AdaptEllipticalSliceSampler._build_chains(unlinked_samples, names, stats, varname_to_symbol)
    else
        return AdaptEllipticalSliceSampler._build_chains(unlinked_samples, param_names, stats, missing)
    end
end

# Turing itself defines `sample(rng, model::DynamicPPL.Model, spl::AbstractSampler, N; ...)`
# (with its own `chain_type` default of `VNChain`, which doesn't apply to AGESSSampler since
# our transitions are plain vectors, not VarName-tagged). That method and our own
# `sample(rng, model::AbstractMCMC.AbstractModel, sampler::AGESSSampler, N; ...)` override
# (see MCMCChains_interface.jl) are equally specific for this combination, so calling
# `sample(rng, turing_model, AGESSSampler(...), N)` is ambiguous unless we add a strictly
# more specific method here to resolve it in favor of our own `MCMCChains.Chains` default.
function AbstractMCMC.sample(
    rng::Random.AbstractRNG,
    model::DynamicPPL.Model,
    sampler::AGESSSampler,
    N::Integer;
    chain_type::Type = MCMCChains.Chains,
    kwargs...,
)
    return AbstractMCMC.mcmcsample(rng, model, sampler, N; chain_type = chain_type, kwargs...)
end

# Overloading the function to allow the user to use the default rng
function AbstractMCMC.sample(
    model::DynamicPPL.Model,
    sampler::AGESSSampler,
    N::Integer;
    kwargs...,
)
    return AbstractMCMC.sample(Random.default_rng(), model, sampler, N; kwargs...)
end

# Turing also defines the initial `step(rng, model::DynamicPPL.Model, spl::AbstractSampler;
# initial_params, ...)`, which sets up a `VarInfo` and dispatches to a per-sampler
# `initialstep` that AGESSSampler doesn't (and shouldn't) implement, since its initial state
# already comes from `sampler.init_x` rather than Turing's own initialization strategies.
# Same ambiguity as `sample` above; resolve it the same way, delegating to the shared
# `_initial_step` helper instead of duplicating its body.
function AbstractMCMC.step(rng::Random.AbstractRNG, model::DynamicPPL.Model, sampler::AGESSSampler; kwargs...)
    return AdaptEllipticalSliceSampler._initial_step(rng, model, sampler)
end

end
