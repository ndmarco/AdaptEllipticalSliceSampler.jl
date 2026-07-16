module AdaptEllipticalSliceSamplerDynamicPPLExt

using AdaptEllipticalSliceSampler: AdaptEllipticalSliceSampler, AGESSSampler
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
        DynamicPPL.LogDensityFunction(model, DynamicPPL.getlogjoint, DynamicPPL.LinkAll())
    end
end

function AdaptEllipticalSliceSampler._logdensity(model::DynamicPPL.Model, x::AbstractVector)
    return LogDensityProblems.logdensity(_get_ldf(model), x)
end

function AdaptEllipticalSliceSampler._dimension(model::DynamicPPL.Model)
    return LogDensityProblems.dimension(_get_ldf(model))
end

"""
Classifies the transform applied to a single scalar component by comparing its natural-scale
and linked-scale values behaviorally (rather than introspecting bijector types, which keeps
breaking across DynamicPPL versions): identity, `log` (positive support), `logit` (unit
interval), or an unrecognized transform.
"""
function _transform_prefix(natural::Real, linked::Real; atol = 1e-6, rtol = 1e-6)
    isapprox(natural, linked; atol = atol, rtol = rtol) && return ""
    natural > 0 && isapprox(linked, log(natural); atol = atol, rtol = rtol) && return "log_"
    if 0 < natural < 1
        isapprox(linked, log(natural / (1 - natural)); atol = atol, rtol = rtol) && return "logit_"
    end
    return "transformed_"
end



"""
Builds default `param_names` for a `DynamicPPL.Model`, prefixing each variable with `log_`,
`logit_`, or a generic `transformed_` marker when it's on the linked (unconstrained) scale, so
column names reflect what's actually stored rather than silently mislabeling e.g. `log(s2)` as
`s2`. Runs the model twice (once unlinked, once linked) with an identically-seeded RNG in an
isolated `Task`, so the two draws correspond to the same underlying values without touching the
user's own RNG state.
"""
function _default_param_names(model::DynamicPPL.Model)
    x_unlinked, x_linked, ranges_and_transforms = fetch(Threads.@spawn begin
        Random.seed!(0)
        ldf_u = DynamicPPL.LogDensityFunction(model, DynamicPPL.getlogjoint, DynamicPPL.UnlinkAll())
        xu = DynamicPPL.get_sample_input_vector(ldf_u)
        Random.seed!(0)
        ldf_l = DynamicPPL.LogDensityFunction(model, DynamicPPL.getlogjoint, DynamicPPL.LinkAll())
        xl = DynamicPPL.get_sample_input_vector(ldf_l)
        (xu, xl, DynamicPPL.get_all_ranges_and_transforms(ldf_u))
    end)

    names = Symbol[]
    for (vn, rt) in pairs(ranges_and_transforms)
        r = rt.range
        base = string(vn)
        prefix = _transform_prefix(x_unlinked[first(r)], x_linked[first(r)])
        if length(r) == 1
            push!(names, Symbol(prefix, base))
        else
            for k in 1:length(r)
                push!(names, Symbol(prefix, base, "[", k, "]"))
            end
        end
    end
    return names
end

# Call the sample function using a DynamicPPL (Turing) model; extracting out the parameter names
function AbstractMCMC.sample(
    rng::Random.AbstractRNG,
    model::DynamicPPL.Model,
    sampler::AGESSSampler,
    N::Integer;
    chain_type::Type = MCMCChains.Chains,
    param_names = _default_param_names(model),
    kwargs...,
)
    return AbstractMCMC.mcmcsample(rng, model, sampler, N; chain_type = chain_type, param_names = param_names, kwargs...)
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
