module AdaptEllipticalSliceSamplerDynamicPPLExt

using AdaptEllipticalSliceSampler: AdaptEllipticalSliceSampler, AGESSSampler, AGESSTransition, AGESSBlock
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

Deliberately uses a FRESH, UNLINKED `VarInfo` here rather than `_get_ldf(model)` (which is
linked): the chain's stored samples are always unlinked/natural-scale (see
`_unlink_vector_and_logjoint`), but the *linked* dimension of a variable is not always equal to
its *natural* dimension — e.g. a `Dirichlet(K)`'s simplex bijector maps its K-dimensional
natural value to a (K-1)-dimensional unconstrained one. Building names from the linked range
would then be one short per such variable and misalign every subsequent column against
`_build_chains`' natural-scale array.
"""
function _default_param_names_and_varname_map(model::DynamicPPL.Model)
    ldf_natural = DynamicPPL.LogDensityFunction(model, DynamicPPL.getlogjoint, DynamicPPL.VarInfo(model))
    ranges_and_transforms = DynamicPPL.get_all_ranges_and_transforms(ldf_natural)

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

Also recomputes the log density on natural scale (`DynamicPPL.getlogjoint`, no Jacobian term)
rather than reusing the transition's original `lpdf`, which was accumulated internally via
`getlogjoint_internal` (= logprior + loglikelihood - logjacobian) since that's what AGESS
walks in linked space. Reusing it directly would make the `:lp` column silently include a
per-sample Jacobian correction that Turing's own samplers' `:lp`/`:logjoint` do not, making
the two incomparable for any model with a constrained (linked) variable.
"""
function _unlink_vector_and_logjoint(model::DynamicPPL.Model, x_linked::AbstractVector)
    ldf_linked = _get_ldf(model)
    init_strategy = DynamicPPL.InitFromVector(x_linked, ldf_linked)
    oavi = DynamicPPL.OnlyAccsVarInfo(DynamicPPL.VectorValueAccumulator())
    _, oavi_natural = DynamicPPL.init!!(model, oavi, init_strategy, DynamicPPL.UnlinkAll())
    ldf_natural = DynamicPPL.LogDensityFunction(model, DynamicPPL.getlogjoint, oavi_natural)
    x_natural = DynamicPPL.get_sample_input_vector(ldf_natural)
    lpdf_natural = LogDensityProblems.logdensity(ldf_natural, x_natural)
    return x_natural, lpdf_natural
end

"""
Bundles a `DynamicPPL.Model` run into an `MCMCChains.Chains` object with natural-scale values
and (for scalar variables) `VarName`-indexable metadata — unlinking each `AGESSTransition`'s
vector before handing off to the shared `_build_chains` logic. `param_names`/`varname_to_symbol`
are auto-derived from the model unless the caller supplies their own `param_names`, in which
case we can't safely correlate custom names back to `VarName`s, so that metadata is skipped.

The chain's `:lp` column holds the natural-scale log joint (`DynamicPPL.getlogjoint`), not the
linked-space value AGESS sampled with internally.
"""
function AbstractMCMC.bundle_samples(
    samples::Vector{<:AGESSTransition},
    model::DynamicPPL.Model,
    sampler::AGESSSampler,
    state,
    ::Type{MCMCChains.Chains};
    param_names = nothing,
    stats = missing,
    save_state::Bool = true,
    thinning::Integer = 1,
    kwargs...,
)
    unlinked_samples = [AGESSTransition(_unlink_vector_and_logjoint(model, s.x)...) for s in samples]

    ## `state` is stashed as-is (not unlinked): its `x_current` is in the same linked
    ## representation AGESS sampled with internally, which is exactly what `step` expects back
    ## on resume -- only the *returned samples* need unlinking, for display.
    if param_names === nothing
        names, varname_to_symbol = _default_param_names_and_varname_map(model)
        return AdaptEllipticalSliceSampler._build_chains(unlinked_samples, names, stats, varname_to_symbol, state, save_state, thinning)
    else
        return AdaptEllipticalSliceSampler._build_chains(unlinked_samples, param_names, stats, missing, state, save_state, thinning)
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

"""
    AGESSBlock(indices, submodel_constructor, context_indices)

Note: `context_indices` has no default value here (even though `Int[]`, i.e. "no context
needed", is a common choice) because giving it one would generate a 2-argument
`AGESSBlock(indices, submodel_constructor)` method that is ambiguous with `AGESSBlock`'s
base 2-argument `(indices, conditional)` constructor -- both would apply to a call with a
concrete `Function` second argument, and neither signature is more specific than the other.

Convenience constructor for a `DynamicPPL`-backed block (see `AGESSSampler`'s `blocks`
keyword): builds an `AGESSBlock` whose `conditional` evaluates a smaller, block-local Turing
submodel instead of the full model's `log_posterior` -- useful when `indices` is a
group/block whose likelihood and prior terms depend only on a handful of shared "context"
parameters (`context_indices`) plus its own local variables.

# Arguments
- `indices::AbstractVector{<:Integer}`: this block's coordinates in the full model's
  parameter vector, in the same order the submodel returned by `submodel_constructor`
  samples them (e.g. `[3]` if this block is the third parameter and the submodel has one
  variable; get this from `DynamicPPL.get_all_ranges_and_transforms` on the full model's
  `LogDensityFunction` if unsure).
- `submodel_constructor::Function`: `context_vals::Vector{Float64} -> DynamicPPL.Model`.
  Given the current values of the parameters at `context_indices`, returns a
  `DynamicPPL.Model` whose free (`~`-sampled) variables are exactly this block's `indices`.
  **`context_vals` are in AGESS's internal *linked* (unconstrained) scale**, i.e. exactly
  the values found in `x` during sampling -- not the natural/constrained scale of the
  parameters as your model would otherwise see them. If a context parameter has a
  constrained prior (e.g. `τ ~ InverseGamma(...)`), you must unconstrain/constrain it
  yourself inside `submodel_constructor` (e.g. `τ = exp(context_vals[2])`) to match
  whatever `DynamicPPL.link`'s transform for that variable's distribution actually is.
- `context_indices::AbstractVector{<:Integer}`: indices (in the full model's parameter
  vector) of the shared/global parameters `submodel_constructor` depends on. Pass `Int[]` if
  the submodel has no external dependencies.

The returned `conditional` caches the submodel's `LogDensityFunction` and only rebuilds it
when `context_vals` actually changes between calls (compared by value, since
`submodel_constructor` returns a fresh `DynamicPPL.Model` object every call and so can't be
cached by object identity the way the full model is in `_get_ldf`). Within a single block
visit -- which runs one scalar shrink loop per coordinate in the block, all with the same
context -- and across consecutive visits where the context hasn't moved, the
`LogDensityFunction` setup cost is paid once, not on every evaluation.
```
"""
function AdaptEllipticalSliceSampler.AGESSBlock(
    indices::AbstractVector{<:Integer},
    submodel_constructor::Function,
    context_indices::AbstractVector{<:Integer},
)
    idx = collect(Int, indices)
    ctx_idx = collect(Int, context_indices)
    last_ctx = Ref{Union{Nothing, Vector{Float64}}}(nothing)
    cached_ldf = Ref{Any}(nothing)

    conditional = function (x::AbstractVector)
        ctx = Float64.(view(x, ctx_idx))
        if last_ctx[] === nothing || ctx != last_ctx[]
            submodel = submodel_constructor(ctx)
            cached_ldf[] = DynamicPPL.LogDensityFunction(submodel, DynamicPPL.getlogjoint_internal, DynamicPPL.LinkAll())
            last_ctx[] = ctx
        end
        return LogDensityProblems.logdensity(cached_ldf[], x[idx])
    end

    return AGESSBlock(idx; conditional = conditional)
end

end
