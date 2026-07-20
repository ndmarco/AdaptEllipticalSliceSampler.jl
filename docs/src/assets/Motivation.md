```@example Motivation

@model function gdemo(x)
    s² ~ InverseGamma(2, 3)
    m ~ Normal(0, sqrt(s²))
    bumps = sin(m) + cos(m)
    m = m + 5 * bumps
    for i in eachindex(x)
        x[i] ~ Normal(m, sqrt(s²))
    end
    return s², m
end

# Define our data points.
x = [1.5, 2.0, 13.0, 2.1, 0.0]

# Set up the model call.
model = gdemo(x)

# Evaluate surface at coordinates.
evaluate(m1, m2) = logjoint(model, (m=m2, s²=exp(m1)))

function plot_base(chain; label="")
    # Extract values from chain.
    ss = log.(chain[:s²])
    ms = chain[:m]
    lps = chain[:lp]

    # How many surface points to sample.
    granularity = 10_000

    # Range start/stop points.
    spread = 0.5
    σ_start = minimum(ss) - spread * std(ss)
    σ_stop = maximum(ss) + spread * std(ss)
    μ_start = minimum(ms) - spread * std(ms)
    μ_stop = maximum(ms) + spread * std(ms)
    σ_rng = collect(range(σ_start; stop=σ_stop, length=granularity))
    μ_rng = collect(range(μ_start; stop=μ_stop, length=granularity))

    # Make surface plot.
    p = surface(
        σ_rng,
        μ_rng,
        evaluate;
        camera=(30, 65),
        colorbar=false,
        color=:inferno,
        title=label,
    )

    return p
end;

function plot_scatter_agess(chain, p; label="")
    # Extract values from chain.
    ss = log.(chain[:s²])
    ms = chain[:m]
    lps = chain[:lp]

    line_range = 1:length(ms)

    p1 = deepcopy(p)

    p1 = scatter3d!(
        ss[line_range],
        ms[line_range],
        lps[line_range];
        mc=:viridis,
        marker_z=collect(line_range),
        msw=0,
        legend=false,
        colorbar=false,
        alpha=0.5,
        xlabel="log(s²)",
        ylabel="m",
        zlabel="Log probability",
        title=label,
    )

    return p1
end;

n_MCMC = 10_000
sampler = AGESSSampler(model, n_MCMC)

chain = AbstractMCMC.sample(Xoshiro(468), model, sampler, n_MCMC) 
chain = chain[2501:end,:,:]
describe(chain)

p = plot_base(chain)
p = plot_scatter_agess(chain, p)

function plot_scatter(chain, p; label="")
    # Extract values from chain.
    ss = log.(chain[:s²])
    ms = chain[:m]
    lps = chain[:logjoint]

    line_range = 1:length(ms)

    p

    scatter3d!(
        ss[line_range],
        ms[line_range],
        lps[line_range];
        mc=:viridis,
        marker_z=collect(line_range),
        msw=0,
        legend=false,
        colorbar=false,
        alpha=0.5,
        xlabel="log(s²)",
        ylabel="m",
        zlabel="Log probability",
        title=label,
    )

    return p
end

c1 = sample(Xoshiro(468), model, PG(20) , 10_000; chain_type = MCMCChains.Chains)
c1 = c1[2501:10000]
describe(c1)
p1  = plot_base(chain)
p1 = plot_scatter(c1,p1)


c2 = sample(Xoshiro(468), model, MH(), 10_000; chain_type = MCMCChains.Chains)
c2 = c2[2501:10000]
describe(c2)
p2  = plot_base(chain)
p2 = plot_scatter(c2,p2)

c3 = sample(Xoshiro(468), model, NUTS(0.65), 10000; chain_type = MCMCChains.Chains)
c3 = c3[2501:10000]
describe(c3)
p3  = plot_base(chain)
p3 = plot_scatter(c3,p3)


plot(p, p1, p2, p3)
plot!(size = (2000, 1000))

```