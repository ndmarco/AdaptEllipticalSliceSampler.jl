# Constrained Inference

As discussed in the main manuscript[^1], we show that the adaptive algorithm (AGESS) is ergodic 
for fairly general target distributions on $(\mathcal{X}, \mathcal{B}(\mathcal{X}))$, where 
$\mathcal{X}$ is an open subset of $\mathbb{R}^P$. This allows us to consider using AGESS in 
constrained inference settings that can posed as inference on an open set. Consider the following
constrained inference linear regression problem[^2]:

$$p(\boldsymbol{\beta} \mid \mathbf{y}) \propto \exp(-0.5 \lVert \mathbf{y} - \mathbf{X}\boldsymbol{\beta}\rVert^2_2) \mathbb{1}_{\{\boldsymbol{\beta}: \lVert \boldsymbol{\beta} \rVert^2_2 < 1\}}(\boldsymbol{\beta}).$$

Thus, we can see that the support is constrained to the interior of the unit ball -- an open set. 
We will start by generating data from an unconstrained linear regression framework, with $\boldsymbol{\beta}$
outside the set of interest.

```@example Constrained
import Random
using AdaptEllipticalSliceSampler
using Distributions
using Plots
using LinearAlgebra

Random.seed!(123)

D = 2
N = 100

### Generate Design matrix
X = randn(N, D)
### True beta outside of set of interest
β = [-1, 1]
### Generate Response Variable
y = X * β .+ randn(N)
```

Now that we have generated our data, let's construct a function that evaluates the posterior
log pdf (up to a constant). **Note: This function can be implemented more efficiently, however
this implementation is sufficient for this demonstration.**

```@example Constrained
function log_posterior(β::AbstractVector{Y}, X::AbstractMatrix{Y}, 
                       y::AbstractVector{Y}) where {Y <: AbstractFloat}
    l_lpdf::Float64 = -Inf
    ### Incorporate constraint
    if norm(β) < 1
        l_lpdf = -0.5 * norm(y - X * β)^2
    end

    return l_lpdf
end
```

With this function, we can use the `AGESS` function to draw samples from the posterior.

```@example Constrained
### Specify the dimension of the target distribution
P = D
### Specify the number of MCMC iterations
n_MCMC = 5000

### Run AGESS
results = AGESS(β -> log_posterior(β, X, y), n_MCMC, P)
```

Let's visualize the samples from the target distribution.

```@example Constrained
function circleShape()
    θ = LinRange(0, 2*π, 100)
    x = cos.(θ)
    y = sin.(θ)
    return x, y
end

plot(circleShape(), ylims=(-1.5, 1.5), xlims=(-1.5, 1.5), legend=false)

scatter!([β[1]], [β[2]], color="red")
scatter!(results.value[1000:end,1,:], results.value[1000:end,2,:], alpha = 0.4)
```

In the figure above, we can see that the red point is the true unconstrained $\boldsymbol{\beta}$,
from which the data was generated from. In green, we can see samples generated from our Markov chain.
We can see that AGESS provides an alternative method to conduct constrained Bayesian inference --
targeting the exact target distribution, rather than smooth relaxations of the sharply constrained
priors [^2] [^3] [^4].


[^1]: N. Marco and S. T. Tokdar. Adaptive generalized elliptical slice sampling. arXiv preprint arXiv:2605.21659, 2026.

[^2]: Rick Presman, Jason Xu. Distance-to-set priors and constrained bayesian inference. Proceedings of The 26th International Conference on Artificial Intelligence and Statistics, PMLR 206:2310-2326, 2023.

[^3]: L. L. Duan, A. L. Young, A. Nishimura, and D. B. Dunson. Bayesian constraint relaxation. Biometrika, 107(1):191–204, 2020.

[^4]: X. Zhou, Q. Heng, E. C. Chi, and H. Zhou. Proximal mcmc for bayesian inference of constrained and regularized estimation. The American Statistician, 78(4):379–390, 2024.