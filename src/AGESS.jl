struct AGESS_MCMC_params
    log_posterior::Function
    μ_0::AbstractVector{<:AbstractFloat}
    Σ_0::AbstractMatrix{<:AbstractFloat}
    t_dist::Bool
    ν::AbstractFloat
    β::AbstractFloat
    single_step_prop::AbstractFloat
    burnin::AbstractFloat
    ϵ::AbstractFloat
    n_MCMC::Integer
    P::Integer
end

struct MCMC_output
    samps::AbstractMatrix{<:AbstractFloat}
    l_pdf::AbstractVector{<:AbstractFloat}
    params::AGESS_MCMC_params
    adapted_Σ::AbstractMatrix{<:AbstractFloat}
    adapted_μ::AbstractVector{<:AbstractFloat}
end




function AGESS_single_step!(x::AbstractMatrix{Y}, z::AbstractVector{Y}, params::AGESS_MCMC_params, 
                            ph::AbstractVector{Y},μ_adapt::AbstractVector{Y}, 
                            Σ_chol_adapt::LowerTriangular{Y, <:AbstractMatrix{Y}}, i::T) where {Y<:AbstractFloat, T<:Integer}
    l_pdf::eltype(x) = 0.0
    y::eltype(x) = 0.0
    L_star::eltype(x) = 0.0
    ## Propose new z
    if params.t_dist == true
        @views cond_rMvT!(z, x[i,:], μ_adapt, Σ_chol_adapt, params.ν, ph, params.P)
    else
        z .= Σ_chol_adapt * randn(params.P) .+ μ_adapt
    end

    @views y = params.log_posterior(x[i,:]) + log(rand())
    if params.t_dist == true
        @views y -= dMvT(x[i,:], μ_adapt, Σ_chol_adapt, ph, params.ν, params.P)
    else
        @views y -= dMvN(x[i,:], μ_adapt, Σ_chol_adapt, ph)
    end

    ## Propose Initial Angle
    θ = rand(eltype(x)) * 2 * π
    θ_min = θ - 2 * π
    θ_max = θ

    ## Propose initial first move
    @views x[i,:] .= ((x[i-1,:] - μ_adapt) .* cos(θ) .+  (z - μ_adapt) .* sin(θ)) .+ μ_adapt
    @views l_pdf = params.log_posterior(x[i,:])
    L_star = l_pdf
    if params.t_dist == true
        @views L_star -= dMvT(x[i,:], μ_adapt, Σ_chol_adapt, ph, params.ν, params.P)
    else
        @views L_star -= dMvN(x[i,:], μ_adapt, Σ_chol_adapt, ph)
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
        θ = θ_min + rand(eltype(x)) * (θ_max - θ_min)
        @views x[i,:] .= ((x[i-1,:] - μ_adapt) .* cos(θ) .+  (z - μ_adapt) .* sin(θ)) .+ μ_adapt
        @views l_pdf = params.log_posterior(x[i,:])
        L_star = l_pdf
        if params.t_dist == true
            @views L_star -= dMvT(x[i,:], μ_adapt, Σ_chol_adapt, ph, params.ν, params.P)
        else
            @views L_star -= dMvN(x[i,:], μ_adapt, Σ_chol_adapt, ph)
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


function AGESS_single_step_1d!(x::AbstractMatrix{Y}, params::AGESS_MCMC_params, 
                               μ_adapt::AbstractVector{Y}, 
                               Σ_chol_adapt::LowerTriangular{Y, <:AbstractMatrix{Y}}, i::T) where {Y<:AbstractFloat, T<:Integer}
    l_pdf::eltype(x) = 0.0
    z::eltype(x) = 0.0
    y::eltype(x) = 0.0
    L_star::eltype(x) = 0.0
    for j in randperm(params.P)
        
        ## Propose new z from N(0, Σ)
        if params.t_dist == true
            z = cond_rMvT_1d!(x[i,j], μ_adapt[j], Σ_chol_adapt[j,j], params.ν)
        else
            z = Σ_chol_adapt[j,j] * randn() + μ_adapt[j]
        end

        @views y = params.log_posterior(x[i,:]) + log(rand())
        if params.t_dist == true
            y -= dMvT_1d(x[i,j], μ_adapt[j], Σ_chol_adapt[j,j], params.ν)
        else
            y -= dMvN_1d(x[i,j], μ_adapt[j], Σ_chol_adapt[j,j])
        end

        ## Propose Initial Angle
        θ = rand(eltype(x)) * 2 * π
        θ_min = θ - 2 * π
        θ_max = θ

        ## Propose initial first move
        x[i,j] = ((x[i-1,j] - μ_adapt[j]) * cos(θ) +  (z - μ_adapt[j]) * sin(θ)) + μ_adapt[j]
        @views l_pdf = params.log_posterior(x[i,:])
        L_star =  l_pdf
        if params.t_dist == true
            L_star -= dMvT_1d(x[i,j], μ_adapt[j], Σ_chol_adapt[j,j], params.ν)
        else
            L_star -= dMvN_1d(x[i,j], μ_adapt[j], Σ_chol_adapt[j,j])
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
            θ = θ_min + rand(eltype(x)) * (θ_max - θ_min)
            x[i,j] = ((x[i-1,j] - μ_adapt[j]) * cos(θ) +  (z - μ_adapt[j]) * sin(θ)) + μ_adapt[j]
            @views l_pdf = params.log_posterior(x[i,:])
            L_star =  l_pdf
            if params.t_dist == true
                L_star -= dMvT_1d(x[i,j], μ_adapt[j], Σ_chol_adapt[j,j], params.ν)
            else
                L_star -= dMvN_1d(x[i,j], μ_adapt[j], Σ_chol_adapt[j,j])
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

function AGESS(log_posterior::Function, n_MCMC::T, P::T;
               μ_0::Union{<:AbstractVector{Y},Y} = 0.0, Σ_0::Union{<:AbstractMatrix{Y},Y} = 1.0,
               init_x::Union{<:AbstractVector{Y},Y} = 0.0, t_dist::Bool = true, ν::Y = 6.0, burnin::Y = 0.25,
               ϵ::Y = 0.1, single_step_prop::Y = 0.05, β::Y = 0.5) where {Y<:AbstractFloat, T<:Integer}
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

    ## Check rest of arguments
    @argcheck P >= 1
    @argcheck burnin >= 0.0
    @argcheck burnin < 1.0
    @argcheck ϵ >= 0.0
    @argcheck ϵ < 1.0
    @argcheck single_step_prop >= 0.0
    @argcheck single_step_prop < 1.0
    
    ## Initialize MCMC iterations
    x = ones(eltype(μ_0), n_MCMC, P)
    x[1,:] .= init_x
    x[2,:] .= init_x

    ## Auxiliary variable
    z = zeros(eltype(μ_0), P)

    ## Storage of log_posterior evaluations
    l_pdf = zeros(eltype(μ_0), n_MCMC) 

    ## Construct AGESS_MCMC_params
    params = AGESS_MCMC_params(log_posterior, μ_0, Σ_0, t_dist, ν, β, single_step_prop,
                               burnin, ϵ, n_MCMC, P)

    l_pdf[1] = params.log_posterior(init_x)
    @argcheck isfinite(l_pdf[1]) "Initial starting position of Markov chain must have finite posterior density above 0"


    burnin_num = floor(typeof(n_MCMC), params.burnin * params.n_MCMC)

    μ_adapt = copy(μ_0)
    μ_adapt_ph = copy(μ_0)
    ph = similar(μ_adapt)

    Σ_chol = cholesky(Σ_0)
    Σ_chol_adapt = deepcopy(Σ_chol)
    Σ_chol_adapt_ph = deepcopy(Σ_chol)

    ph_cholesky_update = ones(eltype(μ_0), P)
    w_const = max(2/3, ((cbrt(P) - 1) / cbrt(P)))
    N_J = 2
    n_j = 2

    prog = Progress(n_MCMC)

    for i in 2:n_MCMC
        if P >= 10
            if i < burnin_num * single_step_prop
                l_pdf[i] = AGESS_single_step_1d!(x, params, μ_adapt, Σ_chol_adapt.L, i)
            else
                if rand() > ϵ
                    l_pdf[i] = AGESS_single_step!(x, z, params, ph, μ_adapt, Σ_chol_adapt.L, i)
                elseif rand() > 0.5
                    l_pdf[i] = AGESS_single_step_1d!(x, params, μ_adapt, Σ_chol_adapt.L, i)
                else
                    l_pdf[i] = AGESS_single_step!(x, z, params, ph, μ_0, Σ_chol.L, i)
                end
            end
        else
            if rand() > ϵ
                l_pdf[i] = AGESS_single_step!(x, z, params, ph, μ_adapt, Σ_chol_adapt.L, i)
            else
                l_pdf[i] = AGESS_single_step!(x, z, params, ph, μ_0, Σ_chol.L, i)
            end

        end
        
        w_i = i^(-w_const)
        Σ_chol_adapt_ph.U .= sqrt((1 - w_i)) .*  Σ_chol_adapt_ph.U
        @views ph_cholesky_update .= sqrt(w_i) .* (x[i,:] .- μ_adapt_ph)
        lowrankupdate!(Σ_chol_adapt_ph, ph_cholesky_update)
        @views μ_adapt_ph .= (1 - w_i) * μ_adapt_ph +  w_i * x[i,:]
        
        ## Adapt mean and covariance
        if i == N_J
            Σ_chol_adapt.U .= Σ_chol_adapt_ph.U
            μ_adapt .= μ_adapt_ph
            n_j += 1
            N_J += floor(n_j^β)
        end

        ## Populate next value in Markov Chain
        if i < n_MCMC
            @views x[i+1,:] .= x[i,:]
        end

        # Update User
        if (i % 100) == 0
            update!(prog, i; showvalues = [("average lpdf", mean(l_pdf[(i-99):i]))])
        end
        
    end

    output = MCMC_output(x, l_pdf, params, Σ_chol_adapt.L * Σ_chol_adapt.U, μ_adapt)

    return output
end
