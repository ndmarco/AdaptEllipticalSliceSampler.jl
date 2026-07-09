"""
    dMvT(x, μ, Σ_chol, ph, ν, P)

Calculates the pdf of a multivariate T distribution
"""
function dMvT(x::AbstractVector{Y}, μ::AbstractVector{Y}, Σ_chol::LowerTriangular{Y, <:AbstractMatrix{Y}}, 
              ph::AbstractVector{Y}, ν::Y, P::T) where {Y<:AbstractFloat, T<:Integer}
    ph .= (x .- μ)
    ldiv!(Σ_chol, ph)
    pdf::eltype(x) = -(ν + P) * 0.5 * log1p((dot(ph, ph) / ν))
    return pdf
end

"""
    dMvN(x, μ, Σ_chol, ph)

Calculates the pdf of a multivariate normal distribution
"""
function dMvN(x::AbstractVector{Y}, μ::AbstractVector{Y}, Σ_chol::LowerTriangular{Y, <:AbstractMatrix{Y}}, 
              ph::AbstractVector{Y}) where {Y<:AbstractFloat}
    ph .= (x .- μ)
    ldiv!(Σ_chol, ph)
    pdf::eltype(x) = -0.5 * dot(ph, ph)
    return pdf
end

"""
    cond_rMvT(z, x, μ, Σ_chol, ν, ph, P)

Conditionally generates a sample from a multivariate T distribution
"""
function cond_rMvT!(z::AbstractVector{Y}, x::AbstractVector{Y}, μ::AbstractVector{Y}, 
                    Σ_chol::LowerTriangular{Y, <:AbstractMatrix{Y}}, ν::Y, 
                    ph::AbstractVector{Y}, P::T) where {Y<:AbstractFloat, T<:Integer}
    randn!(z)
    lmul!(Σ_chol, z)
    ph .= (x .- μ)
    ldiv!(Σ_chol, ph)
    d = dot(ph, ph)
    ṽ = ν + P

    scale = sqrt((ν + d) / ṽ) / sqrt(rand(Gamma(ṽ / 2, 2 / ṽ)))
    @. z = z * scale + μ

    return nothing
end

"""
    dMvT_1d(x, μ, σ, ν)

Calculates the pdf of a one-dimensional T distribution
"""
function dMvT_1d(x::Y, μ::Y, σ::Y, ν::Y) where {Y<:AbstractFloat}
    pdf::typeof(x) = -(ν + 1) * 0.5 * log1p(((x - μ)^2 / (σ^2)) / ν)
    return pdf
end

"""
    dMvN_1d(x, μ, σ)

Calculates the pdf of a one-dimensional normal distribution
"""
function dMvN_1d(x::Y, μ::Y, σ::Y) where {Y<:AbstractFloat} 
    pdf::typeof(x) =  - 0.5 * ((x - μ)^2 / (σ^2))
    return pdf
end

"""
    cond_rMvT_1d(z, x, μ, σ, ν)

Conditionally generates a sample from a one-dimensional T distribution
"""
function cond_rMvT_1d(x::Y, μ::Y, σ::Y, ν::Y) where {Y<:AbstractFloat}
    z::typeof(x) = σ * randn() 
    d = (x - μ)^2 / (σ^2)
    ṽ = ν + 1
    z *= sqrt((ν + d) / ṽ)
    z *= 1 / sqrt(rand(Gamma(ṽ/2, 2/ ṽ)))
    z += μ

    return z
end