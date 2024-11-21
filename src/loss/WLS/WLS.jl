##### weighted least squares

############################################################################################
### Types
############################################################################################
"""
Weighted least squares estimation.

# Constructor

    SemWLS(;
        observed,
        meanstructure = false,
        wls_weight_matrix = nothing,
        wls_weight_matrix_mean = nothing,
        approximate_hessian = false,
        kwargs...)

# Arguments
- `observed`: the `SemObserved` part of the model
- `meanstructure::Bool`: does the model have a meanstructure?
- `approximate_hessian::Bool`: should the hessian be swapped for an approximation
- `wls_weight_matrix`: the weight matrix for weighted least squares.
    Defaults to GLS estimation (``0.5*(D^T*kron(S,S)*D)`` where D is the duplication matrix
    and S is the inverse of the observed covariance matrix)
- `wls_weight_matrix_mean`: the weight matrix for the mean part of weighted least squares.
    Defaults to GLS estimation (the inverse of the observed covariance matrix)

# Examples
```julia
my_wls = SemWLS(observed = my_observed)
```

# Interfaces
Analytic gradients are available, and for models without a meanstructure, also analytic hessians.

# Extended help
## Implementation
Subtype of `SemLossFunction`.
"""
struct SemWLS{HE <: HessianEval, Vt, St, C} <: SemLossFunction
    hessianeval::HE
    V::Vt
    σₒ::St
    V_μ::C
end

############################################################################################
### Constructors
############################################################################################

SemWLS{HE}(args...) where {HE <: HessianEval} =
    SemWLS{HE, map(typeof, args)...}(HE(), args...)

function SemWLS(;
    observed,
    wls_weight_matrix = nothing,
    wls_weight_matrix_mean = nothing,
    approximate_hessian = false,
    meanstructure = false,
    kwargs...,
)
    nobs_vars = nobserved_vars(observed)
    tril_ind = filter(x -> (x[1] >= x[2]), CartesianIndices(obs_cov(observed)))
    s = obs_cov(observed)[tril_ind]

    # compute V here
    if isnothing(wls_weight_matrix)
        D = duplication_matrix(nobs_vars)
        S = inv(obs_cov(observed))
        S = kron(S, S)
        wls_weight_matrix = 0.5 * (D' * S * D)
    else
        size(wls_weight_matrix) == (length(tril_ind), length(tril_ind)) ||
            DimensionMismatch(
                "wls_weight_matrix has to be of size $(length(tril_ind))×$(length(tril_ind))",
            )
    end

    if meanstructure
        if isnothing(wls_weight_matrix_mean)
            wls_weight_matrix_mean = inv(obs_cov(observed))
        else
            size(wls_weight_matrix_mean) == (nobs_vars, nobs_vars) || DimensionMismatch(
                "wls_weight_matrix_mean has to be of size $(nobs_vars)×$(nobs_vars)",
            )
        end
    else
        isnothing(wls_weight_matrix_mean) ||
            @warn "Ignoring wls_weight_matrix_mean since meanstructure is disabled"
        wls_weight_matrix_mean = nothing
    end
    HE = approximate_hessian ? ApproxHessian : ExactHessian

    return SemWLS{HE}(wls_weight_matrix, s, wls_weight_matrix_mean)
end

############################################################################
### methods
############################################################################

function evaluate!(
    objective,
    gradient,
    hessian,
    semwls::SemWLS,
    implied::SemImplySymbolic,
    model::AbstractSemSingle,
    par,
)
    if !isnothing(hessian) && (MeanStruct(implied) === HasMeanStruct)
        error("hessian of WLS with meanstructure is not available")
    end

    V = semwls.V
    ∇σ = implied.∇Σ

    σ = implied.Σ
    σₒ = semwls.σₒ
    σ₋ = σₒ - σ

    isnothing(objective) || (objective = dot(σ₋, V, σ₋))
    if !isnothing(gradient)
        if issparse(∇σ)
            gradient .= (σ₋' * V * ∇σ)'
        else # save one allocation
            mul!(gradient, σ₋' * V, ∇σ) # actually transposed, but should be fine for vectors
        end
        gradient .*= -2
    end
    isnothing(hessian) || (mul!(hessian, ∇σ' * V, ∇σ, 2, 0))
    if !isnothing(hessian) && (HessianEval(semwls) === ExactHessian)
        ∇²Σ_function! = implied.∇²Σ_function
        ∇²Σ = implied.∇²Σ
        J = -2 * (σ₋' * semwls.V)'
        ∇²Σ_function!(∇²Σ, J, par)
        hessian .+= ∇²Σ
    end
    if MeanStruct(implied) === HasMeanStruct
        μ = implied.μ
        μₒ = obs_mean(observed(model))
        μ₋ = μₒ - μ
        V_μ = semwls.V_μ
        if !isnothing(objective)
            objective += dot(μ₋, V_μ, μ₋)
        end
        if !isnothing(gradient)
            mul!(gradient, (V_μ * implied.∇μ)', μ₋, -2, 1)
        end
    end

    return objective
end

############################################################################################
### Recommended methods
############################################################################################

update_observed(lossfun::SemWLS, observed::SemObserved; kwargs...) =
    SemWLS(; observed = observed, kwargs...)
