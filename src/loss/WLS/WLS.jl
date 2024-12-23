##### weighted least squares

############################################################################################
### Types
############################################################################################
"""
Weighted least squares estimation.

# Constructor

    SemWLS(;
        observed, implied,
        wls_weight_matrix = nothing,
        wls_weight_matrix_mean = nothing,
        approximate_hessian = false,
        kwargs...)

# Arguments
- `observed`: the `SemObserved` part of the model
- `implied::SemImplied`: the implied part of the model
- `approximate_hessian::Bool`: should the hessian be swapped for an approximation
- `wls_weight_matrix`: the weight matrix for weighted least squares.
    Defaults to GLS estimation (``0.5*(D^T*kron(S,S)*D)`` where D is the duplication matrix
    and S is the inverse of the observed covariance matrix)
- `wls_weight_matrix_mean`: the weight matrix for the mean part of weighted least squares.
    Defaults to GLS estimation (the inverse of the observed covariance matrix)

# Examples
```julia
my_wls = SemWLS(observed = my_observed, implied = my_implied)
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
    observed::SemObserved,
    implied::SemImplied,
    wls_weight_matrix::Union{AbstractMatrix, Nothing} = nothing,
    wls_weight_matrix_mean::Union{AbstractMatrix, Nothing} = nothing,
    approximate_hessian::Bool = false,
    kwargs...,
)
    nobs_vars = nobserved_vars(observed)
    s = vech(obs_cov(observed))
    size(s) == size(implied.Σ) ||
        throw(DimensionMismatch("SemWLS requires implied covariance to be in vech-ed form " *
                                "(vectorized lower triangular part of Σ matrix): $(size(s)) expected, $(size(implied.Σ)) found.\n" *
                                "$(nameof(typeof(implied))) must be constructed with vech=true."))

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
    size(wls_weight_matrix) == (length(s), length(s)) ||
        DimensionMismatch("wls_weight_matrix has to be of size $(length(s))×$(length(s))")

    if MeanStruct(implied) == HasMeanStruct
        if isnothing(wls_weight_matrix_mean)
        	@warn "Computing WLS weight matrix for the meanstructure using obs_cov()"
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
    implied::SemImpliedSymbolic,
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
        ∇²Σ = implied.∇²Σ
        J = -2 * (σ₋' * semwls.V)'
        implied.∇²Σ_eval!(∇²Σ, J, par)
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
