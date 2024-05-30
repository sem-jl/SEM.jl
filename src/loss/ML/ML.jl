# Ordinary Maximum Likelihood Estimation

############################################################################################
### Types
############################################################################################
"""
Maximum likelihood estimation.

# Constructor

    SemML(; observed, imply, approximate_hessian = false, kwargs...)

# Arguments
- `observed::SemObserved`: the observed part of the model
- `imply::SemImply`: the implied part of the model
- `approximate_hessian::Bool`: if hessian-based optimization is used, should the hessian be swapped for an approximation

# Examples
```julia
my_ml = SemML(observed = my_observed, imply = my_implied)
```

# Interfaces
Analytic gradients are available, and for models without a meanstructure, also analytic hessians.

# Extended help
## Implementation
Subtype of `SemLossFunction`.
"""
struct SemML{HE <: HessianEval, M} <: SemLossFunction
    hessianeval::HE

    # pre-allocated arrays to store intermediate results in evaluate!()
    obsXobs_1::M
    obsXobs_2::M
    obsXobs_3::M
    obsXvar_1::M
    varXvar_1::M
    varXvar_2::M
    varXvar_3::M
end

############################################################################################
### Constructors
############################################################################################

function SemML(;
    observed::SemObserved,
    imply::SemImply,
    approximate_hessian::Bool = false,
    kwargs...,
)
    he = approximate_hessian ? ApproxHessian() : ExactHessian()
    obsXobs = parent(obs_cov(observed))
    nobs = nobserved_vars(observed)
    nvar = nvars(imply)

    return SemML{typeof(he), typeof(obsXobs)}(
        he,
        similar(obsXobs),
        similar(obsXobs),
        similar(obsXobs),
        similar(obsXobs, (nobs, nvar)),
        similar(obsXobs, (nvar, nvar)),
        similar(obsXobs, (nvar, nvar)),
        similar(obsXobs, (nvar, nvar)),
    )
end

############################################################################################
### objective, gradient, hessian methods
############################################################################################

############################################################################################
### Symbolic Imply Types

function evaluate!(
    objective,
    gradient,
    hessian,
    semml::SemML,
    implied::SemImplySymbolic,
    model::AbstractSemSingle,
    par,
)
    if !isnothing(hessian)
        (MeanStruct(implied) === HasMeanStruct) &&
            throw(DomainError(H, "hessian of ML + meanstructure is not available"))
    end

    Σ = implied.Σ
    Σₒ = obs_cov(observed(model))

    Σ⁻¹ = copy!(semml.obsXobs_1, Σ)
    Σ_chol = cholesky!(Symmetric(Σ⁻¹); check = false)
    if !isposdef(Σ_chol)
        #@warn "∑⁻¹ is not positive definite"
        isnothing(objective) || (objective = non_posdef_return(par))
        isnothing(gradient) || fill!(gradient, 1)
        isnothing(hessian) || copyto!(hessian, I)
        return objective
    end
    ld = logdet(Σ_chol)
    Σ⁻¹ = LinearAlgebra.inv!(Σ_chol)
    Σ⁻¹Σₒ = mul!(semml.obsXobs_2, Σ⁻¹, Σₒ)
    isnothing(objective) || (objective = ld + tr(Σ⁻¹Σₒ))

    if MeanStruct(implied) === HasMeanStruct
        μ = implied.μ
        μₒ = obs_mean(observed(model))
        μ₋ = μₒ - μ

        isnothing(objective) || (objective += dot(μ₋, Σ⁻¹, μ₋))
        if !isnothing(gradient)
            ∇Σ = implied.∇Σ
            ∇μ = implied.∇μ
            μ₋ᵀΣ⁻¹ = μ₋' * Σ⁻¹
            J = copyto!(semml.obsXobs_3, Σ⁻¹)
            mul!(J, Σ⁻¹Σₒ, Σ⁻¹, -1, 1)
            mul!(J, μ₋ᵀΣ⁻¹', μ₋ᵀΣ⁻¹, -1, 1)
            mul!(gradient, ∇Σ', vec(J))
            mul!(gradient, ∇μ', μ₋ᵀΣ⁻¹', -2, 1)
        end
    elseif !isnothing(gradient) || !isnothing(hessian)
        ∇Σ = implied.∇Σ
        Σ⁻¹ΣₒΣ⁻¹ = mul!(semml.obsXobs_3, Σ⁻¹Σₒ, Σ⁻¹)
        J = vec(Σ⁻¹ - Σ⁻¹ΣₒΣ⁻¹)'
        if !isnothing(gradient)
            mul!(gradient, ∇Σ', J')
        end
        if !isnothing(hessian)
            if HessianEval(semml) === ApproxHessian
                mul!(hessian, ∇Σ' * kron(Σ⁻¹, Σ⁻¹), ∇Σ, 2, 0)
            else
                ∇²Σ_function! = implied.∇²Σ_function
                ∇²Σ = implied.∇²Σ
                # inner
                ∇²Σ_function!(∇²Σ, J, par)
                # outer
                H_outer = kron(2Σ⁻¹ΣₒΣ⁻¹ - Σ⁻¹, Σ⁻¹)
                mul!(hessian, ∇Σ' * H_outer, ∇Σ)
                hessian .+= ∇²Σ
            end
        end
    end
    return objective
end

############################################################################################
### Non-Symbolic Imply Types

function evaluate!(
    objective,
    gradient,
    hessian,
    semml::SemML,
    implied::RAM,
    model::AbstractSemSingle,
    par,
)
    if !isnothing(hessian)
        error("hessian of ML + non-symbolic imply type is not available")
    end

    Σ = implied.Σ
    Σₒ = obs_cov(observed(model))

    Σ⁻¹ = copy!(semml.obsXobs_1, Σ)
    Σ_chol = cholesky!(Symmetric(Σ⁻¹); check = false)
    if !isposdef(Σ_chol)
        #@warn "Σ⁻¹ is not positive definite"
        isnothing(objective) || (objective = non_posdef_return(par))
        isnothing(gradient) || fill!(gradient, 1)
        isnothing(hessian) || copyto!(hessian, I)
        return objective
    end
    ld = logdet(Σ_chol)
    Σ⁻¹ = LinearAlgebra.inv!(Σ_chol)
    Σ⁻¹Σₒ = mul!(semml.obsXobs_2, Σ⁻¹, Σₒ)

    if !isnothing(objective)
        objective = ld + tr(Σ⁻¹Σₒ)

        if MeanStruct(implied) === HasMeanStruct
            μ = implied.μ
            μₒ = obs_mean(observed(model))
            μ₋ = μₒ - μ
            objective += dot(μ₋, Σ⁻¹, μ₋)
        end
    end

    if !isnothing(gradient)
        S = implied.S
        F⨉I_A⁻¹ = implied.F⨉I_A⁻¹
        I_A⁻¹ = implied.I_A⁻¹
        ∇A = implied.∇A
        ∇S = implied.∇S

        # reuse Σ⁻¹Σₒ to calculate I-Σ⁻¹Σₒ
        one_Σ⁻¹Σₒ = Σ⁻¹Σₒ
        lmul!(-1, one_Σ⁻¹Σₒ)
        one_Σ⁻¹Σₒ[diagind(one_Σ⁻¹Σₒ)] .+= 1

        C = mul!(
            semml.varXvar_1,
            F⨉I_A⁻¹',
            mul!(
                semml.obsXvar_1,
                Symmetric(mul!(semml.obsXobs_3, one_Σ⁻¹Σₒ, Σ⁻¹)),
                F⨉I_A⁻¹,
            ),
        )
        mul!(
            gradient,
            ∇A',
            vec(mul!(semml.varXvar_3, Symmetric(C), mul!(semml.varXvar_2, S, I_A⁻¹'))),
            2,
            0,
        )
        mul!(gradient, ∇S', vec(C), 1, 1)

        if MeanStruct(implied) === HasMeanStruct
            μ = implied.μ
            μₒ = obs_mean(observed(model))
            ∇M = implied.∇M
            M = implied.M
            μ₋ = μₒ - μ
            μ₋ᵀΣ⁻¹ = μ₋' * Σ⁻¹
            k = μ₋ᵀΣ⁻¹ * F⨉I_A⁻¹
            mul!(gradient, ∇M', k', -2, 1)
            mul!(
                gradient,
                ∇A',
                vec(mul!(semml.varXvar_1, k', (I_A⁻¹ * (M + S * k'))')),
                -2,
                1,
            )
            mul!(gradient, ∇S', vec(mul!(semml.varXvar_2, k', k)), -1, 1)
        end
    end

    return objective
end

############################################################################################
### additional functions
############################################################################################

function non_posdef_return(par)
    if eltype(par) <: AbstractFloat
        return floatmax(eltype(par))
    else
        return typemax(eltype(par))
    end
end

############################################################################################
### recommended methods
############################################################################################

update_observed(lossfun::SemML, observed::SemObservedMissing; kwargs...) =
    error("ML estimation does not work with missing data - use FIML instead")

function update_observed(lossfun::SemML, observed::SemObserved; kwargs...)
    if size(lossfun.Σ⁻¹) == size(obs_cov(observed))
        return lossfun
    else
        return SemML(; observed = observed, kwargs...)
    end
end
