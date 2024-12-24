# Ordinary Maximum Likelihood Estimation

############################################################################################
### Types
############################################################################################
"""
Maximum likelihood estimation.

# Constructor

    SemML(observed, implied; approximate_hessian = false)

# Arguments
- `observed::SemObserved`: the observed part of the model
- `implied::SemImplied`: [`SemImplied`](@ref) instance
- `approximate_hessian::Bool`: if hessian-based optimization is used, should the hessian be swapped for an approximation

# Examples
```julia
my_ml = SemML(my_observed, my_implied)
```
"""
struct SemML{O, I, HE <: HessianEval, M} <: SemLoss{O, I}
    hessianeval::HE

    observed::O
    implied::I

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

function SemML(
    observed::SemObserved,
    implied::SemImplied;
    approximate_hessian::Bool = false,
    kwargs...,
)
    # check integrity
    check_observed_vars(observed, implied)

    he = approximate_hessian ? ApproxHessian() : ExactHessian()
    obsXobs = parent(obs_cov(observed))
    nobs_vars = nobserved_vars(implied)
    nvars = SEM.nvars(implied)

    return SemML{typeof(observed), typeof(implied), typeof(he), typeof(obsXobs)}(
        he,
        observed,
        implied,
        similar(obsXobs),
        similar(obsXobs),
        similar(obsXobs),
        similar(obsXobs, (nobs_vars, nvars)),
        similar(obsXobs, (nvars, nvars)),
        similar(obsXobs, (nvars, nvars)),
        similar(obsXobs, (nvars, nvars)),
    )
end

############################################################################################
### objective, gradient, hessian methods
############################################################################################

############################################################################################
### Symbolic Implied Types

function evaluate!(
    objective,
    gradient,
    hessian,
    ml::SemML{<:Any, <:SemImpliedSymbolic},
    par,
)
    implied = SEM.implied(ml)

    if !isnothing(hessian)
        (MeanStruct(implied) === HasMeanStruct) &&
            throw(DomainError(H, "hessian of ML + meanstructure is not available"))
    end

    Σ = implied.Σ
    Σₒ = obs_cov(observed(ml))

    Σ⁻¹ = copy!(ml.obsXobs_1, Σ)
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
    Σ⁻¹Σₒ = mul!(ml.obsXobs_2, Σ⁻¹, Σₒ)
    isnothing(objective) || (objective = ld + tr(Σ⁻¹Σₒ))

    if MeanStruct(implied) === HasMeanStruct
        μ = implied.μ
        μₒ = obs_mean(observed(ml))
        μ₋ = μₒ - μ

        isnothing(objective) || (objective += dot(μ₋, Σ⁻¹, μ₋))
        if !isnothing(gradient)
            ∇Σ = implied.∇Σ
            ∇μ = implied.∇μ
            μ₋ᵀΣ⁻¹ = μ₋' * Σ⁻¹
            J = copyto!(ml.obsXobs_3, Σ⁻¹)
            mul!(J, Σ⁻¹Σₒ, Σ⁻¹, -1, 1)
            mul!(J, μ₋ᵀΣ⁻¹', μ₋ᵀΣ⁻¹, -1, 1)
            mul!(gradient, ∇Σ', vec(J))
            mul!(gradient, ∇μ', μ₋ᵀΣ⁻¹', -2, 1)
        end
    elseif !isnothing(gradient) || !isnothing(hessian)
        ∇Σ = implied.∇Σ
        Σ⁻¹ΣₒΣ⁻¹ = mul!(ml.obsXobs_3, Σ⁻¹Σₒ, Σ⁻¹)
        J = vec(Σ⁻¹ - Σ⁻¹ΣₒΣ⁻¹)'
        if !isnothing(gradient)
            mul!(gradient, ∇Σ', J')
        end
        if !isnothing(hessian)
            if HessianEval(ml) === ApproxHessian
                mul!(hessian, ∇Σ' * kron(Σ⁻¹, Σ⁻¹), ∇Σ, 2, 0)
            else
                ∇²Σ = implied.∇²Σ
                # inner
                implied.∇²Σ_eval!(∇²Σ, J, par)
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
### Non-Symbolic Implied Types

function evaluate!(objective, gradient, hessian, ml::SemML, par)
    if !isnothing(hessian)
        error("hessian of ML + non-symbolic implied type is not available")
    end

    implied = SEM.implied(ml)

    Σ = implied.Σ
    Σₒ = obs_cov(observed(ml))

    Σ⁻¹ = copy!(ml.obsXobs_1, Σ)
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
    Σ⁻¹Σₒ = mul!(ml.obsXobs_2, Σ⁻¹, Σₒ)

    if !isnothing(objective)
        objective = ld + tr(Σ⁻¹Σₒ)

        if MeanStruct(implied) === HasMeanStruct
            μ = implied.μ
            μₒ = obs_mean(observed(ml))
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
            ml.varXvar_1,
            F⨉I_A⁻¹',
            mul!(ml.obsXvar_1, Symmetric(mul!(ml.obsXobs_3, one_Σ⁻¹Σₒ, Σ⁻¹)), F⨉I_A⁻¹),
        )
        mul!(
            gradient,
            ∇A',
            vec(mul!(ml.varXvar_3, Symmetric(C), mul!(ml.varXvar_2, S, I_A⁻¹'))),
            2,
            0,
        )
        mul!(gradient, ∇S', vec(C), 1, 1)

        if MeanStruct(implied) === HasMeanStruct
            μ = implied.μ
            μₒ = obs_mean(observed(ml))
            ∇M = implied.∇M
            M = implied.M
            μ₋ = μₒ - μ
            μ₋ᵀΣ⁻¹ = μ₋' * Σ⁻¹
            k = μ₋ᵀΣ⁻¹ * F⨉I_A⁻¹
            mul!(gradient, ∇M', k', -2, 1)
            mul!(gradient, ∇A', vec(mul!(ml.varXvar_1, k', (I_A⁻¹ * (M + S * k'))')), -2, 1)
            mul!(gradient, ∇S', vec(mul!(ml.varXvar_2, k', k)), -1, 1)
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

function update_observed(
    lossfun::SemML,
    observed::SemObserved;
    implied::SemImplied,
    kwargs...,
)
    if (obs_cov(lossfun) == obs_cov(observed)) && (obs_mean(lossfun) == obs_mean(observed))
        return lossfun # no change
    else
        return SemML(observed, implied; kwargs...)
    end
end
