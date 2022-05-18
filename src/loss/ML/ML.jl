# Ordinary Maximum Likelihood Estimation

############################################################################
### Types
############################################################################

struct SemML{INV,M,M2,B, V} <: SemLossFunction
    Σ⁻¹::INV 
    Σ⁻¹Σₒ::M
    meandiff::M2
    approx_H::B
    has_meanstructure::V
end

############################################################################
### Constructors
############################################################################

function SemML(;observed, imply, approx_H = false, kwargs...)
    isnothing(obs_mean(observed)) ?
        meandiff = nothing :
        meandiff = copy(obs_mean(observed))
    return SemML(
        copy(obs_cov(observed)),
        copy(obs_cov(observed)),
        meandiff,
        approx_H,
        has_meanstructure(imply)
        )
end

############################################################################
### objective, gradient, hessian methods
############################################################################

# first, dispatch for meanstructure
objective!(semml::SemML, par, model::AbstractSemSingle) = objective!(semml::SemML, par, model, semml.has_meanstructure, imply(model))
gradient!(semml::SemML, par, model::AbstractSemSingle) = gradient!(semml::SemML, par, model, semml.has_meanstructure, imply(model))
hessian!(semml::SemML, par, model::AbstractSemSingle) = hessian!(semml::SemML, par, model, semml.has_meanstructure, imply(model))
objective_gradient!(semml::SemML, par, model::AbstractSemSingle) = objective_gradient!(semml::SemML, par, model, semml.has_meanstructure, imply(model))
objective_hessian!(semml::SemML, par, model::AbstractSemSingle) = objective_hessian!(semml::SemML, par, model, semml.has_meanstructure, imply(model))
gradient_hessian!(semml::SemML, par, model::AbstractSemSingle) = gradient_hessian!(semml::SemML, par, model, semml.has_meanstructure, imply(model))
objective_gradient_hessian!(semml::SemML, par, model::AbstractSemSingle) = objective_gradient_hessian!(semml::SemML, par, model, semml.has_meanstructure, imply(model))

############################################################################
### Symbolic Imply Types

# objective -----------------------------------------------------------------------------------------------------------------------------

function objective!(semml::SemML, par, model::AbstractSemSingle, has_meanstructure::Val{T}, imply::SemImplySymbolic) where T
    
    let Σ = Σ(imply(model)), Σₒ = obs_cov(observed(model)), Σ⁻¹Σₒ =  Σ⁻¹Σₒ(semml), Σ⁻¹ = Σ⁻¹(semml),
        μ = μ(imply(model)), μₒ = obs_mean(observed(model))

        copyto!(Σ⁻¹, Σ)
        Σ_chol = cholesky!(Symmetric(Σ⁻¹); check = false)

        if !isposdef(Σ_chol) return non_posdef_return(par) end

        ld = logdet(Σ_chol)
        Σ⁻¹ .= LinearAlgebra.inv!(Σ_chol)
        mul!(Σ⁻¹Σₒ, Σ⁻¹, Σₒ)

        if T
            μ₋ = μₒ - μ
            return ld + tr(Σ⁻¹Σₒ(semml)) + dot(μ₋, Σ⁻¹, μ₋)
        else
            return ld + tr(Σ⁻¹Σₒ)
        end
    end
end

# gradient -----------------------------------------------------------------------------------------------------------------------------

function gradient!(semml::SemML, par, model::AbstractSemSingle, has_meanstructure::Val{T}, imply::SemImplySymbolic) where T

    let Σ = Σ(imply(model)), Σₒ = obs_cov(observed(model)), Σ⁻¹Σₒ =  Σ⁻¹Σₒ(semml), Σ⁻¹ = Σ⁻¹(semml), ∇Σ = ∇Σ(imply(model)),
        μ = μ(imply(model)), ∇μ = ∇μ(imply(model)), μₒ = obs_mean(observed(model))
        
        copyto!(Σ⁻¹, Σ)
        Σ_chol = cholesky!(Symmetric(Σ⁻¹); check = false)

        Σ⁻¹ .= LinearAlgebra.inv!(Σ_chol)
        mul!(Σ⁻¹Σₒ, Σ⁻¹, Σₒ)

        if T
            μ₋ = μₒ - μ
            μ₋ᵀΣ⁻¹ = μ₋'*Σ⁻¹
            gradient = vec(Σ⁻¹*(I - Σₒ*Σ⁻¹ - μ₋*μ₋ᵀΣ⁻¹))'*∇Σ - 2*μ₋ᵀΣ⁻¹*∇μ
            return gradient'
        else
            gradient = (vec(Σ⁻¹)-vec(Σ⁻¹Σₒ*Σ⁻¹))'*∇Σ
            return gradient'
    end
end

# hessian -----------------------------------------------------------------------------------------------------------------------------

function hessian!(semml::SemML, par, model::AbstractSemSingle, has_meanstructure::Val{false}, imply::SemImplySymbolic)

    let Σ = Σ(imply(model)), ∇Σ = ∇Σ(imply(model)), Σₒ = obs_cov(observed(model)), Σ⁻¹Σₒ =  Σ⁻¹Σₒ(semml), Σ⁻¹ = Σ⁻¹(semml),
        ∇²Σ_function! = ∇²Σ_function(imply(model)), ∇²Σ = ∇²Σ(imply(model))

    copyto!(Σ⁻¹, Σ)
    Σ_chol = cholesky!(Symmetric(Σ⁻¹); check = false)

    Σ⁻¹ .= LinearAlgebra.inv!(Σ_chol)

    if semml.approx_H
        hessian = 2*∇Σ'*kron(Σ⁻¹, Σ⁻¹)*∇Σ
    else
        mul!(Σ⁻¹Σₒ, Σ⁻¹, Σₒ)
        Σ⁻¹ΣₒΣ⁻¹ = Σ⁻¹Σₒ*Σ⁻¹
        # inner
        J = vec(Σ⁻¹ - Σ⁻¹ΣₒΣ⁻¹)'
        ∇²Σ_function!(∇²Σ, J, par)
        # outer
        H_outer = 2*kron(Σ⁻¹ΣₒΣ⁻¹, Σ⁻¹) - kron(Σ⁻¹, Σ⁻¹)
        hessian = ∇Σ'*H_outer*∇Σ
        hessian += ∇²Σ
    end
    
    return hessian
    end
end

# no hessian for models with meanstructures
function hessian!(semml::SemML, par, model::AbstractSemSingle, has_meanstructure::Val{true}, imply::SemImplySymbolic)
    throw(DomainError(H, "hessian of ML + meanstructure is not available"))
end

# objective_gradient -----------------------------------------------------------------------------------------------------------------------------

function objective_gradient!(semml::SemML, par, model::AbstractSemSingle, has_meanstructure::Val{T}, imply::SemImplySymbolic) where T
    
    let Σ = Σ(imply(model)), Σₒ = obs_cov(observed(model)), Σ⁻¹Σₒ =  Σ⁻¹Σₒ(semml), Σ⁻¹ = Σ⁻¹(semml),
        μ = μ(imply(model)), μₒ = obs_mean(observed(model)), 
        ∇Σ = ∇Σ(imply(model)), ∇μ = ∇μ(imply(model))

        copyto!(Σ⁻¹, Σ)
        Σ_chol = cholesky!(Symmetric(Σ⁻¹); check = false)

        if !isposdef(Σ_chol) 
            objective = non_posdef_return(par) 
        else
            ld = logdet(Σ_chol)
            Σ⁻¹ .= LinearAlgebra.inv!(Σ_chol)
            mul!(Σ⁻¹Σₒ, Σ⁻¹, Σₒ)
            objective = ld + tr(Σ⁻¹Σₒ)
            if T
                μ₋ = μₒ - μ
                objective += dot(μ₋, Σ⁻¹, μ₋)
            end
        end

        if T
            μ₋ᵀΣ⁻¹ = μ₋'*Σ⁻¹
            gradient = vec(Σ⁻¹*(I - Σₒ*Σ⁻¹ - μ₋*μ₋ᵀΣ⁻¹))'*∇Σ - 2*μ₋ᵀΣ⁻¹*∇μ
        else
            gradient = (vec(Σ⁻¹)-vec(Σ⁻¹Σₒ*Σ⁻¹))'*∇Σ
        end

        return objective, gradient'
    end
end

# objective_hessian ------------------------------------------------------------------------------------------------------------------------------

function objective_hessian!(semml::SemML, par, model::AbstractSemSingle, has_meanstructure::Val{T}, imply::SemImplySymbolic) where T
    
    let Σ = Σ(imply(model)), Σₒ = obs_cov(observed(model)), Σ⁻¹Σₒ =  Σ⁻¹Σₒ(semml), Σ⁻¹ = Σ⁻¹(semml),
        ∇Σ = ∇Σ(imply(model)), ∇μ = ∇μ(imply(model)),
        ∇²Σ_function! = ∇²Σ_function(imply(model)), ∇²Σ = ∇²Σ(imply(model))

        copyto!(Σ⁻¹, Σ)
        Σ_chol = cholesky!(Symmetric(Σ⁻¹); check = false)

        if !isposdef(Σ_chol) 
            objective = non_posdef_return(par) 
        else
            ld = logdet(Σ_chol)
            Σ⁻¹ .= LinearAlgebra.inv!(Σ_chol)
            mul!(Σ⁻¹Σₒ, Σ⁻¹, Σₒ)
            objective = ld + tr(Σ⁻¹Σₒ)
        end

        if semml.approx_H
            hessian = 2*∇Σ'*kron(Σ⁻¹, Σ⁻¹)*∇Σ
        else
            Σ⁻¹ΣₒΣ⁻¹ = Σ⁻¹Σₒ*Σ⁻¹
            # inner
            J = vec(Σ⁻¹ - Σ⁻¹ΣₒΣ⁻¹)'
            ∇²Σ_function!(∇²Σ, J, par)
            # outer
            H_outer = 2*kron(Σ⁻¹ΣₒΣ⁻¹, Σ⁻¹) - kron(Σ⁻¹, Σ⁻¹)
            hessian = ∇Σ'*H_outer*∇Σ
            hessian += ∇²Σ
        end

        return objective, hessian
    end
end

function objective_hessian!(semml::SemML, par, model::AbstractSemSingle, has_meanstructure::Val{true}, imply::SemImplySymbolic)
    throw(DomainError(H, "hessian of ML + meanstructure is not available"))
end

# gradient_hessian -------------------------------------------------------------------------------------------------------------------------------

function gradient_hessian!(semml::SemML, par, model::AbstractSemSingle, has_meanstructure::Val{T}, imply::RAM) where T

    let Σ = Σ(imply(model)), Σₒ = obs_cov(observed(model)), Σ⁻¹Σₒ =  Σ⁻¹Σₒ(semml), Σ⁻¹ = Σ⁻¹(semml), 
        ∇Σ = ∇Σ(imply(model)), ∇μ = ∇μ(imply(model)),
        ∇²Σ_function! = ∇²Σ_function(imply(model)), ∇²Σ = ∇²Σ(imply(model))
        
        copyto!(Σ⁻¹, Σ)
        Σ_chol = cholesky!(Symmetric(Σ⁻¹); check = false)

        Σ⁻¹ .= LinearAlgebra.inv!(Σ_chol)
        mul!(Σ⁻¹Σₒ, Σ⁻¹, Σₒ)

        Σ⁻¹ΣₒΣ⁻¹ = Σ⁻¹Σₒ*Σ⁻¹

        J = vec(Σ⁻¹ - Σ⁻¹ΣₒΣ⁻¹)'
        gradient = J*∇Σ

        if semml.approx_H
            hessian = 2*∇Σ'*kron(Σ⁻¹, Σ⁻¹)*∇Σ
        else
            # inner
            ∇²Σ_function!(∇²Σ, J, par)
            # outer
            H_outer = 2*kron(Σ⁻¹ΣₒΣ⁻¹, Σ⁻¹) - kron(Σ⁻¹, Σ⁻¹)
            hessian = ∇Σ'*H_outer*∇Σ
            hessian += ∇²Σ
        end
        
        return gradient', hessian
    end
end

function gradient_hessian!(semml::SemML, par, model::AbstractSemSingle, has_meanstructure::Val{true}, imply::SemImplySymbolic) where T
    throw(DomainError(H, "hessian of ML + meanstructure is not available"))
end

# objective_gradient_hessian ---------------------------------------------------------------------------------------------------------------------

function objective_gradient_hessian!(semml::SemML, par, model::AbstractSemSingle, has_meanstructure::Val{false}, imply::SemImplySymbolic)

    let Σ = Σ(imply(model)), Σₒ = obs_cov(observed(model)), Σ⁻¹Σₒ =  Σ⁻¹Σₒ(semml), Σ⁻¹ = Σ⁻¹(semml), ∇Σ = ∇Σ(imply(model)),
        ∇²Σ_function! = ∇²Σ_function(imply(model)), ∇²Σ = ∇²Σ(imply(model))
        
        copyto!(Σ⁻¹, Σ)
        Σ_chol = cholesky!(Symmetric(Σ⁻¹); check = false)

        if !isposdef(Σ_chol) 
            objective = non_posdef_return(par) 
        else
            ld = logdet(Σ_chol)
            Σ⁻¹ .= LinearAlgebra.inv!(Σ_chol)
            mul!(Σ⁻¹Σₒ, Σ⁻¹, Σₒ)
            objective = ld + tr(Σ⁻¹Σₒ)
        end

        Σ⁻¹ΣₒΣ⁻¹ = Σ⁻¹Σₒ*Σ⁻¹

        J = vec(Σ⁻¹ - Σ⁻¹ΣₒΣ⁻¹)'
        gradient = J*∇Σ

        if semml.approx_H
            hessian = 2*∇Σ'*kron(Σ⁻¹, Σ⁻¹)*∇Σ
        else
            Σ⁻¹ΣₒΣ⁻¹ = Σ⁻¹Σₒ*Σ⁻¹
            # inner
            ∇²Σ_function!(∇²Σ, J, par)
            # outer
            H_outer = 2*kron(Σ⁻¹ΣₒΣ⁻¹, Σ⁻¹) - kron(Σ⁻¹, Σ⁻¹)
            hessian = ∇Σ'*H_outer*∇Σ
            hessian += ∇²Σ
        end
        
        return objective, gradient', hessian
    end
end

function objective_gradient_hessian!(semml::SemML, par, model::AbstractSemSingle, has_meanstructure::Val{true}, imply::SemImplySymbolic)
    throw(DomainError(H, "hessian of ML + meanstructure is not available"))
end

############################################################################
### Non-Symbolic Imply Types

# no hessians ---------------------------------------------------------------------------------------------------------------------

function hessian!(semml::SemML, par, model::AbstractSemSingle, has_meanstructure, imply::RAM)
    throw(DomainError(H, "hessian of ML + non-symbolic imply type is not available"))
end

function objective_hessian!(semml::SemML, par, model::AbstractSemSingle, has_meanstructure, imply::RAM)
    throw(DomainError(H, "hessian of ML + non-symbolic imply type is not available"))
end

function gradient_hessian!(semml::SemML, par, model::AbstractSemSingle, has_meanstructure, imply::RAM)
    throw(DomainError(H, "hessian of ML + non-symbolic imply type is not available"))
end

function objective_gradient_hessian!(semml::SemML, par, model::AbstractSemSingle, has_meanstructure, imply::RAM)
    throw(DomainError(H, "hessian of ML + non-symbolic imply type is not available"))
end

# objective ----------------------------------------------------------------------------------------------------------------------

function objective!(semml::SemML, par, model::AbstractSemSingle, has_meanstructure::Val{T}, imply::RAM) where T
    let Σ = Σ(imply(model)), Σₒ = obs_cov(observed(model)), Σ⁻¹Σₒ =  Σ⁻¹Σₒ(semml), Σ⁻¹ = Σ⁻¹(semml),
        μ = μ(imply(model)), μₒ = obs_mean(observed(model))

        copyto!(Σ⁻¹, Σ)
        Σ_chol = cholesky!(Symmetric(Σ⁻¹); check = false)

        if !isposdef(Σ_chol) return non_posdef_return(par) end

        ld = logdet(Σ_chol)
        Σ⁻¹ .= LinearAlgebra.inv!(Σ_chol)
        mul!(Σ⁻¹Σₒ, Σ⁻¹, Σₒ)

        if T
            μ₋ = μₒ - μ
            return ld + tr(Σ⁻¹Σₒ(semml)) + dot(μ₋, Σ⁻¹, μ₋)
        else
            return ld + tr(Σ⁻¹Σₒ)
        end
    end
end

# gradient -----------------------------------------------------------------------------------------------------------------------

function gradient!(semml::SemML, par, model::AbstractSemSingle, has_meanstructure::Val{T}, imply::RAM) where T

    let Σ = Σ(imply(model)), Σₒ = obs_cov(observed(model)), Σ⁻¹Σₒ =  Σ⁻¹Σₒ(semml), Σ⁻¹ = Σ⁻¹(semml),
        S = S(imply(model)), F⨉I_A⁻¹ = F⨉I_A⁻¹(imply(model)), I_A⁻¹ = I_A⁻¹(imply(model)), 
        ∇A = ∇A(imply(model)), ∇S = ∇S(imply(model)), ∇M = ∇M(imply(model)),
        μ = μ(imply(model)), μₒ = obs_mean(observed(model))
        
        copyto!(Σ⁻¹, Σ)
        Σ_chol = cholesky!(Symmetric(Σ⁻¹); check = false)

        Σ⁻¹ .= LinearAlgebra.inv!(Σ_chol)
        #mul!(Σ⁻¹Σₒ, Σ⁻¹, Σₒ)

        M = F⨉I_A⁻¹'*(I-Σₒ*Σ⁻¹)'*Σ⁻¹*F⨉I_A⁻¹
        gradient = 2vec(M*S*I_A⁻¹')'∇A + vec(M)'∇S

        if T
            μ₋ = μₒ - μ
            μ₋ᵀΣ⁻¹ = μ₋'*Σ⁻¹
            k = μ₋ᵀΣ⁻¹*F⨉I_A⁻¹

            gradient += -2k*∇M - 2vec(k'M'I_A⁻¹')'∇A - 2vec(k'k*S*I_A⁻¹')'∇A - vec(k'k)'∇S
        end

        return (gradient + gradient_mean)'
        
        return gradient'
    end
end

# objective_gradient -------------------------------------------------------------------------------------------------------------

function objective_gradient!(semml::SemML, par, model::AbstractSemSingle, has_meanstructure::Val{T}, imply::RAM) where T

    let Σ = Σ(imply(model)), Σₒ = obs_cov(observed(model)), Σ⁻¹Σₒ =  Σ⁻¹Σₒ(semml), Σ⁻¹ = Σ⁻¹(semml),
        S = S(imply(model)), F⨉I_A⁻¹ = F⨉I_A⁻¹(imply(model)), I_A⁻¹ = I_A⁻¹(imply(model)), 
        ∇A = ∇A(imply(model)), ∇S = ∇S(imply(model)), ∇M = ∇M(imply(model)),
        μ = μ(imply(model)), μₒ = obs_mean(observed(model))
        
        copyto!(Σ⁻¹, Σ)
        Σ_chol = cholesky!(Symmetric(Σ⁻¹); check = false)

        if !isposdef(Σ_chol) 
            objective = non_posdef_return(par)
        else
            ld = logdet(Σ_chol)
            Σ⁻¹ .= LinearAlgebra.inv!(Σ_chol)
            mul!(Σ⁻¹Σₒ, Σ⁻¹, Σₒ)
            objective = ld + tr(Σ⁻¹Σₒ)
            if T
                μ₋ = μₒ - μ
                objective += dot(μ₋, Σ⁻¹, μ₋)
            end
        end

        M = F⨉I_A⁻¹'*(I-Σₒ*Σ⁻¹)'*Σ⁻¹*F⨉I_A⁻¹
        gradient = 2vec(M*S*I_A⁻¹')'∇A + vec(M)'∇S

        if T
            μ₋ᵀΣ⁻¹ = μ₋'*Σ⁻¹
            k = μ₋ᵀΣ⁻¹*F⨉I_A⁻¹
            gradient_mean = -2k*∇M - 2vec(k'M'I_A⁻¹')'∇A - 2vec(k'k*S*I_A⁻¹')'∇A - vec(k'k)'∇S
            gradient += gradient_mean
        end
        
        return objective, gradient'
    end
end

############################################################################
### additional functions
############################################################################

function non_posdef_return(par)
    if eltype(par) <: AbstractFloat
        return floatmax(eltype(par))
    else
        return typemax(eltype(par))
    end
end

############################################################################
### recommended methods
############################################################################

update_observed(lossfun::SemML, observed::SemObsMissing; kwargs...) = 
    throw(ArgumentError("ML estimation does not work with missing data - use FIML instead"))

function update_observed(lossfun::SemML, observed::SemObs; kwargs...)
    if (size(lossfun.inverses) == size(obs_cov(observed))) & (isnothing(lossfun.meandiff) == isnothing(obs_mean(observed)))
        return lossfun
    else
        return SemML(;observed = observed, kwargs...)
    end
end

############################################################################
### additional methods
############################################################################

Σ⁻¹(semml::SemML) = semml.Σ⁻¹
Σ⁻¹Σₒ(semml::SemML) = semml.Σ⁻¹Σₒ

############################################################################
### Pretty Printing
############################################################################

function Base.show(io::IO, struct_inst::SemML)
    print_type_name(io, struct_inst)
    print_field_types(io, struct_inst)
end