############################################################################################
### Types
############################################################################################
"""
Full information maximum likelihood estimation. Can handle observed data with missings.

# Constructor

    SemFIML(;observed, specification, kwargs...)

# Arguments
- `observed::SemObservedMissing`: the observed part of the model
- `specification`: either a `RAMMatrices` or `ParameterTable` object

# Examples
```julia
my_fiml = SemFIML(observed = my_observed, specification = my_parameter_table)
```

# Interfaces
Analytic gradients are available.

# Extended help
## Implementation
Subtype of `SemLossFunction`.
"""
mutable struct SemFIML{INV, C, L, O, M, IM, I, T, W} <: SemLossFunction
    hessianeval::ExactHessian
    inverses::INV #preallocated inverses of imp_cov
    choleskys::C #preallocated choleskys
    logdets::L #logdets of implied covmats

    ∇ind::O

    imp_mean::IM
    meandiff::M
    imp_inv::I

    mult::T

    commutator::CommutationMatrix

    interaction::W
end

############################################################################################
### Constructors
############################################################################################

function SemFIML(; observed, specification, kwargs...)
    inverses = broadcast(x -> zeros(x, x), pattern_nobs_vars(observed))
    choleskys = Array{Cholesky{Float64, Array{Float64, 2}}, 1}(undef, length(inverses))

    n_patterns = size(pattern_rows(observed), 1)
    logdets = zeros(n_patterns)

    imp_mean = zeros.(pattern_nobs_vars(observed))
    meandiff = zeros.(pattern_nobs_vars(observed))

    nobs_vars = nobserved_vars(observed)
    imp_inv = zeros(nobs_vars, nobs_vars)
    mult = similar.(inverses)

    ∇ind = vec(CartesianIndices(Array{Float64}(undef, nobs_vars, nobs_vars)))
    ∇ind =
        [findall(x -> !(x[1] ∈ ind || x[2] ∈ ind), ∇ind) for ind in patterns_not(observed)]

    return SemFIML(
        ExactHessian(),
        inverses,
        choleskys,
        logdets,
        ∇ind,
        imp_mean,
        meandiff,
        imp_inv,
        mult,
        CommutationMatrix(nvars(specification)),
        nothing,
    )
end

############################################################################################
### methods
############################################################################################

function evaluate!(
    objective,
    gradient,
    hessian,
    semfiml::SemFIML,
    implied::SemImply,
    model::AbstractSemSingle,
    params,
)
    isnothing(hessian) || error("Hessian not implemented for FIML")

    if !check_fiml(semfiml, model)
        isnothing(objective) || (objective = non_posdef_return(params))
        isnothing(gradient) || fill!(gradient, 1)
        return objective
    end

    prepare_SemFIML!(semfiml, model)

    scale = inv(nsamples(observed(model)))
    obs_rows = pattern_rows(observed(model))
    isnothing(objective) || (objective = scale * F_FIML(obs_rows, semfiml, model, params))
    isnothing(gradient) ||
        (∇F_FIML!(gradient, obs_rows, semfiml, model); gradient .*= scale)

    return objective
end

############################################################################################
### Recommended methods
############################################################################################

update_observed(lossfun::SemFIML, observed::SemObserved; kwargs...) =
    SemFIML(; observed = observed, kwargs...)

############################################################################################
### additional functions
############################################################################################

function F_one_pattern(meandiff, inverse, obs_cov, logdet, N)
    F = logdet + dot(meandiff, inverse, meandiff)
    if N > one(N)
        F += dot(obs_cov, inverse)
    end
    return F * N
end

function ∇F_one_pattern(μ_diff, Σ⁻¹, S, pattern, ∇ind, N, Jμ, JΣ, model)
    diff⨉inv = μ_diff' * Σ⁻¹

    if N > one(N)
        JΣ[∇ind] .+= N * vec(Σ⁻¹ * (I - S * Σ⁻¹ - μ_diff * diff⨉inv))
        @. Jμ[pattern] += (N * 2 * diff⨉inv)'

    else
        JΣ[∇ind] .+= vec(Σ⁻¹ * (I - μ_diff * diff⨉inv))
        @. Jμ[pattern] += (2 * diff⨉inv)'
    end
end

function ∇F_fiml_outer!(G, JΣ, Jμ, imply::SemImplySymbolic, model, semfiml)
    mul!(G, imply.∇Σ', JΣ) # should be transposed
    mul!(G, imply.∇μ', Jμ, -1, 1)
end

function ∇F_fiml_outer!(G, JΣ, Jμ, imply, model, semfiml)
    Iₙ = sparse(1.0I, size(imply.A)...)
    P = kron(imply.F⨉I_A⁻¹, imply.F⨉I_A⁻¹)
    Q = kron(imply.S * imply.I_A⁻¹', Iₙ)
    Q .+= semfiml.commutator * Q

    ∇Σ = P * (imply.∇S + Q * imply.∇A)

    ∇μ = imply.F⨉I_A⁻¹ * imply.∇M + kron((imply.I_A⁻¹ * imply.M)', imply.F⨉I_A⁻¹) * imply.∇A

    mul!(G, ∇Σ', JΣ) # actually transposed
    mul!(G, ∇μ', Jμ, -1, 1)
end

function F_FIML(rows, semfiml, model, params)
    F = zero(eltype(params))
    for i in 1:size(rows, 1)
        F += F_one_pattern(
            semfiml.meandiff[i],
            semfiml.inverses[i],
            obs_cov(observed(model))[i],
            semfiml.logdets[i],
            pattern_nsamples(observed(model))[i],
        )
    end
    return F
end

function ∇F_FIML!(G, rows, semfiml, model)
    Jμ = zeros(nobserved_vars(model))
    JΣ = zeros(nobserved_vars(model)^2)

    for i in 1:size(rows, 1)
        ∇F_one_pattern(
            semfiml.meandiff[i],
            semfiml.inverses[i],
            obs_cov(observed(model))[i],
            patterns(observed(model))[i],
            semfiml.∇ind[i],
            pattern_nsamples(observed(model))[i],
            Jμ,
            JΣ,
            model,
        )
    end
    return ∇F_fiml_outer!(G, JΣ, Jμ, imply(model), model, semfiml)
end

function prepare_SemFIML!(semfiml, model)
    copy_per_pattern!(semfiml, model)
    batch_cholesky!(semfiml, model)
    #batch_sym_inv_update!(semfiml, model)
    batch_inv!(semfiml, model)
    for i in 1:size(pattern_nsamples(observed(model)), 1)
        semfiml.meandiff[i] .= obs_mean(observed(model))[i] - semfiml.imp_mean[i]
    end
end

function copy_per_pattern!(inverses, source_inverses, means, source_means, patterns)
    @views for i in 1:size(patterns, 1)
        inverses[i] .= source_inverses[patterns[i], patterns[i]]
    end

    @views for i in 1:size(patterns, 1)
        means[i] .= source_means[patterns[i]]
    end
end

copy_per_pattern!(semfiml, model::M where {M <: AbstractSem}) = copy_per_pattern!(
    semfiml.inverses,
    imply(model).Σ,
    semfiml.imp_mean,
    imply(model).μ,
    patterns(observed(model)),
)

function batch_cholesky!(semfiml, model)
    for i in 1:size(semfiml.inverses, 1)
        semfiml.choleskys[i] = cholesky!(Symmetric(semfiml.inverses[i]))
        semfiml.logdets[i] = logdet(semfiml.choleskys[i])
    end
    return true
end

function check_fiml(semfiml, model)
    copyto!(semfiml.imp_inv, imply(model).Σ)
    a = cholesky!(Symmetric(semfiml.imp_inv); check = false)
    return isposdef(a)
end
