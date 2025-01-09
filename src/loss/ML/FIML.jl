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

function SemFIML(; observed::SemObservedMissing, specification, kwargs...)
    inverses =
        [zeros(nmeasured_vars(pat), nmeasured_vars(pat)) for pat in observed.patterns]
    choleskys = Array{Cholesky{Float64, Array{Float64, 2}}, 1}(undef, length(inverses))

    n_patterns = length(observed.patterns)
    logdets = zeros(n_patterns)

    imp_mean = [zeros(nmeasured_vars(pat)) for pat in observed.patterns]
    meandiff = [zeros(nmeasured_vars(pat)) for pat in observed.patterns]

    nobs_vars = nobserved_vars(observed)
    imp_inv = zeros(nobs_vars, nobs_vars)
    mult = similar.(inverses)

    # generate linear indicies of co-observed variable pairs for each pattern
    Σ_linind = LinearIndices((nobs_vars, nobs_vars))
    ∇ind = map(observed.patterns) do pat
        pat_vars = findall(pat.measured_mask)
        vec(Σ_linind[pat_vars, pat_vars])
    end

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
    implied::SemImplied,
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
    isnothing(objective) ||
        (objective = scale * F_FIML(observed(model), semfiml, model, params))
    isnothing(gradient) ||
        (∇F_FIML!(gradient, observed(model), semfiml, model); gradient .*= scale)

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

function ∇F_one_pattern(μ_diff, Σ⁻¹, S, obs_mask, ∇ind, N, Jμ, JΣ, model)
    diff⨉inv = μ_diff' * Σ⁻¹

    if N > one(N)
        JΣ[∇ind] .+= N * vec(Σ⁻¹ * (I - S * Σ⁻¹ - μ_diff * diff⨉inv))
        @. Jμ[obs_mask] += (N * 2 * diff⨉inv)'

    else
        JΣ[∇ind] .+= vec(Σ⁻¹ * (I - μ_diff * diff⨉inv))
        @. Jμ[obs_mask] += (2 * diff⨉inv)'
    end
end

function ∇F_fiml_outer!(G, JΣ, Jμ, implied::SemImpliedSymbolic, model, semfiml)
    mul!(G, implied.∇Σ', JΣ) # should be transposed
    mul!(G, implied.∇μ', Jμ, -1, 1)
end

function ∇F_fiml_outer!(G, JΣ, Jμ, implied, model, semfiml)
    Iₙ = sparse(1.0I, size(implied.A)...)
    P = kron(implied.F⨉I_A⁻¹, implied.F⨉I_A⁻¹)
    Q = kron(implied.S * implied.I_A⁻¹', Iₙ)
    Q .+= semfiml.commutator * Q

    ∇Σ = P * (implied.∇S + Q * implied.∇A)

    ∇μ = implied.F⨉I_A⁻¹ * implied.∇M + kron((implied.I_A⁻¹ * implied.M)', implied.F⨉I_A⁻¹) * implied.∇A

    mul!(G, ∇Σ', JΣ) # actually transposed
    mul!(G, ∇μ', Jμ, -1, 1)
end

function F_FIML(observed::SemObservedMissing, semfiml, model, params)
    F = zero(eltype(params))
    for (i, pat) in enumerate(observed.patterns)
        F += F_one_pattern(
            semfiml.meandiff[i],
            semfiml.inverses[i],
            pat.measured_cov,
            semfiml.logdets[i],
            nsamples(pat),
        )
    end
    return F
end

function ∇F_FIML!(G, observed::SemObservedMissing, semfiml, model)
    Jμ = zeros(nobserved_vars(model))
    JΣ = zeros(nobserved_vars(model)^2)

    for (i, pat) in enumerate(observed.patterns)
        ∇F_one_pattern(
            semfiml.meandiff[i],
            semfiml.inverses[i],
            pat.measured_cov,
            pat.measured_mask,
            semfiml.∇ind[i],
            nsamples(pat),
            Jμ,
            JΣ,
            model,
        )
    end
    return ∇F_fiml_outer!(G, JΣ, Jμ, implied(model), model, semfiml)
end

function prepare_SemFIML!(semfiml, model)
    copy_per_pattern!(semfiml, model)
    batch_cholesky!(semfiml, model)
    #batch_sym_inv_update!(semfiml, model)
    batch_inv!(semfiml, model)
    for (i, pat) in enumerate(observed(model).patterns)
        semfiml.meandiff[i] .= pat.measured_mean .- semfiml.imp_mean[i]
    end
end

function copy_per_pattern!(fiml::SemFIML, model::AbstractSem)
    Σ = implied(model).Σ
    μ = implied(model).μ
    data = observed(model)
    @inbounds @views for (i, pat) in enumerate(data.patterns)
        fiml.inverses[i] .= Σ[pat.measured_mask, pat.measured_mask]
        fiml.imp_mean[i] .= μ[pat.measured_mask]
    end
end

function batch_cholesky!(semfiml, model)
    for i in 1:size(semfiml.inverses, 1)
        semfiml.choleskys[i] = cholesky!(Symmetric(semfiml.inverses[i]))
        semfiml.logdets[i] = logdet(semfiml.choleskys[i])
    end
    return true
end

function check_fiml(semfiml, model)
    copyto!(semfiml.imp_inv, implied(model).Σ)
    a = cholesky!(Symmetric(semfiml.imp_inv); check = false)
    return isposdef(a)
end
