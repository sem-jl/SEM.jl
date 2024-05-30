############################################################################################
### Types
############################################################################################

# state of SemFIML for a specific missing pattern (`SemObservedMissingPattern` type)
struct SemFIMLPattern{T}
    ∇ind::Vector{Int}   # indices of co-observed variable pairs
    Σ⁻¹::Matrix{T}      # preallocated inverse of implied cov
    logdet::Ref{T}      # logdet of implied cov
    μ_diff::Vector{T}   # implied mean difference
end

# allocate arrays for pattern FIML
function SemFIMLPattern(pat::SemObservedMissingPattern)
    nobs_vars = nobserved_vars(pat)
    nmeas_vars = nmeasured_vars(pat)

    # generate linear indicies of co-observed variable pairs for each pattern
    Σ_linind = LinearIndices((nobs_vars, nobs_vars))
    pat_vars = findall(pat.measured_mask)
    ∇ind = vec(Σ_linind[pat_vars, pat_vars]

    return SemFIMLPattern(∇ind, zeros(nmeas_vars, nmeas_vars), Ref(NaN), zeros(nmeas_vars))
end

function prepare!(fiml::SemFIMLPattern, pat::SemObservedMissingPattern, Σ::AbstractMatrix, μ::AbstractVector)
    @inbounds @. @views begin
        fiml.Σ⁻¹ = Σ[pat.measured_mask, pat.measured_mask]
        fiml.μ_diff = pat.measured_mean - μ[pat.measured_mask]
    end
    Σ_chol = cholesky!(Symmetric(fiml.Σ⁻¹))
    fiml.logdet[] = logdet(Σ_chol)
    LinearAlgebra.inv!(Σ_chol) # updates fiml.Σ⁻¹
    return fiml
end

prepare!(fiml::SemFIMLPattern, pat::SemObservedMissingPattern, implied::SemImply) =
	prepare!(fiml, pat, implied.Σ, implied.μ)

function objective(fiml::SemFIMLPattern{T}, pat::SemObservedMissingPattern) where {T}
    F = fiml.logdet[] + dot(fiml.μ_diff, fiml.Σ⁻¹, fiml.μ_diff)
    if nsamples(pat) > 1
        F += dot(pat.measured_cov, fiml.Σ⁻¹)
        F *= nsamples(pat)
    end
    return F
end

function gradient!(JΣ, Jμ, fiml::SemFIMLPattern, pat::SemObservedMissingPattern)
    Σ⁻¹ = Symmetric(fiml.Σ⁻¹)
    μ_diff⨉Σ⁻¹ = fiml.μ_diff' * Σ⁻¹
    if nsamples(pat) > 1
        JΣ_pat = Σ⁻¹ * (I - pat.measured_cov * Σ⁻¹ - fiml.μ_diff * μ_diff⨉Σ⁻¹)
        JΣ_pat .*= nsamples(pat)
    else
        JΣ_pat = Σ⁻¹ * (I - fiml.μ_diff * μ_diff⨉Σ⁻¹)
    end
    @inbounds vec(JΣ)[fiml.∇ind] .+= vec(JΣ_pat)

    lmul!(2 * nsamples(pat), μ_diff⨉Σ⁻¹)
    @inbounds Jμ[pat.measured_mask] .+= μ_diff⨉Σ⁻¹'
    return nothing
end

"""
Full information maximum likelihood estimation. Can handle observed data with missings.

# Constructor

    SemFIML(; observed, imply, kwargs...)

# Arguments
- `observed::SemObservedMissing`: the observed part of the model
- `imply::SemImply`: the implied part of the model

# Examples
```julia
my_fiml = SemFIML(observed = my_observed, imply = my_implied)
```

# Interfaces
Analytic gradients are available.

# Extended help
## Implementation
Subtype of `SemLossFunction`.
"""
struct SemFIML{T, W} <: SemLossFunction
    hessianeval::ExactHessian
    patterns::Vector{SemFIMLPattern{T}}

    imp_inv::Matrix{T}  # implied inverse

    commutator::CommutationMatrix

    interaction::W
end

############################################################################################
### Constructors
############################################################################################

function SemFIML(;
    observed::SemObservedMissing,
    imply::SemImply,
    kwargs...,
)
    return SemFIML(
        ExactHessian(),
        [SemFIMLPattern(pat) for pat in observed.patterns],
        zeros(nobserved_vars(observed), nobserved_vars(observed)),
        CommutationMatrix(nvars(imply)),
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
    fiml::SemFIML,
    implied::SemImply,
    model::AbstractSemSingle,
    params,
)
    isnothing(hessian) || error("Hessian not implemented for FIML")

    if !check(fiml, model)
        isnothing(objective) || (objective = non_posdef_return(params))
        isnothing(gradient) || fill!(gradient, 1)
        return objective
    end

    prepare!(fiml, model)

    scale = inv(nsamples(observed(model)))
    isnothing(objective) ||
        (objective = scale * F_FIML(eltype(params), fiml, observed(model), model))
    isnothing(gradient) ||
        (∇F_FIML!(gradient, fiml, observed(model), model); gradient .*= scale)

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

function prepare!(fiml::SemFIML, Σ::AbstractMatrix, μ::AbstractVector)
    data = observed(model)::SemObservedMissing
    @inbounds for (pat_fiml, pat) in zip(fiml.patterns, data.patterns)
        prepare!(pat_fiml, pat, Σ, μ)
    end
    #batch_sym_inv_update!(fiml, model)
end

prepare!(fiml::SemFIML, model::AbstractSemSingle) =
    prepare!(fiml, imply(model).Σ, imply(model).μ)

function check(fiml::SemFIML, model::AbstractSemSingle)
    copyto!(fiml.imp_inv, imply(model).Σ)
    a = cholesky!(Symmetric(fiml.imp_inv); check = false)
    return isposdef(a)
end

function ∇F_fiml_outer!(G, JΣ, Jμ, fiml::SemFIML, imply::SemImplySymbolic, model)
    mul!(G, imply.∇Σ', JΣ) # should be transposed
    mul!(G, imply.∇μ', Jμ, -1, 1)
end

function ∇F_fiml_outer!(G, JΣ, Jμ, fiml::SemFIML, imply, model)
    Iₙ = sparse(1.0I, size(imply.A)...)
    P = kron(imply.F⨉I_A⁻¹, imply.F⨉I_A⁻¹)
    Q = kron(imply.S * imply.I_A⁻¹', Iₙ)
    Q .+= fiml.commutator * Q

    ∇Σ = P * (imply.∇S + Q * imply.∇A)

    ∇μ = imply.F⨉I_A⁻¹ * imply.∇M + kron((imply.I_A⁻¹ * imply.M)', imply.F⨉I_A⁻¹) * imply.∇A

    mul!(G, ∇Σ', JΣ) # actually transposed
    mul!(G, ∇μ', Jμ, -1, 1)
end

function F_FIML(
    ::Type{T},
    fiml::SemFIML,
    observed::SemObservedMissing,
    model::AbstractSemSingle,
) where {T}
    F = zero(T)
    for (pat_fiml, pat) in zip(fiml.patterns, observed.patterns)
        F += objective(pat_fiml, pat)
    end
    return F
end

function ∇F_FIML!(G, fiml::SemFIML, observed::SemObservedMissing, model::AbstractSemSingle)
    Jμ = zeros(nobserved_vars(model))
    JΣ = zeros(nobserved_vars(model)^2)

    for (pat_fiml, pat) in zip(fiml.patterns, observed.patterns)
        gradient!(JΣ, Jμ, pat_fiml, pat)
    end
    ∇F_fiml_outer!(G, JΣ, Jμ, fiml, imply(model), model)
end
