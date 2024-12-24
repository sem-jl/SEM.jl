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

prepare!(fiml::SemFIMLPattern, pat::SemObservedMissingPattern, implied::SemImplied) =
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

    SemFIML(observed::SemObservedMissing, implied::SemImplied)

# Arguments
- `observed::SemObservedMissing`: the observed part of the model
- `implied::SemImplied`: [`SemImplied`](@ref) instance

# Examples
```julia
my_fiml = SemFIML(my_observed, my_implied)
```

# Interfaces
Analytic gradients are available.

# Extended help
## Implementation
Subtype of `SemLossFunction`.
"""
struct SemFIML{O, I, T, W} <: SemLoss{O, I}
    hessianeval::ExactHessian

    observed::O
    implied::I
    patterns::Vector{SemFIMLPattern{T}}

    imp_inv::Matrix{T}  # implied inverse

    commutator::CommutationMatrix

    interaction::W
end

############################################################################################
### Constructors
############################################################################################

function SemFIML(observed::SemObservedMissing, implied::SemImplied; kwargs...)
    return SemFIML(
        ExactHessian(),
        observed,
        implied,
        [SemFIMLPattern(pat) for pat in observed.patterns],
        zeros(nobserved_vars(observed), nobserved_vars(observed)),
        CommutationMatrix(nvars(implied)),
        nothing,
    )
end

############################################################################################
### methods
############################################################################################

function evaluate!(objective, gradient, hessian, fiml::SemFIML, params)
    isnothing(hessian) || error("Hessian not implemented for FIML")

    implied = SEM.implied(fiml)
    observed = SEM.observed(fiml)

    copyto!(fiml.imp_inv, implied.Σ)
    Σ_chol = cholesky!(Symmetric(fiml.imp_inv); check = false)

    if !isposdef(Σ_chol)
        isnothing(objective) || (objective = non_posdef_return(params))
        isnothing(gradient) || fill!(gradient, 1)
        return objective
    end

    @inbounds for (pat_fiml, pat) in zip(fiml.patterns, observed.patterns)
        prepare!(pat_fiml, pat, implied)
    end

    scale = inv(nsamples(observed))
    isnothing(objective) || (objective = scale * F_FIML(eltype(params), fiml))
    isnothing(gradient) || (∇F_FIML!(gradient, fiml); gradient .*= scale)

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

function ∇F_fiml_outer!(G, JΣ, Jμ, fiml::SemFIML{O, I}) where {O, I <: SemImpliedSymbolic}
    mul!(G, fiml.implied.∇Σ', JΣ) # should be transposed
    mul!(G, fiml.implied.∇μ', Jμ, -1, 1)
end

function ∇F_fiml_outer!(G, JΣ, Jμ, fiml::SemFIML)
    implied = fiml.implied

    Iₙ = sparse(1.0I, size(implied.A)...)
    P = kron(implied.F⨉I_A⁻¹, implied.F⨉I_A⁻¹)
    Q = kron(implied.S * implied.I_A⁻¹', Iₙ)
    Q .+= fiml.commutator * Q

    ∇Σ = P * (implied.∇S + Q * implied.∇A)

    ∇μ =
        implied.F⨉I_A⁻¹ * implied.∇M +
        kron((implied.I_A⁻¹ * implied.M)', implied.F⨉I_A⁻¹) * implied.∇A

    mul!(G, ∇Σ', JΣ) # actually transposed
    mul!(G, ∇μ', Jμ, -1, 1)
end

function F_FIML(::Type{T}, fiml::SemFIML) where {T}
    F = zero(T)
    for (pat_fiml, pat) in zip(fiml.patterns, fiml.observed.patterns)
        F += objective(pat_fiml, pat)
    end
    return F
end

function ∇F_FIML!(G, fiml::SemFIML)
    Jμ = zeros(nobserved_vars(fiml))
    JΣ = zeros(nobserved_vars(fiml)^2)

    for (pat_fiml, pat) in zip(fiml.patterns, fiml.observed.patterns)
        gradient!(JΣ, Jμ, pat_fiml, pat)
    end
    ∇F_fiml_outer!(G, JΣ, Jμ, fiml)
end
