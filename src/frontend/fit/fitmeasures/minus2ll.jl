"""
    minus2ll(sem_fit::SemFit)

Return the negative 2* log likelihood.
"""
function minus2ll end

############################################################################################
# Single Models
############################################################################################

# SemFit splices loss functions ------------------------------------------------------------
minus2ll(
    sem_fit::SemFit{Mi, So, St, Mo, O} where {Mi, So, St, Mo <: AbstractSemSingle, O},
) = minus2ll(
    sem_fit,
    sem_fit.model.observed,
    sem_fit.model.implied,
    sem_fit.model.loss.functions...,
)

minus2ll(sem_fit::SemFit, obs, imp, args...) = minus2ll(sem_fit.minimum, obs, imp, args...)

# SemML ------------------------------------------------------------------------------------
minus2ll(minimum::Number, obs, imp::Union{RAM, RAMSymbolic}, loss_ml::SemML) =
    nsamples(obs) * (minimum + log(2π) * nobserved_vars(obs))

# WLS --------------------------------------------------------------------------------------
minus2ll(minimum::Number, obs, imp::Union{RAM, RAMSymbolic}, loss_ml::SemWLS) = missing

# compute likelihood for missing data - H0 -------------------------------------------------
# -2ll = (∑ log(2π)*(nᵢ + mᵢ)) + F*n
function minus2ll(minimum::Number, observed, imp::Union{RAM, RAMSymbolic}, loss_ml::SemFIML)
    F = minimum * nsamples(observed)
    F += log(2π) * sum(pat -> nsamples(pat) * nmeasured_vars(pat), observed.patterns)
    return F
end

# compute likelihood for missing data - H1 -------------------------------------------------
# -2ll =  ∑ log(2π)*(nᵢ + mᵢ) + ln(Σᵢ) + (mᵢ - μᵢ)ᵀ Σᵢ⁻¹ (mᵢ - μᵢ)) + tr(SᵢΣᵢ)
function minus2ll(observed::SemObservedMissing)
    # fit EM-based mean and cov if not yet fitted
    # FIXME EM could be very computationally expensive
    observed.em_model.fitted || em_mvn(observed)

    Σ = observed.em_model.Σ
    μ = observed.em_model.μ

    F = sum(observed.patterns) do pat
        # implied covariance/mean
        Σᵢ = Σ[pat.measured_mask, pat.measured_mask]
        Σᵢ_chol = cholesky!(Σᵢ)
        ld = logdet(Σᵢ_chol)
        Σᵢ⁻¹ = LinearAlgebra.inv!(Σᵢ_chol)
        meandiffᵢ = pat.measured_mean - μ[pat.measured_mask]

        F_one_pattern(meandiffᵢ, Σᵢ⁻¹, pat.measured_cov, ld, nsamples(pat))
    end

    F += log(2π) * sum(pat -> nsamples(pat) * nmeasured_vars(pat), observed.patterns)

    return F
end

############################################################################################
# Collection
############################################################################################

minus2ll(minimum, model::AbstractSemSingle) =
    minus2ll(minimum, model.observed, model.implied, model.loss.functions...)

function minus2ll(
    sem_fit::SemFit{Mi, So, St, Mo, O} where {Mi, So, St, Mo <: SemEnsemble, O},
)
    m2ll = 0.0
    for sem in sem_fit.model.sems
        minimum = objective!(sem, sem_fit.solution)
        m2ll += minus2ll(minimum, sem)
    end
    return m2ll
end
