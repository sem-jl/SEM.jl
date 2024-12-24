"""
    minus2ll(fit::SemFit)

Return the negative 2* log likelihood.
"""
function minus2ll end

############################################################################################
# Single Models
############################################################################################

minus2ll(fit::SemFit) = minus2ll(fit.model, fit)

function minus2ll(term::SemLoss, fit::SemFit)
    minimum = objective(term, fit.solution)
    return minus2ll(term, minimum)
end

minus2ll(term::SemML, minimum::Number) =
    nsamples(term) * (minimum + log(2π) * nobserved_vars(term))

# WLS --------------------------------------------------------------------------------------
minus2ll(term::SemWLS, minimum::Number) = missing

# compute likelihood for missing data - H0 -------------------------------------------------
# -2ll = (∑ log(2π)*(nᵢ + mᵢ)) + F*n
function minus2ll(term::SemFIML, minimum::Number)
    obs = observed(term)::SemObservedMissing
    F = minimum * nsamples(obs)
    F += log(2π) * sum(pat -> nsamples(pat) * nmeasured_vars(pat), obs.patterns)
    return F
end

# compute likelihood for missing data - H1 -------------------------------------------------
# -2ll =  ∑ log(2π)*(nᵢ + mᵢ) + ln(Σᵢ) + (mᵢ - μᵢ)ᵀ Σᵢ⁻¹ (mᵢ - μᵢ)) + tr(SᵢΣᵢ)
function minus2ll(observed::SemObservedMissing)
    Σ, μ = obs_cov(observed), obs_mean(observed)

    # FIXME: this code is duplicate to objective(fiml, ...)
    F = sum(observed.patterns) do pat
        # implied covariance/mean
        Σᵢ = Symmetric(Σ[pat.measured_mask, pat.measured_mask])
        Σᵢ_chol = cholesky!(Σᵢ)
        ld = logdet(Σᵢ_chol)
        Σᵢ⁻¹ = LinearAlgebra.inv!(Σᵢ_chol)
        μ_diffᵢ = pat.measured_mean - μ[pat.measured_mask]

        F_pat = ld + dot(μ_diffᵢ, Σᵢ⁻¹, μ_diffᵢ)
        if nsamples(pat) > 1
            F_pat += dot(pat.measured_cov, Σᵢ⁻¹)
        end
        F_pat
    end

    F += log(2π) * sum(pat -> nsamples(pat) * nmeasured_vars(pat), observed.patterns)

    return F
end

minus2ll(model::AbstractSem, fit::SemFit) =
    sum(Base.Fix2(minus2ll, fit) ∘ _unwrap ∘ loss, sem_terms(model))
