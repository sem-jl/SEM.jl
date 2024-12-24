"""
    RMSEA(sem_fit::SemFit)

Return the RMSEA.
"""
RMSEA(fit::SemFit) = RMSEA(fit, fit.model)

RMSEA(fit::SemFit, model::AbstractSem) =
    sqrt(nsem_terms(model)) * RMSEA(df(fit), χ²(fit), nsamples(fit))

RMSEA(df::Number, chi2::Number, nsamples::Number) =
    sqrt(max((chi2 - df) / (nsamples * df), 0.0))
