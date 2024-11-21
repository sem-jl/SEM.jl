"""
    RMSEA(sem_fit::SemFit)

Return the RMSEA.
"""
function RMSEA end

RMSEA(sem_fit::SemFit{Mi, So, St, Mo, O} where {Mi, So, St, Mo <: AbstractSemSingle, O}) =
    RMSEA(df(sem_fit), χ²(sem_fit), nsamples(sem_fit))

RMSEA(sem_fit::SemFit{Mi, So, St, Mo, O} where {Mi, So, St, Mo <: SemEnsemble, O}) =
    sqrt(length(sem_fit.model.sems)) * RMSEA(df(sem_fit), χ²(sem_fit), nsamples(sem_fit))

function RMSEA(df, chi2, nsamples)
    rmsea = (chi2 - df) / (nsamples * df)
    rmsea > 0 ? nothing : rmsea = 0
    return sqrt(rmsea)
end
