"""
    df(sem_fit::SemFit)
    df(model::AbstractSem)

Return the degrees of freedom.
"""
function df end

df(sem_fit::SemFit) = df(sem_fit.model)

df(model::AbstractSem) = n_dp(model) - nparams(model)

function n_dp(model::AbstractSemSingle)
    nvars = nobserved_vars(model)
    ndp = 0.5(nvars^2 + nvars)
    if !isnothing(model.imply.μ)
        ndp += nvars
    end
    return ndp
end

n_dp(model::SemEnsemble) = sum(n_dp.(model.sems))
