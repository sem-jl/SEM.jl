"""
    df(sem_fit::SemFit)
    df(model::AbstractSem)

Return the degrees of freedom.
"""
function df end

df(sem_fit::SemFit) = df(sem_fit.model)

df(model::AbstractSem) = n_dp(model) - nparams(model)

# length of Σ and μ (if present)
function n_dp(implied::SemImplied)
    nvars = nobserved_vars(implied)
    ndp = 0.5(nvars^2 + nvars)
    if !isnothing(implied.μ)
        ndp += nvars
    end
    return ndp
end

n_dp(term::SemLoss) = n_dp(implied(term))

n_dp(model::AbstractSem) = sum(n_dp ∘ loss, sem_terms(model))
