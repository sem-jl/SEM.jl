"""
    χ²(sem_fit::SemFit)

Return the χ² value.
"""
function χ² end

############################################################################################
# Single Models
############################################################################################

# SemFit splices loss functions ------------------------------------------------------------
χ²(sem_fit::SemFit{Mi, So, St, Mo, O} where {Mi, So, St, Mo <: AbstractSemSingle, O}) = χ²(
    sem_fit,
    sem_fit.model.observed,
    sem_fit.model.imply,
    sem_fit.model.optimizer,
    sem_fit.model.loss.functions...,
)

# RAM + SemML
χ²(sem_fit::SemFit, observed, imp::Union{RAM, RAMSymbolic}, optimizer, loss_ml::SemML) =
    (nsamples(sem_fit) - 1) *
    (sem_fit.minimum - logdet(observed.obs_cov) - nobserved_vars(observed))

# bollen, p. 115, only correct for GLS weight matrix
χ²(sem_fit::SemFit, observed, imp::Union{RAM, RAMSymbolic}, optimizer, loss_ml::SemWLS) =
    (nsamples(sem_fit) - 1) * sem_fit.minimum

# FIML
function χ²(sem_fit::SemFit, observed::SemObservedMissing, imp, optimizer, loss_ml::SemFIML)
    ll_H0 = minus2ll(sem_fit)
    ll_H1 = minus2ll(observed)
    chi2 = ll_H0 - ll_H1
    return chi2
end

############################################################################################
# Collections
############################################################################################

# SemFit splices loss functions ------------------------------------------------------------
χ²(sem_fit::SemFit{Mi, So, St, Mo, O} where {Mi, So, St, Mo <: SemEnsemble, O}) =
    χ²(sem_fit, sem_fit.model, sem_fit.model.sems[1].loss.functions[1])

function χ²(sem_fit::SemFit, model::SemEnsemble, lossfun::L) where {L <: SemWLS}
    check_ensemble_length(model)
    check_lossfun_types(model, L)
    return (nsamples(model) - 1) * sem_fit.minimum
end

function χ²(sem_fit::SemFit, model::SemEnsemble, lossfun::L) where {L <: SemML}
    check_ensemble_length(model)
    check_lossfun_types(model, L)
    F_G = sem_fit.minimum
    F_G -= sum([
        w * (logdet(m.observed.obs_cov) + nobserved_vars(m.observed)) for
        (w, m) in zip(model.weights, model.sems)
    ])
    return (nsamples(model) - 1) * F_G
end

function χ²(sem_fit::SemFit, model::SemEnsemble, lossfun::L) where {L <: SemFIML}
    check_ensemble_length(model)
    check_lossfun_types(model, L)

    ll_H0 = minus2ll(sem_fit)
    ll_H1 = sum(minus2ll.(observed.(models(model))))
    chi2 = ll_H0 - ll_H1

    return chi2
end

function check_ensemble_length(model)
    for sem in model.sems
        if length(sem.loss.functions) > 1
            @error "A model for one of the groups contains multiple loss functions."
        end
    end
end

function check_lossfun_types(model, type)
    for sem in model.sems
        for lossfun in sem.loss.functions
            if !isa(lossfun, type)
                @error "Your model(s) contain multiple lossfunctions with differing types."
            end
        end
    end
end
