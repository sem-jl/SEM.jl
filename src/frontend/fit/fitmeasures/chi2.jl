"""
    χ²(fit::SemFit)

Return the χ² value.
"""
χ²(fit::SemFit) = χ²(fit, fit.model)

############################################################################################
# Single Models
############################################################################################

χ²(fit::SemFit, model::AbstractSemSingle) =
    sum(loss -> χ²(loss, fit, model), model.loss.functions)

# RAM + SemML
χ²(lossfun::SemML, fit::SemFit, model::AbstractSemSingle) =
    (nsamples(fit) - 1) *
    (fit.minimum - logdet(obs_cov(observed(model))) - nobserved_vars(observed(model)))

# bollen, p. 115, only correct for GLS weight matrix
χ²(lossfun::SemWLS, fit::SemFit, model::AbstractSemSingle) =
    (nsamples(fit) - 1) * fit.minimum

# FIML
function χ²(lossfun::SemFIML, fit::SemFit, model::AbstractSemSingle)
    ll_H0 = minus2ll(fit)
    ll_H1 = minus2ll(observed(model))
    return ll_H0 - ll_H1
end

############################################################################################
# Collections
############################################################################################

function χ²(fit::SemFit, models::SemEnsemble)
    isempty(models.sems) && return 0.0

    lossfun = models.sems[1].loss.functions[1]
    # check that all models use the same single loss function
    L = typeof(lossfun)
    for (i, sem) in enumerate(models.sems)
        if length(sem.loss.functions) > 1
            @error "Model for group #$i has $(length(sem.loss.functions)) loss functions. Only the single one is supported"
        end
        cur_lossfun = sem.loss.functions[1]
        if !isa(cur_lossfun, L)
            @error "Loss function for group #$i model is $(typeof(cur_lossfun)), expected $L. Heterogeneous loss functions are not supported"
        end
    end

    return χ²(lossfun, fit, models)
end

function χ²(lossfun::SemWLS, fit::SemFit, models::SemEnsemble)
    return (nsamples(models) - 1) * fit.minimum
end

function χ²(lossfun::SemML, fit::SemFit, models::SemEnsemble)
    G = sum(zip(models.weights, models.sems)) do (w, model)
        data = observed(model)
        w * (logdet(obs_cov(data)) + nobserved_vars(data))
    end
    return (nsamples(models) - 1) * (fit.minimum - G)
end

function χ²(lossfun::SemFIML, fit::SemFit, models::SemEnsemble)
    ll_H0 = minus2ll(fit)
    ll_H1 = sum(minus2ll ∘ observed, models.sems)
    return ll_H0 - ll_H1
end
