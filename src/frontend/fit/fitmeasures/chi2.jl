"""
    χ²(fit::SemFit)

Return the χ² value.
"""
χ²(fit::SemFit) = χ²(fit, fit.model)

function χ²(fit::SemFit, model::AbstractSem)
    terms = sem_terms(model)
    isempty(terms) && return 0.0

    term1 = _unwrap(loss(terms[1]))
    L = typeof(term1).name

    # check that all SemLoss terms are of the same class (ML, FIML, WLS etc), ignore typeparams
    for (i, term) in enumerate(terms)
        lossterm = _unwrap(loss(term))
        @assert lossterm isa SemLoss
        if typeof(_unwrap(lossterm)).name != L
            @error "SemLoss term #$i is $(typeof(_unwrap(lossterm)).name), expected $L. Heterogeneous loss functions are not supported"
        end
    end

    return χ²(typeof(term1), fit, model)
end

χ²(::Type{<:SemWLS}, fit::SemFit, model::AbstractSem) = (nsamples(model) - 1) * fit.minimum

function χ²(::Type{<:SemML}, fit::SemFit, model::AbstractSem)
    G = sum(loss_terms(model)) do term
        if issemloss(term)
            data = observed(term)
            something(weight(term), 1.0) * (logdet(obs_cov(data)) + nobserved_vars(data))
        else
            return 0.0
        end
    end
    return (nsamples(model) - 1) * (fit.minimum - G)
end

function χ²(::Type{<:SemFIML}, fit::SemFit, model::AbstractSem)
    ll_H0 = minus2ll(fit)
    ll_H1 = sum(minus2ll ∘ observed, sem_terms(model))
    return ll_H0 - ll_H1
end
