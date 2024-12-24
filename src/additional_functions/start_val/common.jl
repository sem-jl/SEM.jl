
# start values for SEM Models (including ensembles)
function start_values(f, model::AbstractSem; kwargs...)
    start_vals = fill(0.0, nparams(model))

    # initialize parameters using the SEM loss terms
    # (first SEM loss term that sets given parameter to nonzero value)
    for term in loss_terms(model)
        issemloss(term) || continue
        term_start_vals = f(loss(term); kwargs...)
        for (i, val) in enumerate(term_start_vals)
            iszero(val) || (start_vals[i] = val)
        end
    end

    return start_vals
end
