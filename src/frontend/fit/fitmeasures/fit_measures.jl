fit_measures(sem_fit) =
    fit_measures(sem_fit, nparams, df, AIC, BIC, RMSEA, χ², p_value, minus2ll)

function fit_measures(sem_fit, args...)
    measures = Dict{Symbol, Union{Float64, Missing}}()

    for arg in args
        push!(measures, Symbol(arg) => arg(sem_fit))
    end

    return measures
end

"""
    fit_measures(sem_fit, args...)

Return a default set of fit measures or the fit measures passed as `arg...`.
"""
function fit_measures end
