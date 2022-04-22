function test_gradient(model, parameters)
    true_grad = FiniteDiff.finite_difference_gradient(x -> objective!(model, x)[1], parameters)

    # F and G
    gradient(model) .= 0
    model(parameters, true, true, false)
    correct1 = isapprox(gradient(model), true_grad)

    # only G
    gradient(model) .= 0
    model(parameters, false, true, false)
    correct2 = isapprox(gradient(model), true_grad)

    return correct1 & correct2
end

function test_hessian(model, parameters)
    true_hessian = FiniteDiff.finite_difference_hessian(x -> objective!(model, x)[1], parameters)

    # F and H
    hessian(model) .= 0
    model(parameters, true, false, true)
    correct1 = isapprox(hessian(model), true_hessian; rtol = 1/1000)

    # G and H
    hessian(model) .= 0
    model(parameters, false, true, true)
    correct2 = isapprox(hessian(model), true_hessian; rtol = 1/1000)

    # only H
    hessian(model) .= 0
    model(parameters, false, false, true)
    correct3 = isapprox(hessian(model), true_hessian; rtol = 1/1000)
    
    return correct1 & correct2 & correct3
end

fitmeasure_names_ml = Dict(
    :AIC => "aic",
    :BIC => "bic",
    :df => "df",
    :χ² => "chisq",
    :p_value => "pvalue",
    :n_par => "npar",
    :RMSEA => "rmsea",
)

fitmeasure_names_ls = Dict(
    :df => "df",
    :χ² => "chisq",
    :p_value => "pvalue",
    :n_par => "npar",
    :RMSEA => "rmsea",
)

function test_fitmeasures(measures, measures_lav; rtol = 1e-4, atol = 1e-3, fitmeasure_names = fitmeasure_names_ml)
    correct = []
    for key in keys(fitmeasure_names)
        measure = measures[key]
        measure_lav = measures_lav.x[measures_lav.Column1 .==  fitmeasure_names[key]][1]
        push!(correct, isapprox(measure, measure_lav; rtol = rtol, atol = atol))
    end
    return correct
end