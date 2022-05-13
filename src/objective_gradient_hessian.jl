#####################################################################################################
# methods for AbstractSem
#####################################################################################################

function objective!(model::AbstractSemSingle, parameters)
    objective!(imply(model), parameters, model)
    return objective!(loss(model), parameters, model)
end

#####################################################################################################
# methods for Sem
#####################################################################################################

# pre-allocated gradient and hessian

function gradient!(gradient, model::AbstractSemSingle, parameters)
    gradient!(imply(model), parameters, model)
    gradient!(gradient, loss(model), parameters, model)
end

function hessian!(hessian, model::AbstractSemSingle, parameters)
    hessian!(imply(model), parameters, model)
    hessian!(hessian, loss(model), parameters, model)
end

function objective_gradient!(gradient, model::AbstractSemSingle, parameters)
    objective_gradient!(imply(model), parameters, model)
    objective_gradient!(gradient, loss(model), parameters, model)
end

function objective_hessian!(hessian, model::AbstractSemSingle, parameters)
    objective_hessian!(imply(model), parameters, model)
    objective_hessian!(hessian, loss(model), parameters, model)
end

function gradient_hessian!(gradient, hessian, model::AbstractSemSingle, parameters)
    gradient_hessian!(imply(model), parameters, model)
    gradient_hessian!(gradient, hessian, loss(model), parameters, model)
end

function objective_gradient_hessian!(gradient, hessian, model::AbstractSemSingle, parameters)
    objective_gradient_hessian!(imply(model), parameters, model)
    return objective_gradient_hessian!(gradient, hessian, loss(model), parameters, model)
end

# non-preallocated gradient
function gradient!(model::AbstractSemSingle, parameters)
    gradient = similar(parameters)
    gradient!(gradient, model, parameters)
end

function objective_gradient!(model::AbstractSemSingle, parameters)
    gradient = similar(parameters)
    objective = objective_gradient!(gradient, model, parameters)
    return objective
end

#####################################################################################################
# methods for SemFiniteDiff and SemForwardDiff
#####################################################################################################

# gradient methods call themselves with the additional model.has_gradient argument

gradient!(gradient, model::Union{SemFiniteDiff, SemForwardDiff}, par) = 
    gradient!(gradient, model, par, model.has_gradient)

objective_gradient!(gradient, model::Union{SemFiniteDiff, SemForwardDiff}, par) = 
    objective_gradient!(gradient, model, par, model.has_gradient)

# methods where autodiff takes place - these are specific to the method of automatic differentiation

# FiniteDiff
gradient!(gradient, model::SemFiniteDiff, par, has_gradient::Val{false}) =
    FiniteDiff.finite_difference_gradient!(gradient, x -> objective!(model, x), par)

hessian!(hessian, model::SemFiniteDiff, par) = 
    FiniteDiff.finite_difference_hessian!(hessian, x -> objective!(model, x), par)

# ForwardDiff
gradient!(gradient, model::SemForwardDiff, par, has_gradient::Val{false}) =
    ForwardDiff.gradient!(gradient, x -> objective!(model, x), par)

hessian!(hessian, model::SemForwardDiff, par) = 
    ForwardDiff.hessian!(hessian, x -> objective!(model, x), par)

# gradient!
function gradient!(gradient, model::Union{SemFiniteDiff, SemForwardDiff}, par, has_gradient::Val{true})
    gradient!(imply(model), parameters, model)
    gradient!(gradient, loss(model), parameters, model)
end

# objective_gradient!
function objective_gradient!(gradient, model::Union{SemFiniteDiff, SemForwardDiff}, par, has_gradient::Val{true})
    objective_gradient!(imply(model), parameters, model)
    return objective_gradient!(gradient, loss(model), parameters, model)
end

function objective_gradient!(gradient, model::Union{SemFiniteDiff, SemForwardDiff}, par, has_gradient::Val{false})
    gradient!(gradient, model, par)
    return objective!(model, par)
end

# other methods
function gradient_hessian!(gradient, hessian, model::Union{SemFiniteDiff, SemForwardDiff}, parameters)
    gradient!(gradient, model, parameters)
    hessian!(hessian, model, parameters)
end

function objective_hessian!(hessian, model::Union{SemFiniteDiff, SemForwardDiff}, par)
    hessian!(hessian, model, par)
    return objective!(model, par)
end

function objective_gradient_hessian!(gradient, hessian, model::Union{SemFiniteDiff, SemForwardDiff}, par)
    hessian!(hessian, model, par)
    return objective_gradient!(gradient, model, par)
end

#####################################################################################################
# methods for SemLoss
#####################################################################################################

function objective!(loss::SemLoss, par, model)
    return mapreduce(
        fun_weight -> fun_weight[2]*objective!(fun_weight[1], par, model), 
        +, 
        zip(loss.functions, loss.weights)
        )
end

function gradient!(gradient, loss::SemLoss, par, model)
    for (lossfun, w) in zip(loss.functions, loss.weights)
        gradient .+= w*gradient!(lossfun, par, model)
    end
end

function hessian!(hessian, loss::SemLoss, par, model)
    for (lossfun, w) in zip(loss.functions, loss.weights)
        hessian .+= w*hessian!(lossfun, par, model)
    end
end

function objective_gradient!(gradient, loss::SemLoss, par, model)
    return mapreduce(
        fun_weight -> objective_gradient_wrap_(gradient, fun_weight[1], par, model, fun_weight[2]),
        +, 
        zip(loss.functions, loss.weights)
        )
end

function objective_hessian!(hessian, loss::SemLoss, par, model)
    return mapreduce(
        fun_weight -> objective_hessian_wrap_(hessian, fun_weight[1], par, model, fun_weight[2]),
        +, 
        zip(loss.functions, loss.weights)
        )
end

function gradient_hessian!(gradient, hessian, loss::SemLoss, par, model)
    for (lossfun, w) in zip(loss.functions, loss.weights)
        new_gradient, new_hessian = gradient_hessian!(lossfun, par, model)
        gradient .+= w*new_gradient
        hessian .+= w*new_hessian
    end
end

function objective_gradient_hessian!(gradient, hessian, loss::SemLoss, par, model)
    return mapreduce(
        fun_weight -> objective_gradient_hessian_wrap_(gradient, hessian, fun_weight[1], par, model, fun_weight[2]),
        +, 
        zip(loss.functions, loss.weights)
        )
end

# wrapper to update gradient/hessian and return objective value
function objective_gradient_wrap_(gradient, lossfun, par, model, w)
    new_objective, new_gradient = objective_gradient!(lossfun, par, model)
    gradient .+= w*new_gradient
    return w*new_objective
end

function objective_hessian_wrap_(hessian, lossfun, par, model, w)
    new_objective, new_hessian = objective_hessian!(lossfun, par, model)
    hessian .+= w*new_hessian
    return w*new_objective
end

function objective_gradient_hessian_wrap_(gradient, hessian, lossfun, par, model, w)
    new_objective, new_gradient, new_hessian = objective_gradient_hessian!(lossfun, par, model)
    gradient .+= w*new_gradient
    hessian .+= w*new_hessian
    return w*new_objective
end

