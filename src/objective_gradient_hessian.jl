"Specifies whether objective (O), gradient (G) or hessian (H) evaluation is required"
struct EvaluationTargets{O, G, H} end

EvaluationTargets(objective, gradient, hessian) =
    EvaluationTargets{!isnothing(objective), !isnothing(gradient), !isnothing(hessian)}()

# convenience methods to check type params
is_objective_required(::EvaluationTargets{O}) where {O} = O
is_gradient_required(::EvaluationTargets{<:Any, G}) where {G} = G
is_hessian_required(::EvaluationTargets{<:Any, <:Any, H}) where {H} = H

# return the tuple of the required results
(::EvaluationTargets{true, false, false})(objective, gradient, hessian) = objective
(::EvaluationTargets{false, true, false})(objective, gradient, hessian) = gradient
(::EvaluationTargets{false, false, true})(objective, gradient, hessian) = hessian
(::EvaluationTargets{true, true, false})(objective, gradient, hessian) =
    (objective, gradient)
(::EvaluationTargets{true, false, true})(objective, gradient, hessian) =
    (objective, hessian)
(::EvaluationTargets{false, true, true})(objective, gradient, hessian) = (gradient, hessian)
(::EvaluationTargets{true, true, true})(objective, gradient, hessian) =
    (objective, gradient, hessian)

(targets::EvaluationTargets)(arg_tuple::Tuple) = targets(arg_tuple...)

# dispatch on SemImply
evaluate!(objective, gradient, hessian, loss::SemLossFunction, model::AbstractSem, params) =
    evaluate!(objective, gradient, hessian, loss, imply(model), model, params)

# fallback method
function evaluate!(obj, grad, hess, loss::SemLossFunction, imply::SemImply, model, params)
    isnothing(obj) || (obj = objective(loss, imply, model, params))
    isnothing(grad) || copyto!(grad, gradient(loss, imply, model, params))
    isnothing(hess) || copyto!(hess, hessian(loss, imply, model, params))
    return obj
end

# fallback methods
objective(f::SemLossFunction, imply::SemImply, model, params) = objective(f, model, params)
gradient(f::SemLossFunction, imply::SemImply, model, params) = gradient(f, model, params)
hessian(f::SemLossFunction, imply::SemImply, model, params) = hessian(f, model, params)

# fallback method for SemImply that calls update_xxx!() methods
function update!(targets::EvaluationTargets, imply::SemImply, model, params)
    is_objective_required(targets) && update_objective!(imply, model, params)
    is_gradient_required(targets) && update_gradient!(imply, model, params)
    is_hessian_required(targets) && update_hessian!(imply, model, params)
end

# guess objective type
objective_type(model::AbstractSem, params::Any) = Float64
objective_type(model::AbstractSem, params::AbstractVector{T}) where {T <: Number} = T
objective_zero(model::AbstractSem, params::Any) = zero(objective_type(model, params))

objective_type(objective::T, gradient, hessian) where {T <: Number} = T
objective_type(
    objective::Nothing,
    gradient::AbstractArray{T},
    hessian,
) where {T <: Number} = T
objective_type(
    objective::Nothing,
    gradient::Nothing,
    hessian::AbstractArray{T},
) where {T <: Number} = T
objective_zero(objective, gradient, hessian) =
    zero(objective_type(objective, gradient, hessian))

############################################################################################
# methods for AbstractSem
############################################################################################

function evaluate!(objective, gradient, hessian, model::AbstractSemSingle, params)
    targets = EvaluationTargets(objective, gradient, hessian)
    # update imply state, its gradient and hessian (if required)
    update!(targets, imply(model), model, params)
    return evaluate!(
        !isnothing(objective) ? zero(objective) : nothing,
        gradient,
        hessian,
        loss(model),
        model,
        params,
    )
end

############################################################################################
# methods for SemFiniteDiff (approximate gradient and hessian with finite differences of objective)
############################################################################################

function evaluate!(objective, gradient, hessian, model::SemFiniteDiff, params)
    function obj(p)
        # recalculate imply state for p
        update!(EvaluationTargets{true, false, false}(), imply(model), model, p)
        evaluate!(
            objective_zero(objective, gradient, hessian),
            nothing,
            nothing,
            loss(model),
            model,
            p,
        )
    end
    isnothing(gradient) || FiniteDiff.finite_difference_gradient!(gradient, obj, params)
    isnothing(hessian) || FiniteDiff.finite_difference_hessian!(hessian, obj, params)
    return !isnothing(objective) ? obj(params) : nothing
end

objective(model::AbstractSem, params) =
    evaluate!(objective_zero(model, params), nothing, nothing, model, params)

############################################################################################
# methods for SemLoss (weighted sum of individual SemLossFunctions)
############################################################################################

function evaluate!(objective, gradient, hessian, loss::SemLoss, model::AbstractSem, params)
    isnothing(objective) || (objective = zero(objective))
    isnothing(gradient) || fill!(gradient, zero(eltype(gradient)))
    isnothing(hessian) || fill!(hessian, zero(eltype(hessian)))
    f_grad = isnothing(gradient) ? nothing : similar(gradient)
    f_hess = isnothing(hessian) ? nothing : similar(hessian)
    for (f, weight) in zip(loss.functions, loss.weights)
        f_obj = evaluate!(objective, f_grad, f_hess, f, model, params)
        isnothing(objective) || (objective += weight * f_obj)
        isnothing(gradient) || (gradient .+= weight * f_grad)
        isnothing(hessian) || (hessian .+= weight * f_hess)
    end
    return objective
end

############################################################################################
# methods for SemEnsemble (weighted sum of individual AbstractSemSingle models)
############################################################################################

function evaluate!(objective, gradient, hessian, ensemble::SemEnsemble, params)
    isnothing(objective) || (objective = zero(objective))
    isnothing(gradient) || fill!(gradient, zero(eltype(gradient)))
    isnothing(hessian) || fill!(hessian, zero(eltype(hessian)))
    sem_grad = isnothing(gradient) ? nothing : similar(gradient)
    sem_hess = isnothing(hessian) ? nothing : similar(hessian)
    for (sem, weight) in zip(ensemble.sems, ensemble.weights)
        sem_obj = evaluate!(objective, sem_grad, sem_hess, sem, params)
        isnothing(objective) || (objective += weight * sem_obj)
        isnothing(gradient) || (gradient .+= weight * sem_grad)
        isnothing(hessian) || (hessian .+= weight * sem_hess)
    end
    return objective
end

# throw an error by default if gradient! and hessian! are not implemented

#= gradient!(lossfun::SemLossFunction, par, model) =
    throw(ArgumentError("gradient for $(typeof(lossfun).name.wrapper) is not available"))

hessian!(lossfun::SemLossFunction, par, model) =
    throw(ArgumentError("hessian for $(typeof(lossfun).name.wrapper) is not available")) =#

############################################################################################
# Documentation
############################################################################################
"""
    objective!(model::AbstractSem, params)

Returns the objective value at `params`.
The model object can be modified.

# Implementation
To implement a new `SemImply` or `SemLossFunction` subtype, you need to add a method for
    objective!(newtype::MyNewType, params, model::AbstractSemSingle)

To implement a new `AbstractSem` subtype, you need to add a method for
    objective!(model::MyNewType, params)
"""
function objective! end

"""
    gradient!(gradient, model::AbstractSem, params)

Writes the gradient value at `params` to `gradient`.

# Implementation
To implement a new `SemImply` or `SemLossFunction` type, you can add a method for
    gradient!(newtype::MyNewType, params, model::AbstractSemSingle)

To implement a new `AbstractSem` subtype, you can add a method for
    gradient!(gradient, model::MyNewType, params)
"""
function gradient! end

"""
    hessian!(hessian, model::AbstractSem, params)

Writes the hessian value at `params` to `hessian`.

# Implementation
To implement a new `SemImply` or `SemLossFunction` type, you can add a method for
    hessian!(newtype::MyNewType, params, model::AbstractSemSingle)

To implement a new `AbstractSem` subtype, you can add a method for
    hessian!(hessian, model::MyNewType, params)
"""
function hessian! end

objective!(model::AbstractSem, params) =
    evaluate!(objective_zero(model, params), nothing, nothing, model, params)
gradient!(gradient, model::AbstractSem, params) =
    evaluate!(nothing, gradient, nothing, model, params)
hessian!(hessian, model::AbstractSem, params) =
    evaluate!(nothing, nothing, hessian, model, params)
objective_gradient!(gradient, model::AbstractSem, params) =
    evaluate!(objective_zero(model, params), gradient, nothing, model, params)
objective_hessian!(hessian, model::AbstractSem, params) =
    evaluate!(objective_zero(model, params), nothing, hessian, model, params)
gradient_hessian!(gradient, hessian, model::AbstractSem, params) =
    evaluate!(nothing, gradient, hessian, model, params)
objective_gradient_hessian!(gradient, hessian, model::AbstractSem, params) =
    evaluate!(objective_zero(model, params), gradient, hessian, model, params)
