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

# fallback method
function evaluate!(obj, grad, hess, loss::AbstractLoss, params)
    isnothing(obj) || (obj = objective(loss, params))
    isnothing(grad) || copyto!(grad, gradient(loss, params))
    isnothing(hess) || copyto!(hess, hessian(loss, params))
    return obj
end

evaluate!(obj, grad, hess, term::LossTerm, params) =
    evaluate!(obj, grad, hess, loss(term), params)

# fallback method for SemImplied that calls update_xxx!() methods
function update!(targets::EvaluationTargets, implied::SemImplied, params)
    is_objective_required(targets) && update_objective!(implied, params)
    is_gradient_required(targets) && update_gradient!(implied, params)
    is_hessian_required(targets) && update_hessian!(implied, params)
end

const AbstractSemOrLoss = Union{AbstractSem, AbstractLoss}

# guess objective type
objective_type(model::AbstractSemOrLoss, params::Any) = Float64
objective_type(model::AbstractSemOrLoss, params::AbstractVector{T}) where {T <: Number} = T
objective_zero(model::AbstractSemOrLoss, params::Any) = zero(objective_type(model, params))

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

evaluate!(objective, gradient, hessian, model::AbstractSem, params) =
    error("evaluate!() for $(typeof(model)) is not implemented")

############################################################################################
# methods for Sem
############################################################################################

function evaluate!(objective, gradient, hessian, model::Sem, params)
    # reset output
    isnothing(objective) || (objective = objective_zero(objective, gradient, hessian))
    isnothing(gradient) || fill!(gradient, zero(eltype(gradient)))
    isnothing(hessian) || fill!(hessian, zero(eltype(hessian)))

    # gradient and hessian for individual terms
    t_grad = isnothing(gradient) ? nothing : similar(gradient)
    t_hess = isnothing(hessian) ? nothing : similar(hessian)

    # update implied states of all SemLoss terms before term calculation loop
    # to make sure all terms use updated implied states
    targets = EvaluationTargets(objective, gradient, hessian)
    for term in loss_terms(model)
        issemloss(term) && update!(targets, implied(term), params)
    end

    for term in loss_terms(model)
        t_obj = evaluate!(objective, t_grad, t_hess, term, params)
        #@show nameof(typeof(term)) t_obj
        objective = accumulate_loss!(
            objective,
            gradient,
            hessian,
            weight(term),
            t_obj,
            t_grad,
            t_hess,
        )
    end
    return objective
end

# internal function to accumulate loss objective, gradient and hessian
function accumulate_loss!(
    total_objective,
    total_gradient,
    total_hessian,
    weight::Nothing,
    objective,
    gradient,
    hessian,
)
    isnothing(total_gradient) || (total_gradient .+= gradient)
    isnothing(total_hessian) || (total_hessian .+= hessian)
    return isnothing(total_objective) ? total_objective : (total_objective + objective)
end

function accumulate_loss!(
    total_objective,
    total_gradient,
    total_hessian,
    weight::Number,
    objective,
    gradient,
    hessian,
)
    isnothing(total_gradient) || axpy!(weight, gradient, total_gradient)
    isnothing(total_hessian) || axpy!(weight, hessian, total_hessian)
    return isnothing(total_objective) ? total_objective :
           (total_objective + weight * objective)
end

############################################################################################
# methods for SemFiniteDiff (approximate gradient and hessian with finite differences of objective)
############################################################################################

# evaluate!() wrapper that does some housekeeping, if necessary
_evaluate!(args...) = evaluate!(args...)

# update implied state, its gradient and hessian
function _evaluate!(objective, gradient, hessian, loss::SemLoss, params)
    # note that any other Sem loss terms that are dependent on implied
    # should be enumerated after the SemLoss term
    # otherwise they would be using outdated implied state
    update!(EvaluationTargets(objective, gradient, hessian), implied(loss), params)
    return evaluate!(objective, gradient, hessian, loss, params)
end

objective(model::AbstractSemOrLoss, params) =
    _evaluate!(objective_zero(model, params), nothing, nothing, model, params)

# throw an error by default if gradient! and hessian! are not implemented

#= gradient!(model::AbstractSemOrLoss, par, model) =
    throw(ArgumentError("gradient for $(nameof(typeof(model))) is not available"))

hessian!(model::AbstractSemOrLoss, par, model) =
    throw(ArgumentError("hessian for $(nameof(typeof(model))) is not available")) =#

############################################################################################
# Documentation
############################################################################################
"""
    objective!(model::AbstractSem, params)

Returns the objective value at `params`.
The model object can be modified.

# Implementation
To implement a new `SemImplied` or `SemLossFunction` subtype, you need to add a method for
    objective!(newtype::MyNewType, params, model::AbstractSemSingle)

To implement a new `AbstractSem` subtype, you need to add a method for
    objective!(model::MyNewType, params)
"""
function objective! end

"""
    gradient!(gradient, model::AbstractSem, params)

Writes the gradient value at `params` to `gradient`.

# Implementation
To implement a new `SemImplied` or `SemLossFunction` type, you can add a method for
    gradient!(newtype::MyNewType, params, model::AbstractSemSingle)

To implement a new `AbstractSem` subtype, you can add a method for
    gradient!(gradient, model::MyNewType, params)
"""
function gradient! end

"""
    hessian!(hessian, model::AbstractSem, params)

Writes the hessian value at `params` to `hessian`.

# Implementation
To implement a new `SemImplied` or `SemLossFunction` type, you can add a method for
    hessian!(newtype::MyNewType, params, model::AbstractSemSingle)

To implement a new `AbstractSem` subtype, you can add a method for
    hessian!(hessian, model::MyNewType, params)
"""
function hessian! end

objective!(model::AbstractSem, params) =
    _evaluate!(objective_zero(model, params), nothing, nothing, model, params)
gradient!(gradient, model::AbstractSem, params) =
    _evaluate!(nothing, gradient, nothing, model, params)
hessian!(hessian, model::AbstractSem, params) =
    _evaluate!(nothing, nothing, hessian, model, params)
objective_gradient!(gradient, model::AbstractSem, params) =
    _evaluate!(objective_zero(model, params), gradient, nothing, model, params)
objective_hessian!(hessian, model::AbstractSem, params) =
    _evaluate!(objective_zero(model, params), nothing, hessian, model, params)
gradient_hessian!(gradient, hessian, model::AbstractSem, params) =
    _evaluate!(nothing, gradient, hessian, model, params)
objective_gradient_hessian!(gradient, hessian, model::AbstractSem, params) =
    _evaluate!(objective_zero(model, params), gradient, hessian, model, params)
