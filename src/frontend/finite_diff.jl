_unwrap(wrapper::SemFiniteDiff) = wrapper.model
params(wrapper::SemFiniteDiff) = params(wrapper.model)
loss_terms(wrapper::SemFiniteDiff) = loss_terms(wrapper.model)

FiniteDiffLossWrappers = Union{LossFiniteDiff, SemLossFiniteDiff}

_unwrap(term::AbstractLoss) = term
_unwrap(wrapper::FiniteDiffLossWrappers) = wrapper.loss
implied(wrapper::FiniteDiffLossWrappers) = implied(_unwrap(wrapper))
observed(wrapper::FiniteDiffLossWrappers) = observed(_unwrap(wrapper))

FiniteDiffWrapper(model::AbstractSem) = SemFiniteDiff(model)
FiniteDiffWrapper(loss::AbstractLoss) = LossFiniteDiff(loss)
FiniteDiffWrapper(loss::SemLoss) = SemLossFiniteDiff(loss)

function evaluate!(
    objective,
    gradient,
    hessian,
    sem::Union{SemFiniteDiff, FiniteDiffLossWrappers},
    params,
)
    wrapped = _unwrap(sem)
    obj(p) = _evaluate!(
        objective_zero(objective, gradient, hessian),
        nothing,
        nothing,
        wrapped,
        p,
    )
    isnothing(gradient) || FiniteDiff.finite_difference_gradient!(gradient, obj, params)
    isnothing(hessian) || FiniteDiff.finite_difference_hessian!(hessian, obj, params)
    # FIXME if objective is not calculated, SemLoss implied states may not correspond to params
    return !isnothing(objective) ? obj(params) : nothing
end
