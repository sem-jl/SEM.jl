"""
    (1) swap_observed(model::AbstractSemSingle; kwargs...)

    (2) swap_observed(model::AbstractSemSingle, observed; kwargs...)

Return a new model with swaped observed part.

# Arguments
- `model::AbstractSemSingle`: model to swap the observed part of.
- `kwargs`: additional keyword arguments; typically includes `data = ...`
- `observed`: Either an object of subtype of `SemObserved` or a subtype of `SemObserved`

# Examples
See the online documentation on [Swap observed data](@ref).
"""
function swap_observed end

"""
    update_observed(to_update, observed::SemObserved; kwargs...)

Update a `SemImplied`, `SemLossFunction` or `SemOptimizer` object to use a `SemObserved` object.

# Examples
See the online documentation on [Swap observed data](@ref).

# Implementation
You can provide a method for this function when defining a new type, for more information
on this see the online developer documentation on [Update observed data](@ref).
"""
function update_observed end

############################################################################################
# change observed (data) without reconstructing the whole model
############################################################################################

# use the same observed type as before
swap_observed(model::AbstractSemSingle; kwargs...) =
    swap_observed(model, typeof(observed(model)).name.wrapper; kwargs...)

# construct a new observed type
swap_observed(model::AbstractSemSingle, observed_type; kwargs...) =
    swap_observed(model, observed_type(; kwargs...); kwargs...)

swap_observed(model::AbstractSemSingle, new_observed::SemObserved; kwargs...) =
    swap_observed(
        model,
        observed(model),
        implied(model),
        loss(model),
        new_observed;
        kwargs...,
    )

function swap_observed(
    model::AbstractSemSingle,
    old_observed,
    implied,
    loss,
    new_observed::SemObserved;
    kwargs...,
)
    kwargs = Dict{Symbol, Any}(kwargs...)

    # get field types
    kwargs[:observed_type] = typeof(new_observed)
    kwargs[:old_observed_type] = typeof(old_observed)
    kwargs[:implied_type] = typeof(implied)
    kwargs[:loss_types] = [typeof(lossfun) for lossfun in loss.functions]

    # update implied
    implied = update_observed(implied, new_observed; kwargs...)
    kwargs[:implied] = implied
    kwargs[:nparams] = nparams(implied)

    # update loss
    loss = update_observed(loss, new_observed; kwargs...)
    kwargs[:loss] = loss

    #new_implied = update_observed(model.implied, new_observed; kwargs...)

    return Sem(
        new_observed,
        update_observed(model.implied, new_observed; kwargs...),
        update_observed(model.loss, new_observed; kwargs...),
    )
end

function update_observed(loss::SemLoss, new_observed; kwargs...)
    new_functions = Tuple(
        update_observed(lossfun, new_observed; kwargs...) for lossfun in loss.functions
    )
    return SemLoss(new_functions, loss.weights)
end

############################################################################################
# simulate data
############################################################################################
"""
    (1) rand(model::AbstractSemSingle, params, n)

    (2) rand(model::AbstractSemSingle, n)

Sample normally distributed data from the model-implied covariance matrix and mean vector.

# Arguments
- `model::AbstractSemSingle`: model to simulate from.
- `params`: parameter values to simulate from.
- `n::Integer`: Number of samples.

# Examples
```julia
rand(model, start_simple(model), 100)
```
"""
function Distributions.rand(
    model::AbstractSemSingle{O, I, L},
    params,
    n::Integer,
) where {O, I <: Union{RAM, RAMSymbolic}, L}
    update!(EvaluationTargets{true, false, false}(), model.implied, model, params)
    return rand(model, n)
end

function Distributions.rand(
    model::AbstractSemSingle{O, I, L},
    n::Integer,
) where {O, I <: Union{RAM, RAMSymbolic}, L}
    if MeanStruct(model.implied) === NoMeanStruct
        data = permutedims(rand(MvNormal(Symmetric(model.implied.Σ)), n))
    elseif MeanStruct(model.implied) === HasMeanStruct
        data = permutedims(rand(MvNormal(model.implied.μ, Symmetric(model.implied.Σ)), n))
    end
    return data
end
