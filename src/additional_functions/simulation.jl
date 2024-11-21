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

Update a `SemImply`, `SemLossFunction` or `SemOptimizer` object to use a `SemObserved` object.

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
        imply(model),
        loss(model),
        optimizer(model),
        new_observed;
        kwargs...,
    )

function swap_observed(
    model::AbstractSemSingle,
    old_observed,
    imply,
    loss,
    optimizer,
    new_observed::SemObserved;
    kwargs...,
)
    kwargs = Dict{Symbol, Any}(kwargs...)

    # get field types
    kwargs[:observed_type] = typeof(new_observed)
    kwargs[:old_observed_type] = typeof(old_observed)
    kwargs[:imply_type] = typeof(imply)
    kwargs[:loss_types] = [typeof(lossfun) for lossfun in loss.functions]
    kwargs[:optimizer_type] = typeof(optimizer)

    # update imply
    imply = update_observed(imply, new_observed; kwargs...)
    kwargs[:imply] = imply
    kwargs[:nparams] = nparams(imply)

    # update loss
    loss = update_observed(loss, new_observed; kwargs...)
    kwargs[:loss] = loss

    # update optimizer
    optimizer = update_observed(optimizer, new_observed; kwargs...)

    #new_imply = update_observed(model.imply, new_observed; kwargs...)

    return Sem(
        new_observed,
        update_observed(model.imply, new_observed; kwargs...),
        update_observed(model.loss, new_observed; kwargs...),
        update_observed(model.optimizer, new_observed; kwargs...),
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
    model::AbstractSemSingle{O, I, L, D},
    params,
    n::Integer,
) where {O, I <: Union{RAM, RAMSymbolic}, L, D}
    update!(EvaluationTargets{true, false, false}(), model.imply, model, params)
    return rand(model, n)
end

function Distributions.rand(
    model::AbstractSemSingle{O, I, L, D},
    n::Integer,
) where {O, I <: Union{RAM, RAMSymbolic}, L, D}
    if MeanStruct(model.imply) === NoMeanStruct
        data = permutedims(rand(MvNormal(Symmetric(model.imply.Σ)), n))
    elseif MeanStruct(model.imply) === HasMeanStruct
        data = permutedims(rand(MvNormal(model.imply.μ, Symmetric(model.imply.Σ)), n))
    end
    return data
end
