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

# don't change non-SEM terms
swap_observed(loss::AbstractLoss; kwargs...) = loss

# use the same observed type as before
swap_observed(loss::SemLoss; kwargs...) =
    swap_observed(loss, typeof(SEM.observed(loss)).name.wrapper; kwargs...)

# construct a new observed type
swap_observed(loss::SemLoss, observed_type; kwargs...) =
    swap_observed(loss, observed_type(; kwargs...); kwargs...)

function swap_observed(loss::SemLoss, new_observed::SemObserved; kwargs...)
    kwargs = Dict{Symbol, Any}(kwargs...)
    old_observed = SEM.observed(loss)
    implied = SEM.implied(loss)

    # get field types
    kwargs[:observed_type] = typeof(new_observed)
    kwargs[:old_observed_type] = typeof(old_observed)

    # update implied
    implied = update_observed(implied, new_observed; kwargs...)
    kwargs[:implied] = implied
    kwargs[:implied_type] = typeof(implied)
    kwargs[:nparams] = nparams(implied)

    # update loss
    return update_observed(loss, new_observed; kwargs...)
end

swap_observed(loss::LossTerm; kwargs...) =
    LossTerm(swap_observed(loss.loss; kwargs...), loss.id, loss.weight)

function swap_observed(sem::Sem; kwargs...)
    updated_terms = Tuple(swap_observed(term; kwargs...) for term in loss_terms(sem))
    return Sem(updated_terms...)
end

############################################################################################
# simulate data
############################################################################################
"""
    rand(sem::Union{Sem, SemLoss, SemImplied}, params, n)
    rand(sem::Union{Sem, SemLoss, SemImplied}, n)

Sample from the multivariate normal distribution implied by the SEM model.

# Arguments
- `sem`: SEM model to use. Ensemble models with multiple SEM terms are not supported.
- `params`: SEM model parameters to simulate from.
- `n::Integer`: Number of samples to draw.

# Examples
```julia
rand(model, start_simple(model), 100)
```
"""
function Distributions.rand(implied::SemImplied, params, n::Integer)
    update!(EvaluationTargets{true, false, false}(), implied, params)
    return rand(implied, n)
end

Distributions.rand(implied::SemImplied, n::Integer) =
    error("rand($(typeof(implied)), n) is not implemented")

function Distributions.rand(implied::Union{RAM, RAMSymbolic}, n::Integer)
    Σ = Symmetric(implied.Σ)
    if MeanStruct(implied) === NoMeanStruct
        return permutedims(rand(MvNormal(Σ), n))
    elseif MeanStruct(implied) === HasMeanStruct
        return permutedims(rand(MvNormal(implied.μ, Σ), n))
    end
end

Distributions.rand(loss::SemLoss, params, n::Integer) = rand(SEM.implied(loss), params, n)

Distributions.rand(loss::SemLoss, n::Integer) = rand(SEM.implied(loss), n)

Distributions.rand(model::Sem, params, n::Integer) = rand(sem_term(model), params, n)

Distributions.rand(model::Sem, n::Integer) = rand(sem_term(model), n)
