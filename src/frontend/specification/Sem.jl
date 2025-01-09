############################################################################################
# constructor for Sem types
############################################################################################

function Sem(;
    specification = ParameterTable,
    observed::O = SemObservedData,
    implied::I = RAM,
    loss::L = SemML,
    kwargs...,
) where {O, I, L}
    kwdict = Dict{Symbol, Any}(kwargs...)

    set_field_type_kwargs!(kwdict, observed, implied, loss, O, I)

    observed, implied, loss = get_fields!(kwdict, specification, observed, implied, loss)

    sem = Sem(observed, implied, loss)

    return sem
end

"""
    implied(model::AbstractSemSingle) -> SemImplied

Returns the [*implied*](@ref SemImplied) part of a model.
"""
implied(model::AbstractSemSingle) = model.implied

nvars(model::AbstractSemSingle) = nvars(implied(model))
nobserved_vars(model::AbstractSemSingle) = nobserved_vars(implied(model))
nlatent_vars(model::AbstractSemSingle) = nlatent_vars(implied(model))

vars(model::AbstractSemSingle) = vars(implied(model))
observed_vars(model::AbstractSemSingle) = observed_vars(implied(model))
latent_vars(model::AbstractSemSingle) = latent_vars(implied(model))

params(model::AbstractSemSingle) = params(implied(model))
nparams(model::AbstractSemSingle) = nparams(implied(model))

"""
    observed(model::AbstractSemSingle) -> SemObserved

Returns the [*observed*](@ref SemObserved) part of a model.
"""
observed(model::AbstractSemSingle) = model.observed

nsamples(model::AbstractSemSingle) = nsamples(observed(model))

"""
    loss(model::AbstractSemSingle) -> SemLoss

Returns the [*loss*](@ref SemLoss) function of a model.
"""
loss(model::AbstractSemSingle) = model.loss

# sum of samples in all sub-models
nsamples(ensemble::SemEnsemble) = sum(nsamples, ensemble.sems)

function SemFiniteDiff(;
    specification = ParameterTable,
    observed::O = SemObservedData,
    implied::I = RAM,
    loss::L = SemML,
    kwargs...,
) where {O, I, L}
    kwdict = Dict{Symbol, Any}(kwargs...)

    set_field_type_kwargs!(kwdict, observed, implied, loss, O, I)

    observed, implied, loss = get_fields!(kwdict, specification, observed, implied, loss)

    sem = SemFiniteDiff(observed, implied, loss)

    return sem
end

############################################################################################
# functions
############################################################################################

function set_field_type_kwargs!(kwargs, observed, implied, loss, O, I)
    kwargs[:observed_type] = O <: Type ? observed : typeof(observed)
    kwargs[:implied_type] = I <: Type ? implied : typeof(implied)
    if loss isa SemLoss
        kwargs[:loss_types] = [
            lossfun isa SemLossFunction ? typeof(lossfun) : lossfun for
            lossfun in loss.functions
        ]
    elseif applicable(iterate, loss)
        kwargs[:loss_types] =
            [lossfun isa SemLossFunction ? typeof(lossfun) : lossfun for lossfun in loss]
    else
        kwargs[:loss_types] = [loss isa SemLossFunction ? typeof(loss) : loss]
    end
end

# construct Sem fields
function get_fields!(kwargs, specification, observed, implied, loss)
    if !isa(specification, SemSpecification)
        specification = specification(; kwargs...)
    end

    # observed
    if !isa(observed, SemObserved)
        observed = observed(; specification, kwargs...)
    end
    kwargs[:observed] = observed

    # implied
    if !isa(implied, SemImplied)
        implied = implied(; specification, kwargs...)
    end

    kwargs[:implied] = implied
    kwargs[:nparams] = nparams(implied)

    # loss
    loss = get_SemLoss(loss; specification, kwargs...)
    kwargs[:loss] = loss

    return observed, implied, loss
end

# construct loss field
function get_SemLoss(loss; kwargs...)
    if loss isa SemLoss
        nothing
    elseif applicable(iterate, loss)
        loss_out = []
        for lossfun in loss
            if isa(lossfun, SemLossFunction)
                push!(loss_out, lossfun)
            else
                lossfun = lossfun(; kwargs...)
                push!(loss_out, lossfun)
            end
        end
        loss = SemLoss(loss_out...; kwargs...)
    else
        if !isa(loss, SemLossFunction)
            loss = SemLoss(loss(; kwargs...); kwargs...)
        else
            loss = SemLoss(loss; kwargs...)
        end
    end
    return loss
end

##############################################################
# pretty printing
##############################################################

#= function Base.show(io::IO, sem::Sem{O, I, L, D})  where {O, I, L, D}
    lossfuntypes = @. nameof(typeof(sem.loss.functions))
    print(io, "Sem{$(nameof(O)), $(nameof(I)), $lossfuntypes, $(nameof(D))}")
end =#

function Base.show(io::IO, sem::Sem{O, I, L}) where {O, I, L}
    lossfuntypes = @. string(nameof(typeof(sem.loss.functions)))
    lossfuntypes = "   " .* lossfuntypes .* ("\n")
    print(io, "Structural Equation Model \n")
    print(io, "- Loss Functions \n")
    print(io, lossfuntypes...)
    print(io, "- Fields \n")
    print(io, "   observed:    $(nameof(O)) \n")
    print(io, "   implied:     $(nameof(I)) \n")
end

function Base.show(io::IO, sem::SemFiniteDiff{O, I, L}) where {O, I, L}
    lossfuntypes = @. string(nameof(typeof(sem.loss.functions)))
    lossfuntypes = "   " .* lossfuntypes .* ("\n")
    print(io, "Structural Equation Model : Finite Diff Approximation\n")
    print(io, "- Loss Functions \n")
    print(io, lossfuntypes...)
    print(io, "- Fields \n")
    print(io, "   observed:    $(nameof(O)) \n")
    print(io, "   implied:     $(nameof(I)) \n")
end

function Base.show(io::IO, loss::SemLoss)
    lossfuntypes = @. string(nameof(typeof(loss.functions)))
    lossfuntypes = "   " .* lossfuntypes .* ("\n")
    print(io, "SemLoss \n")
    print(io, "- Loss Functions \n")
    print(io, lossfuntypes...)
    print(io, "- Weights \n")
    for weight in loss.weights
        if isnothing(weight.w)
            print(io, "   one \n")
        else
            print(io, "$(round.(weight.w, digits = 2)) \n")
        end
    end
end

function Base.show(io::IO, models::SemEnsemble)
    print(io, "SemEnsemble \n")
    print(io, "- Number of Models: $(models.n) \n")
    print(io, "- Weights: $(round.(models.weights, digits = 2)) \n")

    print(io, "\n", "Models: \n")
    print(io, "===============================================", "\n")
    for (model, i) in zip(models.sems, 1:models.n)
        print(io, "---------------------- ", i, " ----------------------", "\n")
        print(io, model)
    end
end
