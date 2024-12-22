############################################################################################
# constructor for Sem types
############################################################################################

function Sem(;
    specification = ParameterTable,
    observed::O = SemObservedData,
    imply::I = RAM,
    loss::L = SemML,
    kwargs...,
) where {O, I, L}
    kwdict = Dict{Symbol, Any}(kwargs...)

    set_field_type_kwargs!(kwdict, observed, imply, loss, O, I)

    observed, imply, loss = get_fields!(kwdict, specification, observed, imply, loss)

    sem = Sem(observed, imply, loss)

    return sem
end

"""
    imply(model::AbstractSemSingle) -> SemImply

Returns the [*implied*](@ref SemImply) part of a model.
"""
imply(model::AbstractSemSingle) = model.imply

nvars(model::AbstractSemSingle) = nvars(imply(model))
nobserved_vars(model::AbstractSemSingle) = nobserved_vars(imply(model))
nlatent_vars(model::AbstractSemSingle) = nlatent_vars(imply(model))

vars(model::AbstractSemSingle) = vars(imply(model))
observed_vars(model::AbstractSemSingle) = observed_vars(imply(model))
latent_vars(model::AbstractSemSingle) = latent_vars(imply(model))

params(model::AbstractSemSingle) = params(imply(model))
nparams(model::AbstractSemSingle) = nparams(imply(model))

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
    imply::I = RAM,
    loss::L = SemML,
    kwargs...,
) where {O, I, L}
    kwdict = Dict{Symbol, Any}(kwargs...)

    set_field_type_kwargs!(kwdict, observed, imply, loss, O, I)

    observed, imply, loss = get_fields!(kwdict, specification, observed, imply, loss)

    sem = SemFiniteDiff(observed, imply, loss)

    return sem
end

############################################################################################
# functions
############################################################################################

function set_field_type_kwargs!(kwargs, observed, imply, loss, O, I)
    kwargs[:observed_type] = O <: Type ? observed : typeof(observed)
    kwargs[:imply_type] = I <: Type ? imply : typeof(imply)
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
function get_fields!(kwargs, specification, observed, imply, loss)
    if !isa(specification, SemSpecification)
        specification = specification(; kwargs...)
    end

    # observed
    if !isa(observed, SemObserved)
        observed = observed(; specification, kwargs...)
    end
    kwargs[:observed] = observed

    # imply
    if !isa(imply, SemImply)
        imply = imply(; specification, kwargs...)
    end

    kwargs[:imply] = imply
    kwargs[:nparams] = nparams(imply)

    # loss
    loss = get_SemLoss(loss; specification, kwargs...)
    kwargs[:loss] = loss

    return observed, imply, loss
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
    print(io, "   imply:       $(nameof(I)) \n")
end

function Base.show(io::IO, sem::SemFiniteDiff{O, I, L}) where {O, I, L}
    lossfuntypes = @. string(nameof(typeof(sem.loss.functions)))
    lossfuntypes = "   " .* lossfuntypes .* ("\n")
    print(io, "Structural Equation Model : Finite Diff Approximation\n")
    print(io, "- Loss Functions \n")
    print(io, lossfuntypes...)
    print(io, "- Fields \n")
    print(io, "   observed:    $(nameof(O)) \n")
    print(io, "   imply:       $(nameof(I)) \n")
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
