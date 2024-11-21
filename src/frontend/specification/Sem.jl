############################################################################################
# constructor for Sem types
############################################################################################

function Sem(;
    observed::O = SemObservedData,
    imply::I = RAM,
    loss::L = SemML,
    optimizer::D = SemOptimizerOptim,
    kwargs...,
) where {O, I, L, D}
    kwdict = Dict{Symbol, Any}(kwargs...)

    set_field_type_kwargs!(kwdict, observed, imply, loss, optimizer, O, I, D)

    observed, imply, loss, optimizer = get_fields!(kwdict, observed, imply, loss, optimizer)

    sem = Sem(observed, imply, loss, optimizer)

    return sem
end

nvars(sem::AbstractSemSingle) = nvars(sem.imply)
nobserved_vars(sem::AbstractSemSingle) = nobserved_vars(sem.imply)
nlatent_vars(sem::AbstractSemSingle) = nlatent_vars(sem.imply)

vars(sem::AbstractSemSingle) = vars(sem.imply)
observed_vars(sem::AbstractSemSingle) = observed_vars(sem.imply)
latent_vars(sem::AbstractSemSingle) = latent_vars(sem.imply)

nsamples(sem::AbstractSemSingle) = nsamples(sem.observed)

params(model::AbstractSem) = params(model.imply)

# sum of samples in all sub-models
nsamples(ensemble::SemEnsemble) = sum(nsamples, ensemble.sems)

############################################################################################
# additional methods
############################################################################################
"""
    observed(model::AbstractSemSingle) -> SemObserved

Returns the observed part of a model.
"""
observed(model::AbstractSemSingle) = model.observed

"""
    imply(model::AbstractSemSingle) -> SemImply

Returns the imply part of a model.
"""
imply(model::AbstractSemSingle) = model.imply

"""
    loss(model::AbstractSemSingle) -> SemLoss

Returns the loss part of a model.
"""
loss(model::AbstractSemSingle) = model.loss

"""
    optimizer(model::AbstractSemSingle) -> SemOptimizer

Returns the optimizer part of a model.
"""
optimizer(model::AbstractSemSingle) = model.optimizer

function SemFiniteDiff(;
    observed::O = SemObservedData,
    imply::I = RAM,
    loss::L = SemML,
    optimizer::D = SemOptimizerOptim,
    kwargs...,
) where {O, I, L, D}
    kwdict = Dict{Symbol, Any}(kwargs...)

    set_field_type_kwargs!(kwdict, observed, imply, loss, optimizer, O, I, D)

    observed, imply, loss, optimizer = get_fields!(kwdict, observed, imply, loss, optimizer)

    sem = SemFiniteDiff(observed, imply, loss, optimizer)

    return sem
end

############################################################################################
# functions
############################################################################################

function set_field_type_kwargs!(kwargs, observed, imply, loss, optimizer, O, I, D)
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
    kwargs[:optimizer_type] = D <: Type ? optimizer : typeof(optimizer)
end

# construct Sem fields
function get_fields!(kwargs, observed, imply, loss, optimizer)
    # observed
    if !isa(observed, SemObserved)
        observed = observed(; kwargs...)
    end
    kwargs[:observed] = observed

    # imply
    if !isa(imply, SemImply)
        imply = imply(; kwargs...)
    end

    kwargs[:imply] = imply
    kwargs[:nparams] = nparams(imply)

    # loss
    loss = get_SemLoss(loss; kwargs...)
    kwargs[:loss] = loss

    # optimizer
    if !isa(optimizer, SemOptimizer)
        optimizer = optimizer(; kwargs...)
    end

    return observed, imply, loss, optimizer
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

function Base.show(io::IO, sem::Sem{O, I, L, D}) where {O, I, L, D}
    lossfuntypes = @. string(nameof(typeof(sem.loss.functions)))
    lossfuntypes = "   " .* lossfuntypes .* ("\n")
    print(io, "Structural Equation Model \n")
    print(io, "- Loss Functions \n")
    print(io, lossfuntypes...)
    print(io, "- Fields \n")
    print(io, "   observed:    $(nameof(O)) \n")
    print(io, "   imply:       $(nameof(I)) \n")
    print(io, "   optimizer:   $(nameof(D)) \n")
end

function Base.show(io::IO, sem::SemFiniteDiff{O, I, L, D}) where {O, I, L, D}
    lossfuntypes = @. string(nameof(typeof(sem.loss.functions)))
    lossfuntypes = "   " .* lossfuntypes .* ("\n")
    print(io, "Structural Equation Model : Finite Diff Approximation\n")
    print(io, "- Loss Functions \n")
    print(io, lossfuntypes...)
    print(io, "- Fields \n")
    print(io, "   observed:    $(nameof(O)) \n")
    print(io, "   imply:       $(nameof(I)) \n")
    print(io, "   optimizer:   $(nameof(D)) \n")
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
    print(io, "- optimizer: $(nameof(typeof(optimizer(models)))) \n")

    print(io, "\n", "Models: \n")
    print(io, "===============================================", "\n")
    for (model, i) in zip(models.sems, 1:models.n)
        print(io, "---------------------- ", i, " ----------------------", "\n")
        print(io, model)
    end
end
