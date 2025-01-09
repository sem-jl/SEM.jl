losstype(::Type{<:LossTerm{L, W, I}}) where {L, W, I} = L
losstype(term::LossTerm) = losstype(typeof(term))
loss(term::LossTerm) = term.loss
weight(term::LossTerm) = term.weight
id(term::LossTerm) = term.id

"""
    issemloss(term::LossTerm) -> Bool

Check if a SEM model term is a SEM loss function ([@ref SemLoss]).
"""
issemloss(term::LossTerm) = isa(loss(term), SemLoss)

for f in (
    :implied,
    :observed,
    :nsamples,
    :observed_vars,
    :nobserved_vars,
    :vars,
    :nvars,
    :latent_vars,
    :nlatent_vars,
    :params,
    :nparams,
)
    @eval $f(term::LossTerm) = $f(loss(term))
end

function Base.show(io::IO, term::LossTerm)
    if !isnothing(id(term))
        print(io, ":$(id(term)): ")
    end
    print(io, nameof(losstype(term)))
    if issemloss(term)
        print(
            io,
            " ($(nsamples(term)) samples, $(nobserved_vars(term)) observed, $(nlatent_vars(term)) latent variables)",
        )
    end
    if !isnothing(weight(term))
        @printf(io, " w=%.3g", weight(term))
    else
        print(io, " w=1")
    end
end

############################################################################################
# constructor for Sem types
############################################################################################

function Sem(
    loss_terms...;
    params::Union{Vector{Symbol}, Nothing} = nothing,
    default_sem_weights = :nsamples,
)
    default_sem_weights ∈ [:nsamples, :uniform, :one] ||
        throw(ArgumentError("Unsupported default_sem_weights=:$default_sem_weights"))
    # assemble a list of weighted losses and check params equality
    terms = Vector{LossTerm}()
    params = !isnothing(params) ? copy(params) : params
    has_sem_weights = false
    nsems = 0
    for inp_term in loss_terms
        if inp_term isa AbstractLoss
            term = inp_term
            term_w = nothing
            term_id = nothing
        elseif inp_term isa Pair
            if inp_term[1] isa AbstractLoss
                term, term_w = inp_term
                term_id = nothing
            elseif inp_term[2] isa AbstractLoss
                term_id, term = inp_term
                term_w = nothing
            elseif inp_term[2] isa Pair
                term_id, (term, term_w) = inp_term
                isa(term, AbstractLoss) || throw(
                    ArgumentError(
                        "AbstractLoss expected as a second argument of a loss term double pair (id => loss => weight), $(nameof(typeof(term))) found",
                    ),
                )
            end
        elseif inp_term isa LossTerm
            term_id = id(inp_term)
            term = loss(inp_term)
            term_w = weight(inp_term)
        else
            throw(
                ArgumentError(
                    "[id =>] AbstractLoss [=> weight] expected as a loss term, $(nameof(typeof(inp_term))) found",
                ),
            )
        end

        if term isa SemLoss
            nsems += 1
            has_sem_weights |= !isnothing(term_w)
            # check integrity
            if isnothing(params)
                params = SEM.params(term)
            elseif params != SEM.params(term)
                # FIXME the suggestion might no longer be relevant, since ParTable also stores params order
                error("The parameters of your SEM models do not match.\n
Maybe you tried to specify models of an ensemble via ParameterTables.\n
In that case, you may use RAMMatrices instead.")
            end
            check_observed_vars(term)
        elseif !(term isa AbstractLoss)
            throw(
                ArgumentError(
                    "AbstractLoss term expected at $(length(terms)+1) position, $(nameof(typeof(term))) found",
                ),
            )
        end
        push!(terms, LossTerm(term, term_id, term_w))
    end
    isnothing(params) && throw(ErrorException("No SEM models provided."))

    if !has_sem_weights && nsems > 1
        # set the weights of SEMs in the ensemble
        if default_sem_weights == :nsamples
            # weight SEM by the number of samples
            nsamples_total = sum(nsamples(term) for term in terms if issemloss(term))
            for (i, term) in enumerate(terms)
                if issemloss(term)
                    terms[i] =
                        LossTerm(loss(term), id(term), nsamples(term) / nsamples_total)
                end
            end
        elseif default_sem_weights == :uniform # uniform weights
            for (i, term) in enumerate(terms)
                if issemloss(term)
                    terms[i] = LossTerm(loss(term), id(term), 1 / nsems)
                end
            end
        elseif default_sem_weights == :one # do nothing
        end
    end

    terms_tuple = Tuple(terms)
    return Sem{typeof(terms_tuple)}(terms_tuple, params)
end

function Sem(;
    specification = ParameterTable,
    observed::O = SemObservedData,
    implied::I = RAM,
    loss::L = SemML,
    kwargs...,
) where {O, I, L}
    kwdict = Dict{Symbol, Any}(kwargs...)

    set_field_type_kwargs!(kwdict, observed, implied, loss, O, I)

    loss = get_fields!(kwdict, specification, observed, implied, loss)

    return Sem(loss...)
end

############################################################################################
# functions
############################################################################################

params(model::AbstractSem) = model.params

"""
    loss_terms(model::AbstractSem)

Returns a tuple of all [`LossTerm`](@ref) weighted terms in the SEM model.

See also [`sem_terms`](@ref), [`loss_term`](@ref).
"""
loss_terms(model::AbstractSem) = model.loss_terms
nloss_terms(model::AbstractSem) = length(loss_terms(model))

"""
    sem_terms(model::AbstractSem)

Returns a tuple of all weighted SEM terms in the SEM model.

In comparison to [`loss_terms`](@ref) that returns all model terms, including e.g.
regularization terms, this function returns only the [`SemLoss`] terms.

See also [`loss_terms`](@ref), [`sem_term`](@ref).
"""
sem_terms(model::AbstractSem) = Tuple(term for term in loss_terms(model) if issemloss(term))
nsem_terms(model::AbstractSem) = sum(issemloss, loss_terms(model))

nsamples(model::AbstractSem) =
    sum(term -> issemloss(term) ? nsamples(term) : 0, loss_terms(model))

"""
    loss_term(model::AbstractSem, id::Any) -> AbstractLoss

Returns the loss term with the specified `id` from the `model`.
Throws an error if the model has no term with the specified `id`.

See also [`loss_terms`](@ref).
"""
function loss_term(model::AbstractSem, id::Any)
    for term in loss_terms(model)
        if SEM.id(term) == id
            return loss(term)
        end
    end
    error("No loss term with id=$id found")
end

"""
    sem_term(model::AbstractSem, [id]) -> SemLoss

Returns the SEM loss term with the specified `id` from the `model`.
Throws an error if the model has no term with the specified `id` or
if it is not of a [`SemLoss`](@ref) type.

If no `id` is specified and the model contains only one SEM term, the term is returned.
Throws an error if the model contains multiple SEM terms.

See also [`loss_term`](@ref), [`sem_terms`](@ref).
"""
function sem_term(model::AbstractSem, id::Any)
    term = loss_term(model, id)
    issemloss(term) || error("Loss term with id=$id ($(typeof(term))) is not a SEM term")
    return term
end

function sem_term(model::AbstractSem, id::Nothing = nothing)
    if nsem_terms(model) != 1
        error(
            "Model contains $(nsem_terms(model)) SEM terms, you have to specify a specific term",
        )
    end
    for term in loss_terms(model)
        issemloss(term) && return loss(term)
    end
    error("Unreachable reached")
end

# wrappers arounds a single SemLoss term
observed(model::AbstractSem, id::Nothing = nothing) = observed(sem_term(model, id))
implied(model::AbstractSem, id::Nothing = nothing) = implied(sem_term(model, id))
vars(model::AbstractSem, id::Nothing = nothing) = vars(implied(model, id))
observed_vars(model::AbstractSem, id::Nothing = nothing) = observed_vars(implied(model, id))
latent_vars(model::AbstractSem, id::Nothing = nothing) = latent_vars(implied(model, id))

function set_field_type_kwargs!(kwargs, observed, implied, loss, O, I)
    kwargs[:observed_type] = O <: Type ? observed : typeof(observed)
    kwargs[:implied_type] = I <: Type ? implied : typeof(implied)
    if loss isa SemLoss
        kwargs[:loss_types] =
            [aloss isa SemLoss ? typeof(aloss) : aloss for aloss in loss.functions]
    elseif applicable(iterate, loss)
        kwargs[:loss_types] = [aloss isa SemLoss ? typeof(aloss) : aloss for aloss in loss]
    else
        kwargs[:loss_types] = [loss isa SemLoss ? typeof(loss) : loss]
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

    # implied
    if !isa(implied, SemImplied)
        implied = implied(specification; kwargs...)
    end

    if observed_vars(observed) != observed_vars(implied)
        throw(ArgumentError("observed_vars differ between the observed and the implied"))
    end

    kwargs[:nparams] = nparams(implied)

    # loss
    loss = get_SemLoss(loss, observed, implied; kwargs...)
    kwargs[:loss] = loss

    return loss
end

# construct loss field
function get_SemLoss(loss, observed, implied; kwargs...)
    if loss isa SemLoss
        return loss
    elseif applicable(iterate, loss)
        loss_out = AbstractLoss[]
        for aloss in loss
            if isa(aloss, AbstractLoss)
                push!(loss_out, aloss)
            elseif aloss <: SemLoss{O, I} where {O, I}
                res = aloss(observed, implied; kwargs...)
                push!(loss_out, res)
            else
                res = aloss(; kwargs...)
                push!(loss_out, res)
            end
        end
        return Tuple(loss_out)
    else
        return (loss(observed, implied; kwargs...),)
    end
end

function update_observed(sem::Sem, new_observed; kwargs...)
    new_terms = Tuple(
        update_observed(lossterm.loss, new_observed; kwargs...) for
        lossterm in loss_terms(sem)
    )
    return Sem(new_terms...)
end

##############################################################
# pretty printing
##############################################################

function Base.show(io::IO, sem::AbstractSem)
    println(io, "Structural Equation Model ($(nameof(typeof(sem))))")
    println(io, "- $(nparams(sem)) parameters")
    println(io, "- Loss terms:")
    for term in loss_terms(sem)
        print(io, "  - ")
        print(io, term)
        println(io)
    end
end
