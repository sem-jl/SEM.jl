## Interface to the StenoGraphs package

############################################################################################
### Define Modifiers
############################################################################################

#FIXME: remove when StenoGraphs.jl will provide AbstractStenoGraph
const AbstractStenoGraph = AbstractArray{T, 1} where {T <: StenoGraphs.AbstractEdge}

# fixed parameter values
struct Fixed{N} <: EdgeModifier
    value::N
end
fixed(args...) = Fixed(args)
Fixed(value::Int) = Fixed(Float64(value))

# start values
struct Start{N} <: EdgeModifier
    value::N
end
start(args...) = Start(args)
Start(value::Int) = Start(Float64(value))

# labels for equality constraints
struct Label{N} <: EdgeModifier
    value::N
end
label(args...) = Label(args)

# test whether the modifier is NaN
isnanmodval(val::Number) = isnan(val)
isnanmodval(val::Symbol) = val == :NaN
isnanmodval(val::SimpleNode{Symbol}) = val.node == :NaN

############################################################################################
### constructor for parameter table from graph
############################################################################################

function ParameterTable(
    graph::AbstractStenoGraph;
    observed_vars::AbstractVector{Symbol},
    latent_vars::AbstractVector{Symbol},
    params::Union{AbstractVector{Symbol}, Nothing} = nothing,
    group::Union{Integer, Nothing} = nothing,
    param_prefix = :θ,
)
    graph = unique(graph)
    n = length(graph)

    columns = empty_partable_columns(n)
    from = columns[:from]
    relation = columns[:relation]
    to = columns[:to]
    free = columns[:free]
    value_fixed = columns[:value_fixed]
    start = columns[:start]
    param_refs = columns[:param]
    # group = Vector{Symbol}(undef, n)

    for (i, element) in enumerate(graph)
        edge = element isa ModifiedEdge ? element.edge : element
        from[i] = edge.src.node
        to[i] = edge.dst.node
        if edge isa DirectedEdge
            relation[i] = :→
        elseif edge isa UndirectedEdge
            relation[i] = :↔
        else
            throw(
                ArgumentError(
                    "The graph contains an unsupported edge of type $(typeof(edge)).",
                ),
            )
        end
        if element isa ModifiedEdge
            if any(Base.Fix2(isa, Fixed), values(element.modifiers)) &&
               any(Base.Fix2(isa, Label), values(element.modifiers))
                throw(ArgumentError("It is not allowed to label fixed parameters."))
            end
            for modifier in values(element.modifiers)
                if isnothing(group) &&
                   modifier.value isa Union{AbstractVector, Tuple} &&
                   length(modifier.value) > 1
                    throw(
                        ArgumentError(
                            "The graph contains a group of parameters, ParameterTable expects a single value.\n" *
                            "For SEM ensembles, use EnsembleParameterTable instead.",
                        ),
                    )
                end
                modval = modifier.value[something(group, 1)]
                if modifier isa Fixed
                    if isnanmodval(modval)
                        free[i] = true
                        value_fixed[i] = 0.0
                    else
                        free[i] = false
                        value_fixed[i] = modval
                    end
                elseif modifier isa Start
                    if !isnanmodval(modval)
                        start[i] = modval
                    end
                elseif modifier isa Label
                    if isnanmodval(modval)
                        throw(DomainError(NaN, "NaN is not allowed as a parameter label."))
                    end
                    param_refs[i] = modval
                end
            end
        end
    end

    # make identifiers for parameters that are not labeled
    current_id = 1
    for i in eachindex(param_refs)
        if param_refs[i] == Symbol("")
            if free[i]
                param_refs[i] = Symbol(param_prefix, :_, current_id)
                current_id += 1
            else
                param_refs[i] = :const
            end
        elseif !free[i]
            @warn "You labeled a constant ($(param_refs[i])=$(value_fixed[i])). Please check if the labels of your graph are correct."
        end
    end

    return ParameterTable(columns; latent_vars, observed_vars, params)
end

############################################################################################
### constructor for EnsembleParameterTable from graph
############################################################################################

function EnsembleParameterTable(
    graph::AbstractStenoGraph;
    observed_vars::AbstractVector{Symbol},
    latent_vars::AbstractVector{Symbol},
    params::Union{AbstractVector{Symbol}, Nothing} = nothing,
    groups,
)
    graph = unique(graph)

    partables = Dict(
        group => ParameterTable(
            graph;
            observed_vars,
            latent_vars,
            params,
            group = i,
            param_prefix = Symbol(:g, group),
        ) for (i, group) in enumerate(groups)
    )

    return EnsembleParameterTable(partables; params)
end
