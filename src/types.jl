############################################################################################
# Define the basic type system
############################################################################################
"Most abstract supertype for all SEMs"
abstract type AbstractSem end

"Supertype for all single SEMs, e.g. SEMs that have at least the fields `observed`, `implied`, `loss`"
abstract type AbstractSemSingle{O, I, L} <: AbstractSem end

"Supertype for all collections of multiple SEMs"
abstract type AbstractSemCollection <: AbstractSem end

"Meanstructure trait for `SemImplied` subtypes"
abstract type MeanStruct end
"Indicates that `SemImplied` subtype supports mean structure"
struct HasMeanStruct <: MeanStruct end
"Indicates that `SemImplied` subtype does not support mean structure"
struct NoMeanStruct <: MeanStruct end

# default implementation
MeanStruct(::Type{T}) where {T} =
    hasfield(T, :meanstruct) ? fieldtype(T, :meanstruct) :
    error("Objects of type $T do not support MeanStruct trait")

MeanStruct(semobj) = MeanStruct(typeof(semobj))

"Hessian Evaluation trait for `SemImplied` and `SemLossFunction` subtypes"
abstract type HessianEval end
struct ApproxHessian <: HessianEval end
struct ExactHessian <: HessianEval end

# default implementation
HessianEval(::Type{T}) where {T} =
    hasfield(T, :hessianeval) ? fieldtype(T, :hessianeval) :
    error("Objects of type $T do not support HessianEval trait")

HessianEval(semobj) = HessianEval(typeof(semobj))

"Supertype for all loss functions of SEMs. If you want to implement a custom loss function, it should be a subtype of `SemLossFunction`."
abstract type SemLossFunction end

"""
    SemLoss(args...; loss_weights = nothing, ...)

Constructs the loss field of a SEM. Can contain multiple `SemLossFunction`s, the model is optimized over their sum.
See also [`SemLossFunction`](@ref).

# Arguments
- `args...`: Multiple `SemLossFunction`s.
- `loss_weights::Vector`: Weights for each loss function. Defaults to unweighted optimization.

# Examples
```julia
my_ml_loss = SemML(...)
my_ridge_loss = SemRidge(...)
my_loss = SemLoss(SemML, SemRidge; loss_weights = [1.0, 2.0])
```
"""
mutable struct SemLoss{F <: Tuple, T}
    functions::F
    weights::T
end

function SemLoss(functions...; loss_weights = nothing, kwargs...)
    if !isnothing(loss_weights)
        loss_weights = SemWeight.(loss_weights)
    else
        loss_weights = Tuple(SemWeight(nothing) for _ in 1:length(functions))
    end

    return SemLoss(functions, loss_weights)
end

# weights for loss functions or models. If the weight is nothing, multiplication returns the second argument
struct SemWeight{T}
    w::T
end

Base.:*(x::SemWeight{Nothing}, y) = y
Base.:*(x::SemWeight, y) = x.w * y

"""
Supertype of all objects that can serve as the `optimizer` field of a SEM.
Connects the SEM to its optimization backend and controls options like the optimization algorithm.
If you want to connect the SEM package to a new optimization backend, you should implement a subtype of SemOptimizer.
"""
abstract type SemOptimizer{E} end

engine(::Type{SemOptimizer{E}}) where {E} = E
engine(optimizer::SemOptimizer) = engine(typeof(optimizer))

SemOptimizer(args...; engine::Symbol = :Optim, kwargs...) =
    SemOptimizer{engine}(args...; kwargs...)

# fallback optimizer constructor
function SemOptimizer{E}(args...; kwargs...) where {E}
    throw(ErrorException("$E optimizer is not supported."))
end

"""
Supertype of all objects that can serve as the observed field of a SEM.
Pre-processes data and computes sufficient statistics for example.
If you have a special kind of data, e.g. ordinal data, you should implement a subtype of SemObserved.
"""
abstract type SemObserved end

"""
Supertype of all objects that can serve as the implied field of a SEM.
Computed model-implied values that should be compared with the observed data to find parameter estimates,
e. g. the model implied covariance or mean.
If you would like to implement a different notation, e.g. LISREL, you should implement a subtype of SemImplied.
"""
abstract type SemImplied end

"Subtype of SemImplied for all objects that can serve as the implied field of a SEM and use some form of symbolic precomputation."
abstract type SemImpliedSymbolic <: SemImplied end

"""
    Sem(;observed = SemObservedData, implied = RAM, loss = SemML, kwargs...)

Constructor for the basic `Sem` type.
All additional kwargs are passed down to the constructors for the observed, implied, and loss fields.

# Arguments
- `observed`: object of subtype `SemObserved` or a constructor.
- `implied`: object of subtype `SemImplied` or a constructor.
- `loss`: object of subtype `SemLossFunction`s or constructor; or a tuple of such.

Returns a Sem with fields
- `observed::SemObserved`: Stores observed data, sample statistics, etc. See also [`SemObserved`](@ref).
- `implied::SemImplied`: Computes model implied statistics, like Σ, μ, etc. See also [`SemImplied`](@ref).
- `loss::SemLoss`: Computes the objective and gradient of a sum of loss functions. See also [`SemLoss`](@ref).
"""
mutable struct Sem{O <: SemObserved, I <: SemImplied, L <: SemLoss} <:
               AbstractSemSingle{O, I, L}
    observed::O
    implied::I
    loss::L
end

############################################################################################
# automatic differentiation
############################################################################################
"""
    SemFiniteDiff(;observed = SemObservedData, implied = RAM, loss = SemML, kwargs...)

A wrapper around [`Sem`](@ref) that substitutes dedicated evaluation of gradient and hessian with
finite difference approximation.

# Arguments
- `observed`: object of subtype `SemObserved` or a constructor.
- `implied`: object of subtype `SemImplied` or a constructor.
- `loss`: object of subtype `SemLossFunction`s or constructor; or a tuple of such.

Returns a Sem with fields
- `observed::SemObserved`: Stores observed data, sample statistics, etc. See also [`SemObserved`](@ref).
- `implied::SemImplied`: Computes model implied statistics, like Σ, μ, etc. See also [`SemImplied`](@ref).
- `loss::SemLoss`: Computes the objective and gradient of a sum of loss functions. See also [`SemLoss`](@ref).
"""
struct SemFiniteDiff{O <: SemObserved, I <: SemImplied, L <: SemLoss} <:
       AbstractSemSingle{O, I, L}
    observed::O
    implied::I
    loss::L
end

############################################################################################
# ensemble models
############################################################################################
"""
    (1) SemEnsemble(models..., weights = nothing, kwargs...)

    (2) SemEnsemble(;specification, data, groups, column = :group, kwargs...)

Constructor for ensemble models. (2) can be used to conveniently specify multigroup models.

# Arguments
- `models...`: `AbstractSem`s.
- `weights::Vector`:  Weights for each model. Defaults to the number of observed data points.
- `specification::EnsembleParameterTable`: Model specification.
- `data::DataFrame`: Observed data. Must contain a `column` of type `Vector{Symbol}` that contains the group.
- `groups::Vector{Symbol}`: Group names.
- `column::Symbol`: Name of the column in `data` that contains the group.

All additional kwargs are passed down to the model parts.

Returns a SemEnsemble with fields
- `n::Int`: Number of models.
- `sems::Tuple`: `AbstractSem`s.
- `weights::Vector`: Weights for each model.
- `params::Vector`: Stores parameter labels and their position.
"""
struct SemEnsemble{N, T <: Tuple, V <: AbstractVector, I} <: AbstractSemCollection
    n::N
    sems::T
    weights::V
    params::I
end

# constructor from multiple models
function SemEnsemble(models...; weights = nothing, kwargs...)
    n = length(models)

    # default weights

    if isnothing(weights)
        nsamples_total = sum(nsamples, models)
        weights = [nsamples(model) / nsamples_total for model in models]
    end

    # check parameters equality
    params = SEM.params(models[1])
    for model in models
        if params != SEM.params(model)
            throw(ErrorException("The parameters of your models do not match. \n
            Maybe you tried to specify models of an ensemble via ParameterTables. \n
            In that case, you may use RAMMatrices instead."))
        end
    end

    return SemEnsemble(n, models, weights, params)
end

# constructor from EnsembleParameterTable and data set
function SemEnsemble(; specification, data, groups, column = :group, kwargs...)
    if specification isa EnsembleParameterTable
        specification = convert(Dict{Symbol, RAMMatrices}, specification)
    end
    models = []
    for group in groups
        ram_matrices = specification[group]
        data_group = select(filter(r -> r[column] == group, data), Not(column))
        if iszero(nrow(data_group))
            error("Your data does not contain any observations from group `$(group)`.")
        end
        model = Sem(; specification = ram_matrices, data = data_group, kwargs...)
        push!(models, model)
    end
    return SemEnsemble(models...; weights = nothing, kwargs...)
end

params(ensemble::SemEnsemble) = ensemble.params

"""
    n_models(ensemble::SemEnsemble) -> Integer

Returns the number of models in an ensemble model.
"""
n_models(ensemble::SemEnsemble) = ensemble.n
"""
    models(ensemble::SemEnsemble) -> Tuple{AbstractSem}

Returns the models in an ensemble model.
"""
models(ensemble::SemEnsemble) = ensemble.sems
"""
    weights(ensemble::SemEnsemble) -> Vector

Returns the weights of an ensemble model.
"""
weights(ensemble::SemEnsemble) = ensemble.weights

"""
Base type for all SEM specifications.
"""
abstract type SemSpecification end

abstract type AbstractParameterTable <: SemSpecification end
