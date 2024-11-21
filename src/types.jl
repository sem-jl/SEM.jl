############################################################################################
# Define the basic type system
############################################################################################
"Most abstract supertype for all SEMs"
abstract type AbstractSem end

"Supertype for all single SEMs, e.g. SEMs that have at least the fields `observed`, `imply`, `loss` and `optimizer`"
abstract type AbstractSemSingle{O, I, L, D} <: AbstractSem end

"Supertype for all collections of multiple SEMs"
abstract type AbstractSemCollection <: AbstractSem end

"Meanstructure trait for `SemImply` subtypes"
abstract type MeanStruct end
"Indicates that `SemImply` subtype supports mean structure"
struct HasMeanStruct <: MeanStruct end
"Indicates that `SemImply` subtype does not support mean structure"
struct NoMeanStruct <: MeanStruct end

# default implementation
MeanStruct(::Type{T}) where {T} =
    hasfield(T, :meanstruct) ? fieldtype(T, :meanstruct) :
    error("Objects of type $T do not support MeanStruct trait")

MeanStruct(semobj) = MeanStruct(typeof(semobj))

"Hessian Evaluation trait for `SemImply` and `SemLossFunction` subtypes"
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
abstract type SemOptimizer end

"""
Supertype of all objects that can serve as the observed field of a SEM.
Pre-processes data and computes sufficient statistics for example.
If you have a special kind of data, e.g. ordinal data, you should implement a subtype of SemObserved.
"""
abstract type SemObserved end

"""
Supertype of all objects that can serve as the imply field of a SEM.
Computed model-implied values that should be compared with the observed data to find parameter estimates,
e. g. the model implied covariance or mean.
If you would like to implement a different notation, e.g. LISREL, you should implement a subtype of SemImply.
"""
abstract type SemImply end

"Subtype of SemImply for all objects that can serve as the imply field of a SEM and use some form of symbolic precomputation."
abstract type SemImplySymbolic <: SemImply end

"""
    Sem(;observed = SemObservedData, imply = RAM, loss = SemML, optimizer = SemOptimizerOptim, kwargs...)

Constructor for the basic `Sem` type.
All additional kwargs are passed down to the constructors for the observed, imply, loss and optimizer fields.

# Arguments
- `observed`: object of subtype `SemObserved` or a constructor.
- `imply`: object of subtype `SemImply` or a constructor.
- `loss`: object of subtype `SemLossFunction`s or constructor; or a tuple of such.
- `optimizer`: object of subtype `SemOptimizer` or a constructor.

Returns a Sem with fields
- `observed::SemObserved`: Stores observed data, sample statistics, etc. See also [`SemObserved`](@ref).
- `imply::SemImply`: Computes model implied statistics, like Σ, μ, etc. See also [`SemImply`](@ref).
- `loss::SemLoss`: Computes the objective and gradient of a sum of loss functions. See also [`SemLoss`](@ref).
- `optimizer::SemOptimizer`: Connects the model to the optimizer. See also [`SemOptimizer`](@ref).
"""
mutable struct Sem{O <: SemObserved, I <: SemImply, L <: SemLoss, D <: SemOptimizer} <:
               AbstractSemSingle{O, I, L, D}
    observed::O
    imply::I
    loss::L
    optimizer::D
end

############################################################################################
# automatic differentiation
############################################################################################
"""
    SemFiniteDiff(;observed = SemObservedData, imply = RAM, loss = SemML, optimizer = SemOptimizerOptim, kwargs...)

Constructor for `SemFiniteDiff`.
All additional kwargs are passed down to the constructors for the observed, imply, loss and optimizer fields.

# Arguments
- `observed`: object of subtype `SemObserved` or a constructor.
- `imply`: object of subtype `SemImply` or a constructor.
- `loss`: object of subtype `SemLossFunction`s or constructor; or a tuple of such.
- `optimizer`: object of subtype `SemOptimizer` or a constructor.

Returns a Sem with fields
- `observed::SemObserved`: Stores observed data, sample statistics, etc. See also [`SemObserved`](@ref).
- `imply::SemImply`: Computes model implied statistics, like Σ, μ, etc. See also [`SemImply`](@ref).
- `loss::SemLoss`: Computes the objective and gradient of a sum of loss functions. See also [`SemLoss`](@ref).
- `optimizer::SemOptimizer`: Connects the model to the optimizer. See also [`SemOptimizer`](@ref).
"""
struct SemFiniteDiff{O <: SemObserved, I <: SemImply, L <: SemLoss, D <: SemOptimizer} <:
       AbstractSemSingle{O, I, L, D}
    observed::O
    imply::I
    loss::L
    optimizer::D
end

############################################################################################
# ensemble models
############################################################################################
"""
    (1) SemEnsemble(models..., optimizer = SemOptimizerOptim, weights = nothing, kwargs...)

    (2) SemEnsemble(;specification, data, groups, column = :group, optimizer = SemOptimizerOptim, kwargs...)

Constructor for ensemble models. (2) can be used to conveniently specify multigroup models.

# Arguments
- `models...`: `AbstractSem`s.
- `optimizer`: object of subtype `SemOptimizer` or a constructor.
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
- `optimizer::SemOptimizer`: Connects the model to the optimizer. See also [`SemOptimizer`](@ref).
- `params::Vector`: Stores parameter labels and their position.
"""
struct SemEnsemble{N, T <: Tuple, V <: AbstractVector, D, I} <: AbstractSemCollection
    n::N
    sems::T
    weights::V
    optimizer::D
    params::I
end

# constructor from multiple models
function SemEnsemble(models...; optimizer = SemOptimizerOptim, weights = nothing, kwargs...)
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

    # optimizer
    if !isa(optimizer, SemOptimizer)
        optimizer = optimizer(; kwargs...)
    end

    return SemEnsemble(n, models, weights, optimizer, params)
end

# constructor from EnsembleParameterTable and data set
function SemEnsemble(;specification, data, groups, column = :group, optimizer = SemOptimizerOptim, kwargs...)
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
        model = Sem(;
            specification = ram_matrices,
            data = data_group,
            kwargs...
        )
        push!(models, model)
    end
    return SemEnsemble(models...; optimizer = optimizer, weights = nothing, kwargs...)
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
    optimizer(ensemble::SemEnsemble) -> SemOptimizer

Returns the optimizer part of an ensemble model.
"""
optimizer(ensemble::SemEnsemble) = ensemble.optimizer

"""
Base type for all SEM specifications.
"""
abstract type SemSpecification end

abstract type AbstractParameterTable <: SemSpecification end
