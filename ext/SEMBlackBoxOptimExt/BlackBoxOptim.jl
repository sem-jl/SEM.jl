############################################################################################
### connect to BlackBoxOptim.jl as backend
############################################################################################

"""
"""
struct SemOptimizerBlackBoxOptim <: SemOptimizer{:BlackBoxOptim}
    lower_bound::Float64 # default lower bound
    variance_lower_bound::Float64 # default variance lower bound
    lower_bounds::Union{Dict{Symbol, Float64}, Nothing}

    upper_bound::Float64 # default upper bound
    upper_bounds::Union{Dict{Symbol, Float64}, Nothing}
end

function SemOptimizerBlackBoxOptim(;
    lower_bound::Float64 = -1000.0,
    lower_bounds::Union{AbstractDict{Symbol, Float64}, Nothing} = nothing,
    variance_lower_bound::Float64 = 0.001,
    upper_bound::Float64 = 1000.0,
    upper_bounds::Union{AbstractDict{Symbol, Float64}, Nothing} = nothing,
    kwargs...,
)
    if variance_lower_bound < 0.0
        throw(ArgumentError("variance_lower_bound must be non-negative"))
    end
    return SemOptimizerBlackBoxOptim(
        lower_bound,
        variance_lower_bound,
        lower_bounds,
        upper_bound,
        upper_bounds,
    )
end

SEM.SemOptimizer{:BlackBoxOptim}(args...; kwargs...) =
    SemOptimizerBlackBoxOptim(args...; kwargs...)

SEM.algorithm(optimizer::SemOptimizerBlackBoxOptim) = optimizer.algorithm
SEM.options(optimizer::SemOptimizerBlackBoxOptim) = optimizer.options

struct SemModelBlackBoxOptimProblem{M <: AbstractSem} <:
       OptimizationProblem{ScalarFitnessScheme{true}}
    model::M
    fitness_scheme::ScalarFitnessScheme{true}
    search_space::ContinuousRectSearchSpace
end

function BlackBoxOptim.search_space(model::AbstractSem)
    optim = model.optimizer::SemOptimizerBlackBoxOptim
    varparams = Set(SEM.variance_params(model.implied.ram_matrices))
    return ContinuousRectSearchSpace(
        [
            begin
                def = in(p, varparams) ? optim.variance_lower_bound : optim.lower_bound
                isnothing(optim.lower_bounds) ? def : get(optim.lower_bounds, p, def)
            end for p in SEM.params(model)
        ],
        [
            begin
                def = optim.upper_bound
                isnothing(optim.upper_bounds) ? def : get(optim.upper_bounds, p, def)
            end for p in SEM.params(model)
        ],
    )
end

function SemModelBlackBoxOptimProblem(
    model::AbstractSem,
    optimizer::SemOptimizerBlackBoxOptim,
)
    SemModelBlackBoxOptimProblem(model, ScalarFitnessScheme{true}(), search_space(model))
end

BlackBoxOptim.fitness(params::AbstractVector, wrapper::SemModelBlackBoxOptimProblem) =
    return SEM.evaluate!(0.0, nothing, nothing, wrapper.model, params)

# sem_fit method
function SEM.sem_fit(
    optimizer::SemOptimizerBlackBoxOptim,
    model::AbstractSem,
    start_params::AbstractVector;
    MaxSteps::Integer = 50000,
    kwargs...,
)
    problem = SemModelBlackBoxOptimProblem(model, optimizer)
    res = bboptimize(problem; MaxSteps, kwargs...)
    return SemFit(best_fitness(res), best_candidate(res), nothing, model, res)
end
