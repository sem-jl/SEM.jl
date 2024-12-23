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
    return ContinuousRectSearchSpace(
        SEM.lower_bounds(
            optim.lower_bounds,
            model,
            default = optim.lower_bound,
            variance_default = optim.variance_lower_bound,
        ),
        SEM.upper_bounds(optim.upper_bounds, model, default = optim.upper_bound),
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
    Method::Symbol = :adaptive_de_rand_1_bin_with_gradient,
    MaxSteps::Integer = 50000,
    kwargs...,
)
    problem = SemModelBlackBoxOptimProblem(model, optimizer)
    if Method == :adaptive_de_rand_1_bin_with_gradient
        # custom adaptive differential evolution with mutation that moves along the gradient
        bbopt_factory = DefaultDiffEvoFactory(problem; kwargs...)
        bbopt = bbsetup(bbopt_factory; MaxSteps, kwargs...)
    else
        bbopt = bbsetup(problem; Method, MaxSteps, kwargs...)
    end
    res = bboptimize(bbopt)
    return SemFit(best_fitness(res), best_candidate(res), nothing, model, res)
end
