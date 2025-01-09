############################################################################################
### Types
############################################################################################
"""
Connects to `ProximalAlgorithms.jl` as the optimization backend.

# Constructor

    SemOptimizerProximal(;
        algorithm = ProximalAlgorithms.PANOC(),
        operator_g,
        operator_h = nothing,
        kwargs...,

# Arguments
- `algorithm`: optimization algorithm.
- `operator_g`: gradient of the objective function
- `operator_h`: optional hessian of the objective function
"""
mutable struct SemOptimizerProximal{A, B, C} <: SemOptimizer{:Proximal}
    algorithm::A
    operator_g::B
    operator_h::C
end

SEM.SemOptimizer{:Proximal}(args...; kwargs...) = SemOptimizerProximal(args...; kwargs...)

SemOptimizerProximal(;
    algorithm = ProximalAlgorithms.PANOC(),
    operator_g,
    operator_h = nothing,
    kwargs...,
) = SemOptimizerProximal(algorithm, operator_g, operator_h)

############################################################################################
### Recommended methods
############################################################################################

SEM.update_observed(optimizer::SemOptimizerProximal, observed::SemObserved; kwargs...) =
    optimizer

############################################################################################
### additional methods
############################################################################################

SEM.algorithm(optimizer::SemOptimizerProximal) = optimizer.algorithm

############################################################################
### Pretty Printing
############################################################################

function Base.show(io::IO, struct_inst::SemOptimizerProximal)
    print_type_name(io, struct_inst)
    print_field_types(io, struct_inst)
end

## connect to ProximalAlgorithms.jl
function ProximalAlgorithms.value_and_gradient(model::AbstractSem, params)
    grad = similar(params)
    obj = SEM.evaluate!(zero(eltype(params)), grad, nothing, model, params)
    return obj, grad
end

#ProximalCore.prox!(y, f, x, gamma) = ProximalOperators.prox!(y, f, x, gamma)

mutable struct ProximalResult
    result::Any
end

function SEM.sem_fit(
    optim::SemOptimizerProximal,
    model::AbstractSem,
    start_params::AbstractVector;
    kwargs...,
)
    if isnothing(optim.operator_h)
        solution, iterations =
            optim.algorithm(x0 = start_params, f = model, g = optim.operator_g)
    else
        solution, iterations = optim.algorithm(
            x0 = start_params,
            f = model,
            g = optim.operator_g,
            h = optim.operator_h,
        )
    end

    minimum = objective!(model, solution)

    optimization_result = Dict(
        :minimum => minimum,
        :iterations => iterations,
        :algorithm => optim.algorithm,
        :operator_g => optim.operator_g,
    )

    isnothing(optim.operator_h) ||
        push!(optimization_result, :operator_h => optim.operator_h)

    return SemFit(
        minimum,
        solution,
        start_params,
        model,
        ProximalResult(optimization_result),
    )
end

############################################################################################
# pretty printing
############################################################################################

function Base.show(io::IO, result::ProximalResult)
    print(io, "Minimum:          $(round(result.result[:minimum]; digits = 2)) \n")
    print(io, "No. evaluations:  $(result.result[:iterations]) \n")
    print(io, "Operator:         $(nameof(typeof(result.result[:operator_g]))) \n")
    if haskey(result.result, :operator_h)
        print(io, "Second Operator:  $(nameof(typeof(result.result[:operator_h]))) \n")
    end
end
