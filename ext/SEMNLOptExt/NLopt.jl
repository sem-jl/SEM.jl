############################################################################################
### Types
############################################################################################
"""
Connects to `NLopt.jl` as the optimization backend.

# Constructor

    SemOptimizerNLopt(;
        algorithm = :LD_LBFGS,
        options = Dict{Symbol, Any}(),
        local_algorithm = nothing,
        local_options = Dict{Symbol, Any}(),
        equality_constraints = Vector{NLoptConstraint}(),
        inequality_constraints = Vector{NLoptConstraint}(),
        kwargs...)

# Arguments
- `algorithm`: optimization algorithm.
- `options::Dict{Symbol, Any}`: options for the optimization algorithm
- `local_algorithm`: local optimization algorithm
- `local_options::Dict{Symbol, Any}`: options for the local optimization algorithm
- `equality_constraints::Vector{NLoptConstraint}`: vector of equality constraints
- `inequality_constraints::Vector{NLoptConstraint}`: vector of inequality constraints

# Example
```julia
my_optimizer = SemOptimizerNLopt()

# constrained optimization with augmented lagrangian
my_constrained_optimizer = SemOptimizerNLopt(;
    algorithm = :AUGLAG,
    local_algorithm = :LD_LBFGS,
    local_options = Dict(:ftol_rel => 1e-6),
    inequality_constraints = NLoptConstraint(;f = my_constraint, tol = 0.0),
)
```

# Usage
All algorithms and options from the NLopt library are available, for more information see
the NLopt.jl package and the NLopt online documentation.
For information on how to use inequality and equality constraints,
see [Constrained optimization](@ref) in our online documentation.

# Extended help

## Interfaces
- `algorithm(::SemOptimizerNLopt)`
- `local_algorithm(::SemOptimizerNLopt)`
- `options(::SemOptimizerNLopt)`
- `local_options(::SemOptimizerNLopt)`
- `equality_constraints(::SemOptimizerNLopt)`
- `inequality_constraints(::SemOptimizerNLopt)`

## Implementation

Subtype of `SemOptimizer`.
"""
struct SemOptimizerNLopt{A, A2, B, B2, C} <: SemOptimizer{:NLopt}
    algorithm::A
    local_algorithm::A2
    options::B
    local_options::B2
    equality_constraints::C
    inequality_constraints::C
end

Base.@kwdef struct NLoptConstraint
    f::Any
    tol = 0.0
end

Base.convert(
    ::Type{NLoptConstraint},
    tuple::NamedTuple{(:f, :tol), Tuple{F, T}},
) where {F, T} = NLoptConstraint(tuple.f, tuple.tol)

############################################################################################
### Constructor
############################################################################################

function SemOptimizerNLopt(;
    algorithm = :LD_LBFGS,
    local_algorithm = nothing,
    options = Dict{Symbol, Any}(),
    local_options = Dict{Symbol, Any}(),
    equality_constraints = Vector{NLoptConstraint}(),
    inequality_constraints = Vector{NLoptConstraint}(),
    kwargs...,
)
    applicable(iterate, equality_constraints) && !isa(equality_constraints, NamedTuple) ||
        (equality_constraints = [equality_constraints])
    applicable(iterate, inequality_constraints) &&
        !isa(inequality_constraints, NamedTuple) ||
        (inequality_constraints = [inequality_constraints])
    return SemOptimizerNLopt(
        algorithm,
        local_algorithm,
        options,
        local_options,
        convert.(NLoptConstraint, equality_constraints),
        convert.(NLoptConstraint, inequality_constraints),
    )
end

SEM.SemOptimizer{:NLopt}(args...; kwargs...) = SemOptimizerNLopt(args...; kwargs...)

############################################################################################
### Recommended methods
############################################################################################

SEM.update_observed(optimizer::SemOptimizerNLopt, observed::SemObserved; kwargs...) =
    optimizer

############################################################################################
### additional methods
############################################################################################

SEM.algorithm(optimizer::SemOptimizerNLopt) = optimizer.algorithm
local_algorithm(optimizer::SemOptimizerNLopt) = optimizer.local_algorithm
SEM.options(optimizer::SemOptimizerNLopt) = optimizer.options
local_options(optimizer::SemOptimizerNLopt) = optimizer.local_options
equality_constraints(optimizer::SemOptimizerNLopt) = optimizer.equality_constraints
inequality_constraints(optimizer::SemOptimizerNLopt) = optimizer.inequality_constraints

mutable struct NLoptResult
    result::Any
    problem::Any
end

SEM.optimizer(res::NLoptResult) = res.problem.algorithm
SEM.n_iterations(res::NLoptResult) = res.problem.numevals
SEM.convergence(res::NLoptResult) = res.result[3]

# construct SemFit from fitted NLopt object
function SemFit_NLopt(optimization_result, model::AbstractSem, start_val, opt)
    return SemFit(
        optimization_result[1],
        optimization_result[2],
        start_val,
        model,
        NLoptResult(optimization_result, opt),
    )
end

# sem_fit method
function SEM.sem_fit(
    optim::SemOptimizerNLopt,
    model::AbstractSem,
    start_params::AbstractVector;
    kwargs...,
)

    # construct the NLopt problem
    opt = construct_NLopt_problem(optim.algorithm, optim.options, length(start_params))
    set_NLopt_constraints!(opt, optim)
    opt.min_objective =
        (par, G) -> SEM.evaluate!(
            zero(eltype(par)),
            !isnothing(G) && !isempty(G) ? G : nothing,
            nothing,
            model,
            par,
        )

    if !isnothing(optim.local_algorithm)
        opt_local = construct_NLopt_problem(
            optim.local_algorithm,
            optim.local_options,
            length(start_params),
        )
        opt.local_optimizer = opt_local
    end

    # fit
    result = NLopt.optimize(opt, start_params)

    return SemFit_NLopt(result, model, start_params, opt)
end

############################################################################################
### additional functions
############################################################################################

function construct_NLopt_problem(algorithm, options, npar)
    opt = Opt(algorithm, npar)

    for (key, val) in pairs(options)
        setproperty!(opt, key, val)
    end

    return opt
end

function set_NLopt_constraints!(opt::Opt, optimizer::SemOptimizerNLopt)
    for con in optimizer.inequality_constraints
        inequality_constraint!(opt, con.f, con.tol)
    end
    for con in optimizer.equality_constraints
        equality_constraint!(opt, con.f, con.tol)
    end
end

############################################################################################
# pretty printing
############################################################################################

function Base.show(io::IO, result::NLoptResult)
    print(io,   "Optimizer status: $(result.result[3]) \n")
    @printf(io, "Minimum:          %.4g\n", result.result[1])
    print(io,   "Algorithm:        $(result.problem.algorithm) \n")
    print(io,   "No. evaluations:  $(result.problem.numevals) \n")
end
