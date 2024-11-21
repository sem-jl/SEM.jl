############################################################################################
### connect to NLopt.jl as backend
############################################################################################

mutable struct NLoptResult
    result::Any
    problem::Any
end

optimizer(res::NLoptResult) = res.problem.algorithm
n_iterations(res::NLoptResult) = res.problem.numevals
convergence(res::NLoptResult) = res.result[3]

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
function sem_fit(
    optimizer::SemOptimizerNLopt,
    model::AbstractSem;
    start_val = start_val,
    kwargs...,
)

    # starting values
    if !isa(start_val, AbstractVector)
        start_val = start_val(model; kwargs...)
    end

    # construct the NLopt problem
    opt = construct_NLopt_problem(
        model.optimizer.algorithm,
        model.optimizer.options,
        length(start_val),
    )
    set_NLopt_constraints!(opt, model.optimizer)
    opt.min_objective =
        (par, G) -> evaluate!(
            eltype(par),
            !isnothing(G) && !isempty(G) ? G : nothing,
            nothing,
            model,
            par,
        )

    if !isnothing(model.optimizer.local_algorithm)
        opt_local = construct_NLopt_problem(
            model.optimizer.local_algorithm,
            model.optimizer.local_options,
            length(start_val),
        )
        opt.local_optimizer = opt_local
    end

    # fit
    result = NLopt.optimize(opt, start_val)

    return SemFit_NLopt(result, model, start_val, opt)
end

############################################################################################
### additional functions
############################################################################################

function construct_NLopt_problem(algorithm, options, npar)
    opt = Opt(algorithm, npar)

    for key in keys(options)
        setproperty!(opt, key, options[key])
    end

    return opt
end

function set_NLopt_constraints!(opt, optimizer::SemOptimizerNLopt)
    for con in optimizer.inequality_constraints
        inequality_constraint!(opt::Opt, con.f, con.tol)
    end
    for con in optimizer.equality_constraints
        equality_constraint!(opt::Opt, con.f, con.tol)
    end
end

############################################################################################
# pretty printing
############################################################################################

function Base.show(io::IO, result::NLoptResult)
    print(io, "Optimizer status: $(result.result[3]) \n")
    print(io, "Minimum:          $(round(result.result[1]; digits = 2)) \n")
    print(io, "Algorithm:        $(result.problem.algorithm) \n")
    print(io, "No. evaluations:  $(result.problem.numevals) \n")
end
