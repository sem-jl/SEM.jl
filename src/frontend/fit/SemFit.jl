############################################################################################
# struct
############################################################################################
"""
    SemFit

Fitted structural equation model.

# Interfaces
- `minimum(::SemFit)` -> minimum objective value
- `solution(::SemFit)` -> parameter estimates
- `start_val(::SemFit)` -> starting values
- `model(::SemFit)`
- `optimization_result(::SemFit)`

- `optimizer(::SemFit)` -> optimization algorithm
- `n_iterations(::SemFit)` -> number of iterations
- `convergence(::SemFit)` -> convergence properties
"""
mutable struct SemFit{Mi, So, St, Mo, O}
    minimum::Mi
    solution::So
    start_val::St
    model::Mo
    optimization_result::O
end

############################################################################################
# pretty printing
############################################################################################

function Base.show(io::IO, semfit::SemFit)
    print(io, "Fitted Structural Equation Model \n")
    print(io, "=============================================== \n")
    print(io, "--------------------- Model ------------------- \n")
    print(io, "\n")
    print(io, semfit.model)
    print(io, "\n")
    #print(io, "Objective value: $(round(semfit.minimum, digits = 4)) \n")
    print(io, "------------- Optimization result ------------- \n")
    print(io, "\n")
    print(io, semfit.optimization_result)
end

############################################################################################
# additional methods
############################################################################################

params(fit::SemFit) = params(fit.model)
nparams(fit::SemFit) = nparams(fit.model)
nsamples(fit::SemFit) = nsamples(fit.model)

# access fields
minimum(sem_fit::SemFit) = sem_fit.minimum
solution(sem_fit::SemFit) = sem_fit.solution
start_val(sem_fit::SemFit) = sem_fit.start_val
model(sem_fit::SemFit) = sem_fit.model
optimization_result(sem_fit::SemFit) = sem_fit.optimization_result

# optimizer properties
optimizer(sem_fit::SemFit) = optimizer(optimization_result(sem_fit))
n_iterations(sem_fit::SemFit) = n_iterations(optimization_result(sem_fit))
convergence(sem_fit::SemFit) = convergence(optimization_result(sem_fit))
