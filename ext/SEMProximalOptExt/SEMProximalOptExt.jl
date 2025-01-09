module SEMProximalOptExt

using StructuralEquationModels, Printf
using ProximalCore, ProximalAlgorithms, ProximalOperators

export SemOptimizerProximal

SEM = StructuralEquationModels

#ProximalCore.prox!(y, f, x, gamma) = ProximalOperators.prox!(y, f, x, gamma)

include("ProximalAlgorithms.jl")

end
