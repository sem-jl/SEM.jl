module SEMNLOptExt

using StructuralEquationModels, NLopt, Printf

SEM = StructuralEquationModels

export SemOptimizerNLopt, NLoptConstraint

include("NLopt.jl")

end
