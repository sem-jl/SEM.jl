############################################################################################
### Types
############################################################################################
"""
Empty placeholder for models that don't need an implied part.
(For example, models that only regularize parameters.)

# Constructor

    ImpliedEmpty(;specification, kwargs...)

# Arguments
- `specification`: either a `RAMMatrices` or `ParameterTable` object

# Examples
A multigroup model with ridge regularization could be specified as a `SemEnsemble` with one
model per group and an additional model with `ImpliedEmpty` and `SemRidge` for the regularization part.

# Extended help

## Interfaces
- `params(::RAMSymbolic) `-> Vector of parameter labels
- `nparams(::RAMSymbolic)` -> Number of parameters

## Implementation
Subtype of `SemImplied`.
"""
struct ImpliedEmpty{V2} <: SemImplied
    hessianeval::ExactHessian
    meanstruct::NoMeanStruct
    ram_matrices::V2
end

############################################################################################
### Constructors
############################################################################################

function ImpliedEmpty(; specification, kwargs...)
    return ImpliedEmpty(hessianeval, meanstruct, convert(RAMMatrices, specification))
end

############################################################################################
### methods
############################################################################################

update!(targets::EvaluationTargets, implied::ImpliedEmpty, par, model) = nothing

############################################################################################
### Recommended methods
############################################################################################

update_observed(implied::ImpliedEmpty, observed::SemObserved; kwargs...) = implied
