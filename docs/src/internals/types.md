# Type hierarchy

The type hierarchy is implemented in `"src/types.jl"`.

`AbstractSem`: the most abstract type in our package
- `AbstractSemSingle{O, I, L, D} <: AbstractSem` is an abstract parametric type that is a supertype of all single models
    - `Sem`: models that do not need automatic differentiation or finite difference approximation
    - `SemFiniteDiff`: models whose gradients and/or hessians should be computed via finite difference approximation
- `AbstractSemCollection <: AbstractSem` is an abstract supertype of all models that contain multiple `AbstractSem` submodels

Every `AbstractSemSingle` has to have `SemObserved`, `SemImplied`, `SemLoss` and `SemOptimizer` fields (and can have additional fields).

`SemLoss` is a container for multiple `SemLossFunctions`.