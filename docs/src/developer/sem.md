# Custom model types

The abstract supertype for all models is `AbstractSem`, which has two subtypes, `AbstractSemSingle{O, I, L, D}` and `AbstractSemCollection`. Currently, there are 2 subtypes of `AbstractSemSingle`: `Sem`, `SemFiniteDiff`. All subtypes of `AbstractSemSingle` should have at least observed, implied, loss and optimizer fields, and share their types (`{O, I, L, D}`) with the parametric abstract supertype. For example, the `SemFiniteDiff` type is implemented as

```julia
struct SemFiniteDiff{
        O <: SemObserved,
        I <: SemImplied,
        L <: SemLoss,
        D <: SemOptimizer} <: AbstractSemSingle{O, I, L, D}
    observed::O
    implied::I
    loss::L
    optimizer::D
end
```

Additionally, we need to define a method to compute at least the objective value, and if you want to use gradient based optimizers (which you most probably will), we need also to define a method to compute the gradient. For example, the respective fallback methods for all `AbstractSemSingle` models are defined as

```julia
function objective!(model::AbstractSemSingle, parameters)
    objective!(implied(model), parameters, model)
    return objective!(loss(model), parameters, model)
end

function gradient!(gradient, model::AbstractSemSingle, parameters)
    fill!(gradient, zero(eltype(gradient)))
    gradient!(implied(model), parameters, model)
    gradient!(gradient, loss(model), parameters, model)
end
```

Note that the `gradient!` method takes a pre-allocated array that should be filled with the gradient values.

Additionally, we can define constructors like the one in `"src/frontend/specification/Sem.jl"`.

It is also possible to add new subtypes for `AbstractSemCollection`.