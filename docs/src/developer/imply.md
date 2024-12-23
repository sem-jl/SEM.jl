# Custom implied types

We recommend to first read the part [Custom loss functions](@ref), as the overall implementation is the same and we will describe it here more briefly.

Implied types are of subtype `SemImplied`. To implement your own implied type, you should define a struct

```julia
struct MyImplied <: SemImplied
    ...
end
```

and at least a method to compute the objective

```julia
import StructuralEquationModels: objective!

function objective!(implied::MyImplied, par, model::AbstractSemSingle)
    ...
    return nothing
end
```

This method should compute and store things you want to make available to the loss functions, and returns `nothing`. For example, as we have seen in [Second example - maximum likelihood](@ref), the `RAM` implied type computes the model-implied covariance matrix and makes it available via `Σ(implied)`.
To make stored computations available to loss functions, simply write a function - for example, for the `RAM` implied type we defined

```julia
Σ(implied::RAM) = implied.Σ
```

Additionally, you can specify methods for `gradient` and `hessian` as well as the combinations described in [Custom loss functions](@ref).

The last thing nedded to make it work is a method for `nparams` that takes your implied type and returns the number of parameters of the model:

```julia
nparams(implied::MyImplied) = ...
```

Just as described in [Custom loss functions](@ref), you may define a constructor. Typically, this will depend on the `specification = ...` argument that can be a `ParameterTable` or a `RAMMatrices` object.

We implement an `ImpliedEmpty` type in our package that does nothing but serving as an `implied` field in case you are using a loss function that does not need any implied type at all. You may use it as a template for defining your own implied type, as it also shows how to handle the specification objects:

```julia
############################################################################
### Types
############################################################################

struct ImpliedEmpty{V, V2} <: SemImplied
    identifier::V2
    n_par::V
end

############################################################################
### Constructors
############################################################################

function ImpliedEmpty(;
        specification,
        kwargs...)

        ram_matrices = RAMMatrices(specification)
        identifier = StructuralEquationModels.identifier(ram_matrices)

        n_par = length(ram_matrices.parameters)

        return ImpliedEmpty(identifier, n_par)
end

############################################################################
### methods
############################################################################

objective!(implied::ImpliedEmpty, par, model) = nothing
gradient!(implied::ImpliedEmpty, par, model) = nothing
hessian!(implied::ImpliedEmpty, par, model) = nothing

############################################################################
### Recommended methods
############################################################################

identifier(implied::ImpliedEmpty) = implied.identifier
n_par(implied::ImpliedEmpty) = implied.n_par

update_observed(implied::ImpliedEmpty, observed::SemObserved; kwargs...) = implied
```

As you see, similar to [Custom loss functions](@ref) we implement a method for `update_observed`. Additionally, you should store the `identifier` from the specification object and write a method for `identifier`, as this will make it possible to access parameter indices by label.