# Custom imply types

We recommend to first read the part [Custom loss functions](@ref), as the overall implementation is the same and we will describe it here more briefly.

Imply types are of subtype `SemImply`. To implement your own imply type, you should define a struct

```julia
struct MyImply <: SemImply
    ...
end
```

and at least a method to compute the objective

```julia
import StructuralEquationModels: objective!

function objective!(imply::MyImply, par, model::AbstractSemSingle)
    ...
    return nothing
end
```

This method should compute and store things you want to make available to the loss functions, and returns `nothing`. For example, as we have seen in [Second example - maximum likelihood](@ref), the `RAM` imply type computes the model-implied covariance matrix and makes it available via `Σ(imply)`.
To make stored computations available to loss functions, simply write a function - for example, for the `RAM` imply type we defined

```julia
Σ(imply::RAM) = imply.Σ
```

Additionally, you can specify methods for `gradient` and `hessian` as well as the combinations described in [Custom loss functions](@ref).

The last thing nedded to make it work is a method for `nparams` that takes your imply type and returns the number of parameters of the model:

```julia
nparams(imply::MyImply) = ...
```

Just as described in [Custom loss functions](@ref), you may define a constructor. Typically, this will depend on the `specification = ...` argument that can be a `ParameterTable` or a `RAMMatrices` object.

We implement an `ImplyEmpty` type in our package that does nothing but serving as an imply field in case you are using a loss function that does not need any imply type at all. You may use it as a template for defining your own imply type, as it also shows how to handle the specification objects:

```julia
############################################################################
### Types
############################################################################

struct ImplyEmpty{V, V2} <: SemImply
    identifier::V2
    n_par::V
end

############################################################################
### Constructors
############################################################################

function ImplyEmpty(;
        specification,
        kwargs...)

        ram_matrices = RAMMatrices(specification)
        identifier = StructuralEquationModels.identifier(ram_matrices)

        n_par = length(ram_matrices.parameters)

        return ImplyEmpty(identifier, n_par)
end

############################################################################
### methods
############################################################################

objective!(imply::ImplyEmpty, par, model) = nothing
gradient!(imply::ImplyEmpty, par, model) = nothing
hessian!(imply::ImplyEmpty, par, model) = nothing

############################################################################
### Recommended methods
############################################################################

identifier(imply::ImplyEmpty) = imply.identifier
n_par(imply::ImplyEmpty) = imply.n_par

update_observed(imply::ImplyEmpty, observed::SemObserved; kwargs...) = imply
```

As you see, similar to [Custom loss functions](@ref) we implement a method for `update_observed`. Additionally, you should store the `identifier` from the specification object and write a method for `identifier`, as this will make it possible to access parameter indices by label.