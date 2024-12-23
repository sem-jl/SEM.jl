# Models with mean structures

To make use of mean structures in your model, you have to
1. Specify your model with a mean structure. The sections [Graph interface](@ref) and [RAMMatrices interface](@ref) both explain how this works.
2. Build your model with a meanstructure. We explain how that works in the following.

Lets say you specified [A first model](@ref) as a graph with a meanstructure:

```@setup meanstructure
using StructuralEquationModels

observed_vars = [:x1, :x2, :x3, :y1, :y2, :y3, :y4, :y5, :y6, :y7, :y8]
latent_vars = [:ind60, :dem60, :dem65]

graph = @StenoGraph begin

    # loadings
    ind60 → fixed(1)*x1 + x2 + x3
    dem60 → fixed(1)*y1 + y2 + y3 + y4
    dem65 → fixed(1)*y5 + y6 + y7 + y8

    # latent regressions
    dem60 ← ind60
    dem65 ← dem60
    dem65 ← ind60

    # variances
    _(observed_vars) ↔ _(observed_vars)
    _(latent_vars) ↔ _(latent_vars)

    # covariances
    y1 ↔ y5
    y2 ↔ y4 + y6
    y3 ↔ y7
    y8 ↔ y4 + y6

    # means
    Symbol("1") → _(observed_vars)
end

partable = ParameterTable(
    graph,
    latent_vars = latent_vars, 
    observed_vars = observed_vars)
```

```julia
using StructuralEquationModels

observed_vars = [:x1, :x2, :x3, :y1, :y2, :y3, :y4, :y5, :y6, :y7, :y8]
latent_vars = [:ind60, :dem60, :dem65]

graph = @StenoGraph begin

    # loadings
    ind60 → fixed(1)*x1 + x2 + x3
    dem60 → fixed(1)*y1 + y2 + y3 + y4
    dem65 → fixed(1)*y5 + y6 + y7 + y8

    # latent regressions
    dem60 ← ind60
    dem65 ← dem60
    dem65 ← ind60

    # variances
    _(observed_vars) ↔ _(observed_vars)
    _(latent_vars) ↔ _(latent_vars)

    # covariances
    y1 ↔ y5
    y2 ↔ y4 + y6
    y3 ↔ y7
    y8 ↔ y4 + y6

    # means
    Symbol("1") → _(observed_vars)
end

partable = ParameterTable(
    graph,
    latent_vars = latent_vars, 
    observed_vars = observed_vars)
```

that is, all observed variable means are estimated freely.

To build the model with a meanstructure, we proceed as usual, but pass the argument `meanstructure = true`.
For our example,

```@example meanstructure
data = example_data("political_democracy")

model = Sem(
    specification = partable,
    data = data,
    meanstructure = true
)

sem_fit(model)
```

If we build the model by parts, we have to pass the `meanstructure = true` argument to every part that requires it (when in doubt, simply comsult the documentation for the respective part).

For our example,

```@example meanstructure
observed = SemObservedData(specification = partable, data = data, meanstructure = true)

implied_ram = RAM(specification = partable, meanstructure = true)

ml = SemML(observed = observed, meanstructure = true)

model = Sem(observed, implied_ram, SemLoss(ml), SemOptimizerOptim())

sem_fit(model)
```