# Simulation studies

!!! note "Simulation study interface"
    We are currently working on an interface for simulation studies.
    Until we are finished with this, this page is just a collection of tips.

## Swap observed data
In simulation studies, a common task is fitting the same model to many different datasets.
It would be a waste of resources to reconstruct the complete model for each dataset.
We therefore provide the function `swap_observed` to change the `observed` part of a model,
without necessarily reconstructing the other parts.

For the [A first model](@ref), you would use it as

```@setup swap_observed
using StructuralEquationModels

observed_vars = [:x1, :x2, :x3, :y1, :y2, :y3, :y4, :y5, :y6, :y7, :y8]
latent_vars = [:ind60, :dem60, :dem65]

graph = @StenoGraph begin

    # loadings
    ind60 → fixed(1)*x1 + x2 + x3
    dem60 → fixed(1)*y1 + y2 + y3 + y4
    dem65 → fixed(1)*y5 + y6 + y7 + y8

    # latent regressions
    ind60 → dem60
    dem60 → dem65
    ind60 → dem65

    # variances
    _(observed_vars) ↔ _(observed_vars)
    _(latent_vars) ↔ _(latent_vars)

    # covariances
    y1 ↔ y5
    y2 ↔ y4 + y6
    y3 ↔ y7
    y8 ↔ y4 + y6

end

partable = ParameterTable(
    graph,
    latent_vars = latent_vars, 
    observed_vars = observed_vars
)
```

```@example swap_observed
data = example_data("political_democracy")

data_1 = data[1:30, :]

data_2 = data[31:75, :]

model = Sem(
    specification = partable,
    data = data_1
)

model_updated = swap_observed(model; data = data_2, specification = partable)
```

!!! danger "Thread safety"
    *This is only relevant when you are planning to fit updated models in parallel*
    
    Models generated this way may share the same objects in memory (e.g. some parts of 
    `model` and `model_updated` are the same objects in memory.)
    Therefore, fitting both of these models in parallel will lead to **race conditions**, 
    possibly crashing your computer.
    To avoid these problems, you should copy `model` before updating it.

If you are building your models by parts, you can also update each part seperately with the function `update_observed`.
For example,

```@example swap_observed

new_observed = SemObservedData(;data = data_2, specification = partable)

my_optimizer = SemOptimizerOptim()

new_optimizer = update_observed(my_optimizer, new_observed)
```

## API

```@docs
swap_observed
update_observed
```