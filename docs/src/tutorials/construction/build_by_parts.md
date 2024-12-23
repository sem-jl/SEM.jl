# Build by parts

You can always build a model by parts - that is, you construct the observed, implied, loss and optimizer part seperately.

As an example on how this works, we will build [A first model](@ref) in parts.

First, we specify the model just as usual:

```@example build
using StructuralEquationModels

data = example_data("political_democracy")

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
    observed_vars = observed_vars)
```

Now, we construct the different parts:

```@example build
# observed ---------------------------------------------------------------------------------
observed = SemObservedData(specification = partable, data = data)

# implied ------------------------------------------------------------------------------------
implied_ram = RAM(specification = partable)

# loss -------------------------------------------------------------------------------------
ml = SemML(observed = observed)

loss_ml = SemLoss(ml)

# optimizer -------------------------------------------------------------------------------------
optimizer = SemOptimizerOptim()

# model ------------------------------------------------------------------------------------

model_ml = Sem(observed, implied_ram, loss_ml, optimizer)
```