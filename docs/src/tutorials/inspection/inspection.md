# Model inspection

```@setup colored
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
    observed_vars = observed_vars)

data = example_data("political_democracy")

model = Sem(
    specification = partable,
    data = data
)

model_fit = sem_fit(model)
```

After you fitted a model,

```julia
model_fit = sem_fit(model)
```

you end up with an object of type [`SemFit`](@ref).

You can get some more information about it by using the `sem_summary` function:

```@example colored; ansicolor = true
sem_summary(model_fit)
```

To compute fit measures, we use

```@example colored; ansicolor = true
fit_measures(model_fit)
```

or compute them individually:

```@example colored; ansicolor = true
AIC(model_fit)
```

A list of available [Fit measures](@ref) is at the end of this page.

To inspect the parameter estimates, we can update a `ParameterTable` object and call `sem_summary` on it:

```@example colored; ansicolor = true; output = false
update_estimate!(partable, model_fit)

sem_summary(partable)
```

We can also update the `ParameterTable` object with other information via [`update_partable!`](@ref). For example, if we want to compare hessian-based and bootstrap-based standard errors, we may write

```@example colored; ansicolor = true
se_bs = se_bootstrap(model_fit; n_boot = 20)
se_he = se_hessian(model_fit)

update_partable!(partable, :se_hessian, params(model_fit), se_he)
update_partable!(partable, :se_bootstrap, params(model_fit), se_bs)

sem_summary(partable)
```

## Export results

You may convert a `ParameterTable` to a `DataFrame` and use the [`DataFrames`](https://github.com/JuliaData/DataFrames.jl) package for further analysis (or to save it to your hard drive).

```@example colored; ansicolor = true
using DataFrames

parameters_df = DataFrame(partable)
```

# API - model inspection

```@docs
sem_summary
update_estimate!
update_partable!
```

## Additional functions

Additional functions that can be used to extract information from a `SemFit` object:

```@docs
SemFit
```

## Fit measures

```@docs
fit_measures
AIC
BIC
χ²
df
minus2ll
n_man
n_obs
nparams
p_value
RMSEA
```
