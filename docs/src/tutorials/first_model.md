# A first model

In this tutorial, we will fit an example SEM with our package. 
The example we are using is from [the `lavaan` tutorial](https://lavaan.ugent.be/tutorial/sem.html), so it may be familiar.
It looks like this:

![Visualization of the Political Democracy model](../assets/political_democracy.png)

We assume the `StructuralEquationModels` package is already installed. To use it in the current session, we run

```@example high_level
using StructuralEquationModels
```

We then first define the graph of our model in a syntax which is similar to the R-package `lavaan`:

```@setup high_level
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
```

```julia
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
```

!!! note "Time to first model"
    When executing the code from this tutorial the first time in a fresh julia session,
    you may wonder that it takes quite some time. This is not because the implementation is slow,
    but because the functions are compiled the first time you use them.
    Try rerunning the example a second time - you will see that all function executions after the first one
    are quite fast.

We then use this graph to define a `ParameterTable` object

```@example high_level; ansicolor = true
partable = ParameterTable(
    graph,
    latent_vars = latent_vars, 
    observed_vars = observed_vars)
```

load the example data

```julia
data = example_data("political_democracy")
```

```@setup high_level
data = example_data("political_democracy")
```

and specify our model as

```@example high_level; ansicolor = true
model = Sem(
    specification = partable,
    data = data
)
```

We can now fit the model via

```@example high_level; ansicolor = true
model_fit = sem_fit(model)
```

and compute fit measures as

```@example high_level; ansicolor = true
fit_measures(model_fit)
```

We can also get a bit more information about the fitted model via the `sem_summary()` function:

```@example high_level; ansicolor = true
sem_summary(model_fit)
```

To investigate the parameter estimates, we can update our `partable` object to contain the new estimates:

```@example high_level; ansicolor = true
update_estimate!(partable, model_fit)
```

and investigate the solution with

```@example high_level; ansicolor = true
sem_summary(partable)
```

Congratulations, you fitted and inspected your very first model! 
We recommend continuing with [Our Concept of a Structural Equation Model](@ref).