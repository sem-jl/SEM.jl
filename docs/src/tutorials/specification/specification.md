# Model specification

Two things can be used to specify a model: a parameter table or ram matrices.
You can convert them to each other, and to make your life easier, we also provide a way to get parameter tables from graphs.

This leads to the following chart:

![Specification flowchart](../../assets/specification.png)

You can enter model specification at each point, but in general (and especially if you come from `lavaan`), it is the easiest to follow the red arrows: specify a graph object, convert it to a prameter table, and use this parameter table to construct your models ( just like we did in [A first model](@ref)):

```julia
observed_vars = ...
latent_vars   = ...

graph = @StenoGraph begin
    ...
end

partable = ParameterTable(
    graph,
    latent_vars = latent_vars, 
    observed_vars = observed_vars)

model = Sem(
    specification = partable,
    ...
)
```

On the following pages, we explain how to enter the specification process at each step, i.e. how to specify models via the [Graph interface](@ref), the [ParameterTable interface](@ref), and the [RAMMatrices interface](@ref). 
If you have an `OpenMx` background, and are familiar with their way of specifying structural equation models via RAM matrices, the [RAMMatrices interface](@ref) may be of interest for you.