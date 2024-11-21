# Graph interface

## Workflow 
As discussed before, when using the graph interface, you can specify your model as a graph

```julia
graph = @StenoGraph begin
    ...
end
```

and convert it to a ParameterTable to construct your models:

```julia
observed_vars = ...
latent_vars   = ...

partable = ParameterTable(
    graph,
    latent_vars = latent_vars, 
    observed_vars = observed_vars)

model = Sem(
    specification = partable,
    ...
)
```

## Parameters

In general, there are two different types of parameters: **directed** and **indirected** parameters. A directed parameter from the variable `x` to `y` can be specified as `x → y` (or equivalently as `y ← x`); an undirected parameter as `x ↔ y`.
We allow multiple variables on both sides of an arrow, for example `x → [y z]` or `[a b] → [c d]`. The later specifies element wise edges; that is its the same as `a → c; b → d`. If you want edges corresponding to the cross-product, we have the double lined arrow `[a b] ⇒ [c d]`, corresponding to `a → c; a → d; b → c; b → d`. The undirected arrows ↔ (element-wise) and ⇔ (crossproduct) behave the same way.

!!! note "Unicode symbols in julia"
    The `→` symbol is a unicode symbol allowed in julia (among many others; see this [list](https://docs.julialang.org/en/v1/manual/unicode-input/)). You can enter it in the julia REPL or the vscode IDE by typing `\to` followed by hitting `tab`. Similarly, 
    - `←` = `\leftarrow`,
    - `↔` = `\leftrightarrow`,
    - `⇒` = `\Rightarrow`,
    - `⇐` = `\Leftarrow`,
    - `⇔` = `\Leftrightarrow`
    This may seem cumbersome at first, but with some practice allows you to specify your models in a really elegant way:
    `[x₁ x₂ x₃] ← ξ → η → [y₁ y₂ y₃]`.

## Options
The graph syntax allows you to fix parameters to specific values, label them, and encode equality constraints by giving different parameters the same label. The following syntax example

```julia
graph = @StenoGraph begin

    ξ₁ → fixed(1.0)*x1 + x2 + label(:a)*x3
    ξ₂ → fixed(1.0)*x4 + x5 + label(:λ₁)*x6
    ξ₃ → fixed(NaN)*x7 + x8 + label(:λ₁)*x9

    ξ₃ ↔ fixed(1.0)*ξ₃
end
```
would 
- fix the directed effects from `ξ₁` to `x1` and from `ξ₂` to `x2` to `1`
- leave the directed effect from `ξ₃` to `x7` free but instead restrict the variance of `ξ₃` to `1`
- give the effect from `ξ₁` to `x3` the label `:a` (which can be convenient later if you want to retrieve information from your model about that specific parameter)
- constrain the effect from `ξ₂` to `x6` and `ξ₃` to `x9` to be equal as they are both labeled the same.

## Using variables inside the graph specification
As you saw above and in the [A first model](@ref) example, the graph object needs to be converted to a parameter table:

```julia
partable = ParameterTable(
    graph,
    latent_vars = latent_vars, 
    observed_vars = observed_vars)
```

The `ParameterTable` constructor also needs you to specify a vector of observed and latent variables, in the example above this would correspond to

```julia
observed_vars = [:x1 :x2 :x3 :x4 :x5 :x6 :x7 :x8 :x9]
latent_vars   = [:ξ₁ :ξ₂ :ξ₃]
```

The variable names (`:x1`) have to be symbols, the syntax `:something` creates an object of type `Symbol`. But you can also use vectors of symbols inside the graph specification, escaping them with `_(...)`. For example, this graph specification

```julia
@StenoGraph begin
    _(observed_vars) ↔ _(observed_vars)
    _(latent_vars) ⇔ _(latent_vars)
end
```
creates undirected effects coresponding to 
1. the variances of all observed variables and
2. the variances plus covariances of all latent variables
So if you want to work with a subset of variables, simply specify a vector of symbols `somevars = [...]`, and inside the graph specification, refer to them as `_(somevars)`.

## Meanstructure
Mean parameters are specified as a directed effect from `1` to the respective variable. In our example above, to estimate a mean parameter for all observed variables, we may write

```julia
@StenoGraph begin
    Symbol("1") → _(observed_vars)
end
```

## Further Reading

### What's this strange looking `@`-thing?
The syntax to specify graphs (`@StenoGraph`) may seem a bit strange if you are not familiar with the julia language. It is called a **macro**, but explaining this concept in detail is beyond this documentation (and not necessary to understand to specify models). However, if you want to know more about it, you may have a look at the respective part of the [manual](https://docs.julialang.org/en/v1/manual/metaprogramming/#man-macros).

### The StenoGraphs Package
Behind the scenes, we are using the [StenoGraphs](https://github.com/aaronpeikert/StenoGraphs.jl) package to specify our graphs. It makes a domain specific language available that allows you to specify graphs with arbitrary information attached to its edges and nodes (for structural equation models, this may be the name or the value of a parameter). Is also allows you to specify your own types to "attach" to the graph, called a `Modifier`. So if you contemplate about writing your own modifier (e.g., to mark a variable as ordinal, an effect as quadratic, ...), please refer to the `StenoGraphs` [documentation](https://aaronpeikert.github.io/StenoGraphs.jl/dev/).