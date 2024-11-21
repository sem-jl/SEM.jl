# Outer Constructor

We already have seen the outer constructor in action in [A first model](@ref):

```julia
model = Sem(
    specification = partable,
    data = data
)

# output

Structural Equation Model
- Loss Functions
   SemML
- Fields
   observed:  SemObservedCommon
   imply:     RAM
   optimizer: SemOptimizerOptim
```

The output of this call tells you exactly what model you just constructed (i.e. what the loss functions, observed, imply and optimizer parts are).

As you can see, by default, we use maximum likelihood estimation, the RAM imply type and the `Optim.jl` optimization backend. 
To choose something different, you can provide it as a keyword argument:

```julia
model = Sem(
    specification = partable,
    data = data,
    observed = ...,
    imply = ...,
    loss = ...,
    optimizer = ...
)
```

For example, to construct a model for weighted least squares estimation that uses symbolic precomputation and the NLopt backend, write

```julia
model = Sem(
    specification = partable,
    data = data,
    imply = RAMSymbolic,
    loss = SemWLS,
    optimizer = SemOptimizerNLopt
)
```

In the section on [Our Concept of a Structural Equation Model](@ref), we go over the different options you have for each part of the model, and in [API - model parts](@ref) we explain each option in detail.
Let's make another example: to use full information maximum likelihood information (FIML), we use

```julia
model = Sem(
    specification = partable,
    data = data,
    loss = SemFIML,
    observed = SemObservedMissing,
    meanstructure = true
)
```

You may also provide addition arguments for specific parts of the model. For example, WLS estimation uses per default

```math
W = \frac{1}{2} D^T(S^{-1}\otimes S^{-1})D
```
as the weight matrix, where D is the so-called duplication matrix and S is the observed covariance matrix. However, you can pass any other weight matrix you want (e.g., UWL, DWLS, ADF estimation) as a keyword argument:

```julia
W = ...

model = Sem(
    specification = partable,
    data = data,
    imply = RAMSymbolic,
    loss = SemWLS,
    wls_weight_matrix = W
)

```

To see what additional keyword arguments are supported, you can consult the documentation of the specific part of the model (either in the REPL by typing `?` to enter the help mode and then typing the name of the thing you want to know something about, or in the online section [API - model parts](@ref)):

```julia
julia>?

help>SemObservedMissing

# output

  For observed data with missing values.

  Constructor
  ≡≡≡≡≡≡≡≡≡≡≡≡≡

  SemObservedMissing(;
      specification,
      data,
      obs_colnames = nothing,
      kwargs...)

  Arguments
  ≡≡≡≡≡≡≡≡≡≡≡

    •  specification: either a RAMMatrices or ParameterTable object (1)

    •  data: observed data

    •  obs_colnames::Vector{Symbol}: column names of the data (if the object passed as data does not have column names, i.e. is not a data frame)

  ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
Extended help is available with `??`
```

## Optimize loss functions without analytic gradient

For loss functions without analytic gradients, it is possible to use finite difference approximation or automatic differentiation.
All loss functions provided in the package do have analytic gradients (and some even hessians or approximations thereof), so there is no need do use this feature if you are only working with them.
However, if you implement your own loss function, you do not have to provide analytic gradients.
This page is a about finite difference approximation. For information about how to use automatic differentiation, see the documentation of the [AutoDiffSEM](https://github.com/StructuralEquationModels/AutoDiffSEM) package.

To use finite difference approximation, you may construct your model just as before, but swap the `Sem` constructor for `SemFiniteDiff`. For example

```julia
model = SemFiniteDiff(
    specification = partable,
    data = data
)
```

constructs a model that will use finite difference approximation if you estimate the parameters via `sem_fit(model)`.