# Regularization

## Setup

For ridge regularization, you can simply use `SemRidge` as an additional loss function 
(for example, a model with the loss functions `SemML` and `SemRidge` corresponds to ridge-regularized maximum likelihood estimation).

For lasso, elastic net and (far) beyond, we provide the `ProximalSEM` package. You can install it and load it alongside `StructuralEquationModels`:

```@setup reg
import Pkg
Pkg.add(url = "https://github.com/StructuralEquationModels/ProximalSEM.jl")

using StructuralEquationModels, ProximalSEM
```

```julia
import Pkg
Pkg.add(url = "https://github.com/StructuralEquationModels/ProximalSEM.jl")

using StructuralEquationModels, ProximalSEM
```

!!! warning "ProximalSEM is still WIP"
    The ProximalSEM package does not have any releases yet, and is not well tested - until the first release, use at your own risk and expect interfaces to change without prior notice.

Additionally, you need to install and load `ProximalOperators.jl`:

```@setup reg
using ProximalOperators
```

```julia
Pkg.add("ProximalOperators")

using ProximalOperators
```

## `SemOptimizerProximal`

`ProximalSEM` provides a new "building block" for the optimizer part of a model, called `SemOptimizerProximal`.
It connects our package to the [`ProximalAlgorithms.jl`](https://github.com/JuliaFirstOrder/ProximalAlgorithms.jl) optimization backend, providing so-called proximal optimization algorithms. 
Those can handle, amongst other things, various forms of regularization.

It can be used as

```julia
SemOptimizerProximal(
    algorithm = ProximalAlgorithms.PANOC(),
    options = Dict{Symbol, Any}(),
    operator_g,
    operator_h = nothing
    )
```

The proximal operator (aka the regularization function) can be passed as `operator_g`, available options are listed [here](https://juliafirstorder.github.io/ProximalOperators.jl/stable/functions/).
The available Algorithms are listed [here](https://juliafirstorder.github.io/ProximalAlgorithms.jl/stable/guide/implemented_algorithms/).

## First example - lasso

To show how it works, let's revisit [A first model](@ref):

```@example reg
observed_vars = [:x1, :x2, :x3, :y1, :y2, :y3, :y4, :y5, :y6, :y7, :y8]
latent_vars = [:ind60, :dem60, :dem65]

graph = @StenoGraph begin

    ind60 → fixed(1)*x1 + x2 + x3
    dem60 → fixed(1)*y1 + y2 + y3 + y4
    dem65 → fixed(1)*y5 + y6 + y7 + y8

    dem60 ← ind60
    dem65 ← dem60
    dem65 ← ind60

    _(observed_vars) ↔ _(observed_vars)
    _(latent_vars) ↔ _(latent_vars)

    y1 ↔ label(:cov_15)*y5
    y2 ↔ label(:cov_24)*y4 + label(:cov_26)*y6
    y3 ↔ label(:cov_37)*y7
    y4 ↔ label(:cov_48)*y8
    y6 ↔ label(:cov_68)*y8

end

partable = ParameterTable(
    graph,
    latent_vars = latent_vars, 
    observed_vars = observed_vars
)

data = example_data("political_democracy")

model = Sem(
    specification = partable,
    data = data
)
```

We labeled the covariances between the items because we want to regularize those:

```@example reg
ind = param_indices([:cov_15, :cov_24, :cov_26, :cov_37, :cov_48, :cov_68], model)
```

In the following, we fit the same model with lasso regularization of those covariances.
The lasso penalty is defined as

```math
\sum \lambda_i \lvert \theta_i \rvert
```

From the previously linked [documentation](https://juliafirstorder.github.io/ProximalOperators.jl/stable/functions/#ProximalOperators.NormL1), we find that lasso regularization is named `NormL1` in the `ProximalOperators` package, and that we can pass an array of hyperparameters (`λ`) to control the amount of regularization for each parameter. To regularize only the observed item covariances, we define `λ` as

```@example reg
λ = zeros(31); λ[ind] .= 0.02
```

and use `SemOptimizerProximal`.

```@example reg
optimizer_lasso = SemOptimizerProximal(
    operator_g = NormL1(λ)
    )

model_lasso = Sem(
    specification = partable,
    data = data,
    optimizer = optimizer_lasso
)
```

Let's fit the regularized model

```@example reg

fit_lasso = sem_fit(model_lasso)
```

and compare the solution to unregularizted estimates:

```@example reg
fit = sem_fit(model)

update_estimate!(partable, fit)

update_partable!(partable, :estimate_lasso, params(fit_lasso), solution(fit_lasso))

sem_summary(partable)
```

## Second example - mixed l1 and l0 regularization

You can choose to penalize different parameters with different types of regularization functions.
Let's use the lasso again on the covariances, but additionally penalyze the error variances of the observed items via l0 regularization.

The l0 penalty is defined as
```math
\lambda \mathrm{nnz}(\theta)
```

To define a sup of separable proximal operators (i.e. no parameter is penalized twice),
we can use [`SlicedSeparableSum`](https://juliafirstorder.github.io/ProximalOperators.jl/stable/calculus/#ProximalOperators.SlicedSeparableSum) from the `ProximalOperators` package:

```@example reg
prox_operator = SlicedSeparableSum((NormL1(0.02), NormL0(20.0), NormL0(0.0)), ([ind], [12:22], [vcat(1:11, 23:25)]))

model_mixed = Sem(
    specification = partable,
    data = data,
    optimizer = SemOptimizerProximal,
    operator_g = prox_operator
)

fit_mixed = sem_fit(model_mixed)
```

Let's again compare the different results:

```@example reg
update_partable!(partable, :estimate_mixed, params(fit_mixed), solution(fit_mixed))

sem_summary(partable)
```