# Model sorting

In RAM notation, the model implied covariance matrix is computed as

```math
\Sigma = F(I-A)^{-1}S(I-A)^{-T}F^T
```

If the model is acyclic, the observed and latent variables can be reordered such that ``(I-A)`` is lower triangular. This has the computational benefit that the inversion of lower triangular matrices can be carried out by specialized algorithms.

To automatically reorder your variables in a way that makes this optimization possible, we provide a `sort!` method that can be applied to `ParameterTable` objects to sort the observed and latent variables from the most exogenous ones to the most endogenous.

We use it as

```julia
sort_vars!(parameter_table)

model = Sem(
    specification = parameter_table,
    ...
)
```

Models specified from sorted parameter tables will make use of the described optimizations.