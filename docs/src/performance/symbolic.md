# Symbolic precomputation

In RAM notation, the model implied covariance matrix is computed as

```math
\Sigma = F(I-A)^{-1}S(I-A)^{-T}F^T
```

If the model is acyclic, we can compute

```math
(I-A)^{-1} = \sum_{k = 0}^n A^k
```

for some ``n < \infty``.
Typically, the ``S`` and ``A`` matrices are sparse. In our package, we offer symbolic precomputation of ``\Sigma``, ``\nabla\Sigma`` and even ``\nabla^2\Sigma`` for acyclic models to optimally exploit this sparsity. To use this feature, simply use the `RAMSymbolic` implied type for your model.

This can decrase model fitting time, but will also increase model building time (as we have to carry out the symbolic computations and compile specialised functions). As a result, this is probably not beneficial to use if you only fit a single model, but can lead to great improvements if you fit the same modle to multiple datasets (e.g. to compute bootstrap standard errors).