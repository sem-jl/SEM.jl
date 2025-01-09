# Custom observed types

The implementation of new observed types is very similar to loss functions, so we will just go over it briefly (for additional information, revisit [Custom loss functions](@ref)).

First, we need to define a new struct that is a subtype of `SemObserved`:

```julia
struct MyObserved <: SemObserved
    ...
end
```

Additionally, we can write an outer constructor that will typically depend on the keyword argument `data = ...`:

```julia
function MyObserved(;data, kwargs...)
    ...
    return MyObserved(...)
end
```

To compute some fit indices, you need to provide methods for

```julia
# Number of samples (observations) in the dataset
nsamples(observed::MyObserved) = ...
# Number of observed variables
nobserved_vars(observed::MyObserved) = ...
```

As always, you can add additional methods for properties that implied types and loss function want to access, for example (from the `SemObservedCommon` implementation):

```julia
obs_cov(observed::SemObservedCommon) = observed.obs_cov
```