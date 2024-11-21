"""
    samples(observed::SemObservedData)

Gets the matrix of observed data samples.
Rows are samples, columns are observed variables.

## See Also
[`nsamples`](@ref), [`observed_vars`](@ref).
"""
samples(observed::SemObserved) = observed.data
