# API methods supported by multiple SEM.jl types

"""
    nparams(semobj)

Return the number of parameters in a SEM model associated with `semobj`.

See also [`params`](@ref).
"""
nparams(semobj) = length(params(semobj))

"""
    nvars(semobj)

Return the number of variables in a SEM model associated with `semobj`.

See also [`vars`](@ref).
"""
nvars(semobj) = length(vars(semobj))

"""
    nobserved_vars(semobj)

Return the number of observed variables in a SEM model associated with `semobj`.
"""
nobserved_vars(semobj) = length(observed_vars(semobj))

"""
    nlatent_vars(semobj)

Return the number of latent variables in a SEM model associated with `semobj`.
"""
nlatent_vars(semobj) = length(latent_vars(semobj))

"""
    param_indices(semobj)

Returns a dict of parameter names and their indices in `semobj`.

# Examples
```julia
parind = param_indices(my_fitted_sem)
parind[:param_name]
```

See also [`params`](@ref).
"""
param_indices(semobj) = Dict(par => i for (i, par) in enumerate(params(semobj)))

"""
    nsamples(semobj)

Return the number of samples (observed data points).

For ensemble models, return the sum over all submodels.
"""
function nsamples end
