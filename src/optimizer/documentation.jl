"""
    sem_fit(model::AbstractSem; start_val = start_val, kwargs...)

Return the fitted `model`.

# Arguments
- `model`: `AbstractSem` to fit
- `start_val`: vector of starting values or function to compute starting values (1)
- `kwargs...`: keyword arguments, passed to starting value functions

(1) available options are `start_fabin3`, `start_simple` and `start_partable`. 
For more information, we refer to the individual documentations and the online documentation on [Starting values](@ref).

# Examples
```julia
sem_fit(
    my_model; 
    start_val = start_simple,
    start_covariances_latent = 0.5)
```
"""
function sem_fit end

# dispatch on optimizer
sem_fit(model::AbstractSem; kwargs...) = sem_fit(model.optimizer, model; kwargs...)

# fallback method
sem_fit(optimizer::SemOptimizer, model::AbstractSem; kwargs...) =
    error("Optimizer $(optimizer) support not implemented.")
