# (Ridge) regularization

############################################################################################
### Types
############################################################################################
"""
Ridge regularization.

# Constructor

    SemRidge(;α_ridge, which_ridge, nparams, parameter_type = Float64, implied = nothing, kwargs...)

# Arguments
- `α_ridge`: hyperparameter for penalty term
- `which_ridge::Vector`: Vector of parameter labels (Symbols) or indices that indicate which parameters should be regularized.
- `nparams::Int`: number of parameters of the model
- `implied::SemImplied`: implied part of the model
- `parameter_type`: type of the parameters

# Examples
```julia
my_ridge = SemRidge(;α_ridge = 0.02, which_ridge = [:λ₁, :λ₂, :ω₂₃], nparams = 30, implied = my_implied)
```

# Interfaces
Analytic gradients and hessians are available.

# Extended help
## Implementation
Subtype of `SemLossFunction`.
"""
struct SemRidge{P, W1, W2, GT, HT} <: SemLossFunction
    hessianeval::ExactHessian
    α::P
    which::W1
    which_H::W2

    gradient::GT
    hessian::HT
end

############################################################################
### Constructors
############################################################################

function SemRidge(;
    α_ridge,
    which_ridge,
    nparams,
    parameter_type = Float64,
    implied = nothing,
    kwargs...,
)
    if eltype(which_ridge) <: Symbol
        if isnothing(implied)
            throw(
                ArgumentError(
                    "When referring to parameters by label, `implied = ...` has to be specified",
                ),
            )
        else
            par2ind = Dict(par => ind for (ind, par) in enumerate(params(implied)))
            which_ridge = getindex.(Ref(par2ind), which_ridge)
        end
    end
    which_H = [CartesianIndex(x, x) for x in which_ridge]
    return SemRidge(
        ExactHessian(),
        α_ridge,
        which_ridge,
        which_H,
        zeros(parameter_type, nparams),
        zeros(parameter_type, nparams, nparams),
    )
end

############################################################################################
### methods
############################################################################################

objective(ridge::SemRidge, model::AbstractSem, par) =
    @views ridge.α * sum(abs2, par[ridge.which])

function gradient(ridge::SemRidge, model::AbstractSem, par)
    @views ridge.gradient[ridge.which] .= (2 * ridge.α) * par[ridge.which]
    return ridge.gradient
end

function hessian(ridge::SemRidge, model::AbstractSem, par)
    @views @. ridge.hessian[ridge.which_H] .= 2 * ridge.α
    return ridge.hessian
end

############################################################################################
### Recommended methods
############################################################################################

update_observed(loss::SemRidge, observed::SemObserved; kwargs...) = loss
