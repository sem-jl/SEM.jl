############################################################################################
### Types
############################################################################################
@doc raw"""
Subtype of `SemImplied` that implements the RAM notation with symbolic precomputation.

# Constructor

    RAMSymbolic(;specification,
        vech = false,
        gradient = true,
        hessian = false,
        approximate_hessian = false,
        kwargs...)

# Arguments
- `specification`: either a `RAMMatrices` or `ParameterTable` object
- `gradient::Bool`: is gradient-based optimization used
- `hessian::Bool`: is hessian-based optimization used
- `approximate_hessian::Bool`: for hessian based optimization: should the hessian be approximated
- `vech::Bool`: should the half-vectorization of Σ be computed (instead of the full matrix)
    (automatically set to true if any of the loss functions is SemWLS)

# Extended help

## Implementation
Subtype of `SemImplied`.

## Interfaces
- `params(::RAMSymbolic) `-> vector of parameter ids
- `nparams(::RAMSymbolic)` -> number of parameters

- `Σ(::RAMSymbolic)` -> model implied covariance matrix
- `μ(::RAMSymbolic)` -> model implied mean vector

Jacobians (only available in gradient! calls)
- `∇Σ(::RAMSymbolic)` -> ``∂vec(Σ)/∂θᵀ``
- `∇μ(::RAMSymbolic)` -> ``∂μ/∂θᵀ``

- `∇Σ_eval!(::RAMSymbolic)` -> function to evaluate `∇Σ` in place,
    i.e. `∇Σ_eval!(∇Σ, θ)`. Normally, you do not want to use this but simply
    query `∇Σ(::RAMSymbolic)`.

Hessians
The computation of hessians is more involved, and uses the "chain rule for
hessian matrices".
Therefore, we desribe it at length in the mathematical appendix of the online documentation,
and the relevant interfaces are omitted here.

## RAM notation
The model implied covariance matrix is computed as
```math
    \Sigma = F(I-A)^{-1}S(I-A)^{-T}F^T
```
and for models with a meanstructure, the model implied means are computed as
```math
    \mu = F(I-A)^{-1}M
```
"""
struct RAMSymbolic{MS, F1, F2, F3, A1, A2, A3, V2, F4, A4, F5, A5} <: SemImpliedSymbolic
    meanstruct::MS
    hessianeval::ExactHessian
    Σ_eval!::F1
    ∇Σ_eval!::F2
    ∇²Σ_eval!::F3
    Σ::A1
    ∇Σ::A2
    ∇²Σ::A3
    ram_matrices::V2
    μ_eval!::F4
    μ::A4
    ∇μ_eval!::F5
    ∇μ::A5

    RAMSymbolic{MS}(args...) where {MS <: MeanStruct} =
        new{MS, map(typeof, args)...}(MS(), ExactHessian(), args...)
end

############################################################################################
### Constructors
############################################################################################

function RAMSymbolic(
    spec::SemSpecification;
    vech::Bool = false,
    gradient::Bool = true,
    hessian::Bool = false,
    approximate_hessian::Bool = false,
    kwargs...,
)
    ram_matrices = convert(RAMMatrices, spec)

    n_par = nparams(ram_matrices)
    par = (Symbolics.@variables θ[1:n_par])[1]

    A = sparse_materialize(Num, ram_matrices.A, par)
    S = sparse_materialize(Num, ram_matrices.S, par)
    M = !isnothing(ram_matrices.M) ? materialize(Num, ram_matrices.M, par) : nothing
    F = ram_matrices.F

    I_A⁻¹ = neumann_series(A)

    # Σ
    Σ_sym = eval_Σ_symbolic(S, I_A⁻¹, F; vech)
    #print(Symbolics.build_function(Σ_sym)[2])
    Σ_eval! = Symbolics.build_function(Σ_sym, par, expression = Val{false})[2]
    Σ = zeros(size(Σ_sym))
    precompile(Σ_eval!, (typeof(Σ), Vector{Float64}))

    # ∇Σ
    if gradient
        ∇Σ_sym = Symbolics.sparsejacobian(vec(Σ_sym), [par...])
        ∇Σ_eval! = Symbolics.build_function(∇Σ_sym, par, expression = Val{false})[2]
        constr = findnz(∇Σ_sym)
        ∇Σ = sparse(constr[1], constr[2], fill(1.0, nnz(∇Σ_sym)), size(∇Σ_sym)...)
        precompile(∇Σ_eval!, (typeof(∇Σ), Vector{Float64}))
    else
        ∇Σ_eval! = nothing
        ∇Σ = nothing
    end

    if hessian && !approximate_hessian
        n_sig = length(Σ_sym)
        ∇²Σ_sym_vec = [Symbolics.sparsehessian(σᵢ, [par...]) for σᵢ in vec(Σ_sym)]

        @variables J[1:n_sig]
        ∇²Σ_sym = zeros(Num, n_par, n_par)
        for i in 1:n_sig
            ∇²Σ_sym += J[i] * ∇²Σ_sym_vec[i]
        end

        ∇²Σ_eval! = Symbolics.build_function(∇²Σ_sym, J, par, expression = Val{false})[2]
        ∇²Σ = zeros(n_par, n_par)
    else
        ∇²Σ_sym = nothing
        ∇²Σ_eval! = nothing
        ∇²Σ = nothing
    end

    # μ
    if !isnothing(ram_matrices.M)
        MS = HasMeanStruct
        μ_sym = eval_μ_symbolic(M, I_A⁻¹, F)
        μ_eval! = Symbolics.build_function(μ_sym, par, expression = Val{false})[2]
        μ = zeros(size(μ_sym))
        if gradient
            ∇μ_sym = Symbolics.jacobian(μ_sym, [par...])
            ∇μ_eval! = Symbolics.build_function(∇μ_sym, par, expression = Val{false})[2]
            ∇μ = zeros(size(F, 1), size(par, 1))
        else
            ∇μ_eval! = nothing
            ∇μ = nothing
        end
    else
        MS = NoMeanStruct
        μ_eval! = nothing
        μ = nothing
        ∇μ_eval! = nothing
        ∇μ = nothing
    end

    return RAMSymbolic{MS}(
        Σ_eval!,
        ∇Σ_eval!,
        ∇²Σ_eval!,
        Σ,
        ∇Σ,
        ∇²Σ,
        ram_matrices,
        μ_eval!,
        μ,
        ∇μ_eval!,
        ∇μ,
    )
end

############################################################################################
### objective, gradient, hessian
############################################################################################

function update!(
    targets::EvaluationTargets,
    implied::RAMSymbolic,
    model::AbstractSemSingle,
    par,
)
    implied.Σ_eval!(implied.Σ, par)
    if MeanStruct(implied) === HasMeanStruct
        implied.μ_eval!(implied.μ, par)
    end

    if is_gradient_required(targets) || is_hessian_required(targets)
        implied.∇Σ_eval!(implied.∇Σ, par)
        if MeanStruct(implied) === HasMeanStruct
            implied.∇μ_eval!(implied.∇μ, par)
        end
    end
end

############################################################################################
### Recommended methods
############################################################################################

function update_observed(implied::RAMSymbolic, observed::SemObserved; kwargs...)
    if nobserved_vars(observed) == size(implied.Σ, 1)
        return implied
    else
        return RAMSymbolic(; observed = observed, kwargs...)
    end
end

############################################################################################
### additional functions
############################################################################################

# expected covariations of observed vars
function eval_Σ_symbolic(S, I_A⁻¹, F; vech::Bool = false)
    Σ = F * I_A⁻¹ * S * permutedims(I_A⁻¹) * permutedims(F)
    Σ = Array(Σ)
    vech && (Σ = SEM.vech(Σ))
    # Σ = Symbolics.simplify.(Σ)
    Threads.@threads for i in eachindex(Σ)
        Σ[i] = Symbolics.simplify(Σ[i])
    end
    return Σ
end

# expected means of observed vars
function eval_μ_symbolic(M, I_A⁻¹, F)
    μ = F * I_A⁻¹ * M
    μ = Array(μ)
    Threads.@threads for i in eachindex(μ)
        μ[i] = Symbolics.simplify(μ[i])
    end
    return μ
end
