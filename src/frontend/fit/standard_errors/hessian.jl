"""
    se_hessian(fit::SemFit; method = :finitediff)

Return hessian-based standard errors.

# Arguments
- `method`: how to compute the hessian. Options are
    - `:analytic`: (only if an analytic hessian for the model can be computed)
    - `:finitediff`: for finite difference approximation
"""
function se_hessian(fit::SemFit; method = :finitediff)
    c = H_scaling(fit.model)
    params = solution(fit)
    H = similar(params, (length(params), length(params)))

    if method == :analytic
        evaluate!(nothing, nothing, H, fit.model, params)
    elseif method == :finitediff
        FiniteDiff.finite_difference_hessian!(
            H,
            p -> evaluate!(zero(eltype(H)), nothing, nothing, fit.model, p),
            params,
        )
    elseif method == :optimizer
        error("Standard errors from the optimizer hessian are not implemented yet")
    elseif method == :expected
        error("Standard errors based on the expected hessian are not implemented yet")
    else
        throw(ArgumentError("Unsupported hessian calculation method :$method"))
    end

    H_chol = cholesky!(Symmetric(H))
    H_inv = LinearAlgebra.inv!(H_chol)
    return [sqrt(c * H_inv[i]) for i in diagind(H_inv)]
end

# Addition functions -------------------------------------------------------------
H_scaling(loss::SemML) = 2 / (nsamples(loss) - 1)

function H_scaling(loss::SemWLS)
    @warn "Standard errors for WLS are only correct if a GLS weight matrix (the default) is used."
    return 2 / (nsamples(loss) - 1)
end

H_scaling(loss::SemFIML) = 2 / nsamples(loss)

function H_scaling(model::AbstractSem)
    semterms = SEM.sem_terms(model)
    if length(semterms) > 1
        #@warn "Hessian scaling for multiple loss functions is not implemented yet"
        return 2 / nsamples(model)
    else
        return length(semterms) >= 1 ? H_scaling(loss(semterms[1])) : 1.0
    end
end
