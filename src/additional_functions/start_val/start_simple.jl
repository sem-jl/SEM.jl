"""
    start_simple(
        model;
        start_loadings = 0.5,
        start_regressions = 0.0,
        start_variances_observed = 1,
        start_variances_latent = 0.05,
        start_covariances_observed = 0.0,
        start_covariances_latent = 0.0,
        start_covariances_obs_lat = 0.0,
        start_means = 0.0,
        kwargs...)

Return a vector of simple starting values.
"""
function start_simple end

# Single Models ----------------------------------------------------------------------------
function start_simple(model::AbstractSemSingle; kwargs...)
    return start_simple(
        model.observed,
        model.imply,
        model.optimizer,
        model.loss.functions...;
        kwargs...,
    )
end

function start_simple(observed, imply, optimizer, args...; kwargs...)
    return start_simple(imply.ram_matrices; kwargs...)
end

# Ensemble Models --------------------------------------------------------------------------
function start_simple(model::SemEnsemble; kwargs...)
    start_vals = []

    for sem in model.sems
        push!(start_vals, start_simple(sem; kwargs...))
    end

    has_start_val = [.!iszero.(start_val) for start_val in start_vals]

    start_val = similar(start_vals[1])
    start_val .= 0.0

    for (j, indices) in enumerate(has_start_val)
        start_val[indices] .= start_vals[j][indices]
    end

    return start_val
end

function start_simple(
    ram_matrices::RAMMatrices;
    start_loadings = 0.5,
    start_regressions = 0.0,
    start_variances_observed = 1,
    start_variances_latent = 0.05,
    start_covariances_observed = 0.0,
    start_covariances_latent = 0.0,
    start_covariances_obs_lat = 0.0,
    start_means = 0.0,
    kwargs...,
)
    A_ind, S_ind, F_ind, M_ind, n_par = ram_matrices.A_ind,
    ram_matrices.S_ind,
    ram_matrices.F_ind,
    ram_matrices.M_ind,
    nparams(ram_matrices)

    start_val = zeros(n_par)
    n_obs = nobserved_vars(ram_matrices)
    n_var = nvars(ram_matrices)

    C_indices = CartesianIndices((n_var, n_var))

    for i in 1:n_par
        if length(S_ind[i]) != 0
            # use the first occurence of the parameter to determine starting value
            c_ind = C_indices[S_ind[i][1]]
            if c_ind[1] == c_ind[2]
                if c_ind[1] ∈ F_ind
                    start_val[i] = start_variances_observed
                else
                    start_val[i] = start_variances_latent
                end
            else
                o1 = c_ind[1] ∈ F_ind
                o2 = c_ind[2] ∈ F_ind
                if o1 & o2
                    start_val[i] = start_covariances_observed
                elseif !o1 & !o2
                    start_val[i] = start_covariances_latent
                else
                    start_val[i] = start_covariances_obs_lat
                end
            end
        elseif length(A_ind[i]) != 0
            c_ind = C_indices[A_ind[i][1]]
            if (c_ind[1] ∈ F_ind) & !(c_ind[2] ∈ F_ind)
                start_val[i] = start_loadings
            else
                start_val[i] = start_regressions
            end
        elseif !isnothing(M_ind) && (length(M_ind[i]) != 0)
            start_val[i] = start_means
        end
    end
    return start_val
end
