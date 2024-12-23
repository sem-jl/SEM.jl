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
    return start_simple(model.observed, model.implied, model.loss.functions...; kwargs...)
end

function start_simple(observed, implied, args...; kwargs...)
    return start_simple(implied.ram_matrices; kwargs...)
end

# Ensemble Models --------------------------------------------------------------------------
function start_simple(model::SemEnsemble; kwargs...)
    start_vals = fill(0.0, nparams(model))

    for sem in model.sems
        sem_start_vals = start_simple(sem; kwargs...)
        for (i, val) in enumerate(sem_start_vals)
            if !iszero(val)
                start_vals[i] = val
            end
        end
    end

    return start_vals
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
    start_mean_latent = 0.0,
    start_mean_observed = 0.0,
    kwargs...,
)
    A, S, M = ram_matrices.A, ram_matrices.S, ram_matrices.M
    obs_inds = Set(observed_var_indices(ram_matrices))
    C_indices = CartesianIndices(size(A))

    start_vals = Vector{Float64}(undef, nparams(ram_matrices))
    for i in eachindex(start_vals)
        par = 0.0

        Si_ind = param_occurences(S, i)
        if length(Si_ind) != 0
            # use the first occurence of the parameter to determine starting value
            c_ind = C_indices[Si_ind[1]]
            if c_ind[1] == c_ind[2]
                par = ifelse(
                    c_ind[1] ∈ obs_inds,
                    start_variances_observed,
                    start_variances_latent,
                )
            else
                o1 = c_ind[1] ∈ obs_inds
                o2 = c_ind[2] ∈ obs_inds
                par = ifelse(
                    o1 && o2,
                    start_covariances_observed,
                    ifelse(!o1 && !o2, start_covariances_latent, start_covariances_obs_lat),
                )
            end
        end

        Ai_ind = param_occurences(A, i)
        if length(Ai_ind) != 0
            iszero(par) ||
                @warn "param[$i]=$(params(ram_matrices, i)) is already set to $par"
            c_ind = C_indices[Ai_ind[1]]
            par = ifelse(
                (c_ind[1] ∈ obs_inds) && !(c_ind[2] ∈ obs_inds),
                start_loadings,
                start_regressions,
            )
        end

        if !isnothing(M)
            Mi_inds = param_occurences(M, i)
            if length(Mi_inds) != 0
                iszero(par) ||
                    @warn "param[$i]=$(params(ram_matrices, i)) is already set to $par"
                par = ifelse(Mi_inds[1] ∈ obs_inds, start_mean_observed, start_mean_latent)
            end
        end

        start_vals[i] = par
    end
    return start_vals
end
