"""
Type alias for [`SemObservedData`](@ref) that has mean and covariance, but no actual data.
"""
const SemObservedCovariance{S} = SemObservedData{Nothing, S}

function SemObservedCovariance(;
    obs_cov::AbstractMatrix,
    obs_mean::Union{AbstractVector, Nothing} = nothing,
    observed_vars::Union{AbstractVector, Nothing} = nothing,
    specification::Union{SemSpecification, Nothing} = nothing,
    nsamples::Integer,
    observed_var_prefix::Union{Symbol, AbstractString} = :obs,
    kwargs...,
)
    nvars = size(obs_cov, 1)
    size(obs_cov, 2) == nvars || throw(
        DimensionMismatch(
            "The covariance matrix should be square, $(size(obs_cov)) was found.",
        ),
    )
    S = eltype(obs_cov)

    if isnothing(obs_mean)
        obs_mean = zeros(S, nvars)
    else
        length(obs_mean) == nvars || throw(
            DimensionMismatch(
                "The length of the mean vector $(length(obs_mean)) does not match the size of the covariance matrix $(size(obs_cov))",
            ),
        )
        S = promote_type(S, eltype(obs_mean))
    end

    obs_cov = convert(Matrix{S}, obs_cov)
    obs_mean = convert(Vector{S}, obs_mean)

    if !isnothing(observed_vars)
        length(observed_vars) == nvars || throw(
            DimensionMismatch(
                "The length of the observed_vars $(length(observed_vars)) does not match the size of the covariance matrix $(size(obs_cov))",
            ),
        )
    end

    _, obs_vars, obs_vars_perm =
        prepare_data(nothing, observed_vars, specification, nvars; observed_var_prefix)

    # reorder to match the specification
    if !isnothing(obs_vars_perm)
        obs_cov = obs_cov[obs_vars_perm, obs_vars_perm]
        obs_mean = obs_mean[obs_vars_perm]
    end

    return SemObservedData(nothing, obs_vars, obs_cov, obs_mean, nsamples)
end
