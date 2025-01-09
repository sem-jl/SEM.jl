############################################################################################
### Types
############################################################################################

"""
For observed data with missing values.

# Constructor

    SemObservedMissing(;
        data,
        observed_vars = nothing,
        specification = nothing,
        kwargs...)

# Arguments
- `specification`: optional SEM model specification ([`SemSpecification`](@ref))
- `data`: observed data
- `observed_vars::Vector{Symbol}`: column names of the data (if the object passed as data does not have column names, i.e. is not a data frame)

# Extended help
## Interfaces
- `nsamples(::SemObservedMissing)` -> number of samples (data points)
- `nobserved_vars(::SemObservedMissing)` -> number of observed variables

- `samples(::SemObservedMissing)` -> data matrix (contains both measured and missing values)

## Implementation
Subtype of `SemObserved`
"""
struct SemObservedMissing{T <: Real, S <: Real} <: SemObserved
    data::Matrix{Union{T, Missing}}
    observed_vars::Vector{Symbol}
    nsamples::Int
    patterns::Vector{SemObservedMissingPattern{T, S}}

    obs_cov::Matrix{S}
    obs_mean::Vector{S}
end

############################################################################################
### Constructors
############################################################################################

function SemObservedMissing(;
    data,
    observed_vars::Union{AbstractVector, Nothing} = nothing,
    specification::Union{SemSpecification, Nothing} = nothing,
    observed_var_prefix::Union{Symbol, AbstractString} = :obs,
    kwargs...,
)
    data, obs_vars, _ =
        prepare_data(data, observed_vars, specification; observed_var_prefix)
    nsamples, nobs_vars = size(data)

    # detect all different missing patterns with their row indices
    pattern_to_rows = Dict{BitVector, Vector{Int}}()
    for (i, datarow) in zip(axes(data, 1), eachrow(data))
        pattern = BitVector(.!ismissing.(datarow))
        if sum(pattern) > 0 # skip all-missing rows
            pattern_rows = get!(() -> Vector{Int}(), pattern_to_rows, pattern)
            push!(pattern_rows, i)
        end
    end
    # process each pattern and sort from most to least number of observed vars
    patterns = [
        SemObservedMissingPattern(pat, rows, data) for (pat, rows) in pairs(pattern_to_rows)
    ]
    sort!(patterns, by = nmissed_vars)

    em_cov, em_mean = em_mvn(patterns; kwargs...)

    return SemObservedMissing(
        convert(Matrix{Union{nonmissingtype(eltype(data)), Missing}}, data),
        obs_vars,
        nsamples,
        patterns,
        em_cov,
        em_mean,
    )
end

obs_cov(observed::SemObservedMissing) = observed.obs_cov
obs_mean(observed::SemObservedMissing) = observed.obs_mean
