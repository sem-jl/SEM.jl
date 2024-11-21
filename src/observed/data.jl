"""
For observed data without missings.

# Constructor

    SemObservedData(;
        specification,
        data,
        meanstructure = false,
        obs_colnames = nothing,
        kwargs...)

# Arguments
- `specification`: either a `RAMMatrices` or `ParameterTable` object (1)
- `data`: observed data
- `meanstructure::Bool`: does the model have a meanstructure?
- `obs_colnames::Vector{Symbol}`: column names of the data (if the object passed as data does not have column names, i.e. is not a data frame)

# Extended help
## Interfaces
- `nsamples(::SemObservedData)` -> number of observed data points
- `nobserved_vars(::SemObservedData)` -> number of observed (manifested) variables

- `samples(::SemObservedData)` -> observed data
- `obs_cov(::SemObservedData)` -> observed.obs_cov
- `obs_mean(::SemObservedData)` -> observed.obs_mean

## Implementation
Subtype of `SemObserved`

## Remarks
(1) the `specification` argument can also be `nothing`, but this turns of checking whether
the observed data/covariance columns are in the correct order! As a result, you should only
use this if you are sure your observed data is in the right format.

## Additional keyword arguments:
- `spec_colnames::Vector{Symbol} = nothing`: overwrites column names of the specification object
- `compute_covariance::Bool ) = true`: should the covariance of `data` be computed and stored?
"""
struct SemObservedData{A, B, C} <: SemObserved
    data::A
    obs_cov::B
    obs_mean::C
    nobs_vars::Int
    nsamples::Int
end

# error checks
function check_arguments_SemObservedData(kwargs...)
    # data is a data frame,

end

function SemObservedData(;
    specification::Union{SemSpecification, Nothing},
    data,
    obs_colnames = nothing,
    spec_colnames = nothing,
    meanstructure = false,
    compute_covariance = true,
    kwargs...,
)
    if isnothing(spec_colnames) && !isnothing(specification)
        spec_colnames = observed_vars(specification)
    end

    if !isnothing(spec_colnames)
        if isnothing(obs_colnames)
            try
                data = data[:, spec_colnames]
            catch
                throw(
                    ArgumentError(
                        "Your `data` can not be indexed by symbols. " *
                        "Maybe you forgot to provide column names via the `obs_colnames = ...` argument.",
                    ),
                )
            end
        else
            if data isa DataFrame
                throw(
                    ArgumentError(
                        "You passed your data as a `DataFrame`, but also specified `obs_colnames`. " *
                        "Please make sure the column names of your data frame indicate the correct variables " *
                        "or pass your data in a different format.",
                    ),
                )
            end

            if !(eltype(obs_colnames) <: Symbol)
                throw(ArgumentError("please specify `obs_colnames` as a vector of Symbols"))
            end

            data = reorder_data(data, spec_colnames, obs_colnames)
        end
    end

    if data isa DataFrame
        data = Matrix(data)
    end

    return SemObservedData(
        data,
        compute_covariance ? Statistics.cov(data) : nothing,
        meanstructure ? vec(Statistics.mean(data, dims = 1)) : nothing,
        size(data, 2),
        size(data, 1),
    )
end

############################################################################################
### Recommended methods
############################################################################################

nsamples(observed::SemObservedData) = observed.nsamples
nobserved_vars(observed::SemObservedData) = observed.nobs_vars

############################################################################################
### additional methods
############################################################################################

obs_cov(observed::SemObservedData) = observed.obs_cov
obs_mean(observed::SemObservedData) = observed.obs_mean

############################################################################################
### Additional functions
############################################################################################

# reorder data -----------------------------------------------------------------------------
function reorder_data(data::AbstractArray, spec_colnames, obs_colnames)
    if spec_colnames == obs_colnames
        return data
    else
        obs_positions = Dict(col => i for (i, col) in enumerate(obs_colnames))
        new_positions = [obs_positions[col] for col in spec_colnames]
        return data[:, new_positions]
    end
end
