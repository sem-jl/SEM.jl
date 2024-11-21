############################################################################################
### Types
############################################################################################

# Type to store Expectation Maximization result --------------------------------------------
mutable struct EmMVNModel{A, b, B}
    Σ::A
    μ::b
    fitted::B
end

"""
For observed data with missing values.

# Constructor

    SemObservedMissing(;
        specification,
        data,
        obs_colnames = nothing,
        kwargs...)

# Arguments
- `specification`: either a `RAMMatrices` or `ParameterTable` object (1)
- `data`: observed data
- `obs_colnames::Vector{Symbol}`: column names of the data (if the object passed as data does not have column names, i.e. is not a data frame)

# Extended help
## Interfaces
- `nsamples(::SemObservedMissing)` -> number of observed data points
- `nobserved_vars(::SemObservedMissing)` -> number of manifest variables

- `samples(::SemObservedMissing)` -> observed data
- `data_rowwise(::SemObservedMissing)` -> observed data as vector per observation, with missing values deleted

- `patterns(::SemObservedMissing)` -> indices of non-missing variables per missing patterns
- `patterns_not(::SemObservedMissing)` -> indices of missing variables per missing pattern
- `pattern_rows(::SemObservedMissing)` -> row indices of observed data points that belong to each pattern
- `pattern_nsamples(::SemObservedMissing)` -> number of data points per pattern
- `pattern_nobs_vars(::SemObservedMissing)` -> number of non-missing observed variables per pattern
- `obs_mean(::SemObservedMissing)` -> observed mean per pattern
- `obs_cov(::SemObservedMissing)` -> observed covariance per pattern
- `em_model(::SemObservedMissing)` -> `EmMVNModel` that contains the covariance matrix and mean vector found via optimization maximization

## Implementation
Subtype of `SemObserved`

## Remarks
(1) the `specification` argument can also be `nothing`, but this turns of checking whether
the observed data/covariance columns are in the correct order! As a result, you should only
use this if you are sure your observed data is in the right format.

## Additional keyword arguments:
- `spec_colnames::Vector{Symbol} = nothing`: overwrites column names of the specification object
"""
mutable struct SemObservedMissing{
    A <: AbstractArray,
    D <: Number,
    O <: Number,
    P <: Vector,
    P2 <: Vector,
    R <: Vector,
    PD <: AbstractArray,
    PO <: AbstractArray,
    PVO <: AbstractArray,
    A2 <: AbstractArray,
    A3 <: AbstractArray,
    S <: EmMVNModel,
} <: SemObserved
    data::A
    nobs_vars::D
    nsamples::O
    patterns::P # missing patterns
    patterns_not::P2
    pattern_rows::R # coresponding rows in data_rowwise
    data_rowwise::PD # list of data
    pattern_nsamples::PO # observed rows per pattern
    pattern_nobs_vars::PVO # number of non-missing variables per pattern
    obs_mean::A2
    obs_cov::A3
    em_model::S
end

############################################################################################
### Constructors
############################################################################################

function SemObservedMissing(;
    specification::Union{SemSpecification, Nothing},
    data,
    obs_colnames = nothing,
    spec_colnames = nothing,
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

    # remove persons with only missings
    keep = Vector{Int64}()
    for i in 1:size(data, 1)
        if any(.!ismissing.(data[i, :]))
            push!(keep, i)
        end
    end
    data = data[keep, :]

    nsamples, nobs_vars = size(data)

    # compute and store the different missing patterns with their rowindices
    missings = ismissing.(data)
    patterns = [missings[i, :] for i in 1:size(missings, 1)]

    patterns_cart = findall.(!, patterns)
    data_rowwise = [data[i, patterns_cart[i]] for i in 1:nsamples]
    data_rowwise = convert.(Array{Float64}, data_rowwise)

    remember = Vector{BitArray{1}}()
    rows = [Vector{Int64}(undef, 0) for i in 1:size(patterns, 1)]
    for i in 1:size(patterns, 1)
        unknown = true
        for j in 1:size(remember, 1)
            if patterns[i] == remember[j]
                push!(rows[j], i)
                unknown = false
            end
        end
        if unknown
            push!(remember, patterns[i])
            push!(rows[size(remember, 1)], i)
        end
    end
    rows = rows[1:length(remember)]
    n_patterns = size(rows, 1)

    # sort by number of missings
    sort_n_miss = sortperm(sum.(remember))
    remember = remember[sort_n_miss]
    remember_cart = findall.(!, remember)
    remember_cart_not = findall.(remember)
    rows = rows[sort_n_miss]

    pattern_nsamples = size.(rows, 1)
    pattern_nobs_vars = length.(remember_cart)

    cov_mean = [cov_and_mean(data_rowwise[rows]) for rows in rows]
    obs_cov = [cov_mean[1] for cov_mean in cov_mean]
    obs_mean = [cov_mean[2] for cov_mean in cov_mean]

    em_model = EmMVNModel(zeros(nobs_vars, nobs_vars), zeros(nobs_vars), false)

    return SemObservedMissing(
        data,
        nobs_vars,
        nsamples,
        remember_cart,
        remember_cart_not,
        rows,
        data_rowwise,
        pattern_nsamples,
        pattern_nobs_vars,
        obs_mean,
        obs_cov,
        em_model,
    )
end

############################################################################################
### Recommended methods
############################################################################################

nsamples(observed::SemObservedMissing) = observed.nsamples
nobserved_vars(observed::SemObservedMissing) = observed.nobs_vars

############################################################################################
### Additional methods
############################################################################################

patterns(observed::SemObservedMissing) = observed.patterns
patterns_not(observed::SemObservedMissing) = observed.patterns_not
pattern_rows(observed::SemObservedMissing) = observed.pattern_rows
data_rowwise(observed::SemObservedMissing) = observed.data_rowwise
pattern_nsamples(observed::SemObservedMissing) = observed.pattern_nsamples
pattern_nobs_vars(observed::SemObservedMissing) = observed.pattern_nobs_vars
obs_mean(observed::SemObservedMissing) = observed.obs_mean
obs_cov(observed::SemObservedMissing) = observed.obs_cov
em_model(observed::SemObservedMissing) = observed.em_model
