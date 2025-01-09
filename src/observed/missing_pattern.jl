# data associated with the observed variables that all share the same missingness pattern
# variables that have values within that pattern are termed "measured"
# variables that have no measurements are termed "missing"
struct SemObservedMissingPattern{T, S}
    measured_mask::BitVector    # measured vars mask
    miss_mask::BitVector        # missing vars mask
    rows::Vector{Int}           # rows in original data
    data::Matrix{T}             # non-missing submatrix of data (vars × observations)

    measured_mean::Vector{S}    # means of measured vars
    measured_cov::Symmetric{S, Matrix{S}} # covariance of measured vars
end

function SemObservedMissingPattern(
    measured_mask::BitVector,
    rows::AbstractVector{<:Integer},
    data::AbstractMatrix,
)
    T = nonmissingtype(eltype(data))

    pat_data = convert(Matrix{T}, view(data, rows, measured_mask))
    if size(pat_data, 1) > 1
        pat_mean, pat_cov = mean_and_cov(pat_data, 1, corrected = false)
        @assert size(pat_cov) == (size(pat_data, 2), size(pat_data, 2))
    else
        pat_mean = reshape(pat_data[1, :], 1, :)
        # 1x1 covariance matrix since it is not meant to be used
        pat_cov = fill(zero(T), 1, 1)
    end

    return SemObservedMissingPattern{T, eltype(pat_mean)}(
        measured_mask,
        .!measured_mask,
        rows,
        permutedims(pat_data),
        dropdims(pat_mean, dims = 1),
        Symmetric(pat_cov),
    )
end

nobserved_vars(pat::SemObservedMissingPattern) = length(pat.measured_mask)
nsamples(pat::SemObservedMissingPattern) = length(pat.rows)

nmeasured_vars(pat::SemObservedMissingPattern) = length(pat.measured_mean)
nmissed_vars(pat::SemObservedMissingPattern) = nobserved_vars(pat) - nmeasured_vars(pat)
