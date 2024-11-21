"""

    transpose_linear_indices(n, [m])

Put each linear index of the *n×m* matrix to the position of the
corresponding element in the transposed matrix.

## Example
`
1 4
2 5  =>  1 2 3
3 6      4 5 6
`
"""
transpose_linear_indices(n::Integer, m::Integer = n) =
    repeat(1:n, inner = m) .+ repeat((0:(m-1)) * n, outer = n)

"""
    CommutationMatrix(n::Integer) <: AbstractMatrix{Int}

A *commutation matrix* *C* is a n²×n² matrix of 0s and 1s.
If *vec(A)* is a vectorized form of a n×n matrix *A*,
then ``C * vec(A) = vec(Aᵀ)``.
"""
struct CommutationMatrix <: AbstractMatrix{Int}
    n::Int
    n²::Int
    transpose_inds::Vector{Int} # maps the linear indices of n×n matrix *B* to the indices of matrix *B'*

    CommutationMatrix(n::Integer) = new(n, n^2, transpose_linear_indices(n))
end

Base.size(A::CommutationMatrix) = (A.n², A.n²)
Base.size(A::CommutationMatrix, dim::Integer) =
    1 <= dim <= 2 ? A.n² : throw(ArgumentError("invalid matrix dimension $dim"))
Base.length(A::CommutationMatrix) = A.n²^2
Base.getindex(A::CommutationMatrix, i::Int, j::Int) = j == A.transpose_inds[i] ? 1 : 0

function Base.:(*)(A::CommutationMatrix, B::AbstractVector)
    size(A, 2) == size(B, 1) || throw(
        DimensionMismatch("A has $(size(A, 2)) columns, but B has $(size(B, 1)) elements"),
    )
    return B[A.transpose_inds]
end

function Base.:(*)(A::CommutationMatrix, B::AbstractMatrix)
    size(A, 2) == size(B, 1) || throw(
        DimensionMismatch("A has $(size(A, 2)) columns, but B has $(size(B, 1)) rows"),
    )
    return B[A.transpose_inds, :]
end

function Base.:(*)(A::CommutationMatrix, B::SparseMatrixCSC)
    size(A, 2) == size(B, 1) || throw(
        DimensionMismatch("A has $(size(A, 2)) columns, but B has $(size(B, 1)) rows"),
    )
    return SparseMatrixCSC(
        size(B, 1),
        size(B, 2),
        copy(B.colptr),
        A.transpose_inds[B.rowval],
        copy(B.nzval),
    )
end

function LinearAlgebra.lmul!(A::CommutationMatrix, B::SparseMatrixCSC)
    size(A, 2) == size(B, 1) || throw(
        DimensionMismatch("A has $(size(A, 2)) columns, but B has $(size(B, 1)) rows"),
    )

    @inbounds for (i, rowind) in enumerate(B.rowval)
        B.rowval[i] = A.transpose_inds[rowind]
    end
    return B
end
