# fill A, S, and M matrices with the parameter values according to the parameters map
function fill_A_S_M!(
    A::AbstractMatrix,
    S::AbstractMatrix,
    M::Union{AbstractVector, Nothing},
    A_indices::AbstractArrayParamsMap,
    S_indices::AbstractArrayParamsMap,
    M_indices::Union{AbstractArrayParamsMap, Nothing},
    parameters::AbstractVector,
)
    @inbounds for (iA, iS, par) in zip(A_indices, S_indices, parameters)
        for index_A in iA
            A[index_A] = par
        end

        for index_S in iS
            S[index_S] = par
        end
    end

    if !isnothing(M)
        @inbounds for (iM, par) in zip(M_indices, parameters)
            for index_M in iM
                M[index_M] = par
            end
        end
    end
end

# build the map from the index of the parameter to the linear indices
# of this parameter occurences in M
# returns ArrayParamsMap object
function array_parameters_map(parameters::AbstractVector, M::AbstractArray)
    params_index = Dict(param => i for (i, param) in enumerate(parameters))
    T = Base.eltype(eachindex(M))
    res = [Vector{T}() for _ in eachindex(parameters)]
    for (i, val) in enumerate(M)
        par_ind = get(params_index, val, nothing)
        if !isnothing(par_ind)
            push!(res[par_ind], i)
        end
    end
    return res
end

function eachindex_lower(M; linear_indices = false, kwargs...)
    indices = CartesianIndices(M)
    indices = filter(x -> (x[1] >= x[2]), indices)

    if linear_indices
        indices = cartesian2linear(indices, M)
    end

    return indices
end

function cartesian2linear(ind_cart, dims)
    ind_lin = LinearIndices(dims)[ind_cart]
    return ind_lin
end

function linear2cartesian(ind_lin, dims)
    ind_cart = CartesianIndices(dims)[ind_lin]
    return ind_cart
end

function set_constants!(M, M_pre)
    for index in eachindex(M)
        δ = tryparse(Float64, string(M[index]))

        if !iszero(M[index]) & (δ !== nothing)
            M_pre[index] = δ
        end
    end
end

function check_constants(M)
    for index in eachindex(M)
        δ = tryparse(Float64, string(M[index]))

        if !iszero(M[index]) & (δ !== nothing)
            return true
        end
    end

    return false
end

function get_matrix_derivative(M_indices, parameters, n_long)
    ∇M = [
        sparsevec(M_indices[i], ones(length(M_indices[i])), n_long) for
        i in 1:length(parameters)
    ]

    ∇M = reduce(hcat, ∇M)

    return ∇M
end

# fill M with parameters
function fill_matrix!(
    M::AbstractMatrix,
    M_indices::AbstractArrayParamsMap,
    parameters::AbstractVector,
)
    for (iM, par) in zip(M_indices, parameters)
        for index_M in iM
            M[index_M] = par
        end
    end
    return M
end

# range of parameters that are referenced in the matrix
function param_range(mtx_indices::AbstractArrayParamsMap)
    first_i = findfirst(!isempty, mtx_indices)
    last_i = findlast(!isempty, mtx_indices)

    if !isnothing(first_i) && !isnothing(last_i)
        for i in first_i:last_i
            if isempty(mtx_indices[i])
                # TODO show which parameter is missing in which matrix
                throw(
                    ErrorException(
                        "Your parameter vector is not partitioned into directed and undirected effects",
                    ),
                )
            end
        end
    end

    return first_i:last_i
end
