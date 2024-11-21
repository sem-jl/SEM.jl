############################################################################################
### Constants
############################################################################################

struct RAMConstant
    matrix::Symbol
    index::Union{Int, CartesianIndex{2}}
    value::Any
end

function Base.:(==)(c1::RAMConstant, c2::RAMConstant)
    res = ((c1.matrix == c2.matrix) && (c1.index == c2.index) && (c1.value == c2.value))
    return res
end

function append_RAMConstants!(
    constants::AbstractVector{RAMConstant},
    mtx_name::Symbol,
    mtx::AbstractArray;
    skip_zeros::Bool = true,
)
    for (index, val) in pairs(mtx)
        if isa(val, Number) && !(skip_zeros && iszero(val))
            push!(constants, RAMConstant(mtx_name, index, val))
        end
    end
    return constants
end

function set_RAMConstant!(A, S, M, rc::RAMConstant)
    if rc.matrix == :A
        A[rc.index] = rc.value
    elseif rc.matrix == :S
        S[rc.index] = rc.value
        S[rc.index[2], rc.index[1]] = rc.value # symmetric
    elseif rc.matrix == :M
        M[rc.index] = rc.value
    end
end

function set_RAMConstants!(A, S, M, rc_vec::Vector{RAMConstant})
    for rc in rc_vec
        set_RAMConstant!(A, S, M, rc)
    end
end

############################################################################################
### Type
############################################################################################

# map from parameter index to linear indices of matrix/vector positions where it occurs
AbstractArrayParamsMap = AbstractVector{<:AbstractVector{<:Integer}}
ArrayParamsMap = Vector{Vector{Int}}

struct RAMMatrices <: SemSpecification
    A_ind::ArrayParamsMap
    S_ind::ArrayParamsMap
    F_ind::Vector{Int}
    M_ind::Union{ArrayParamsMap, Nothing}
    params::Vector{Symbol}
    colnames::Union{Vector{Symbol}, Nothing}
    constants::Vector{RAMConstant}
    size_F::Tuple{Int, Int}
end

nparams(ram::RAMMatrices) = length(ram.A_ind)

nvars(ram::RAMMatrices) = ram.size_F[2]
nobserved_vars(ram::RAMMatrices) = ram.size_F[1]
nlatent_vars(ram::RAMMatrices) = nvars(ram) - nobserved_vars(ram)

vars(ram::RAMMatrices) = ram.colnames

function observed_vars(ram::RAMMatrices)
    if isnothing(ram.colnames)
        @warn "Your RAMMatrices do not contain column names. Please make sure the order of variables in your data is correct!"
        return nothing
    else
        return view(ram.colnames, ram.F_ind)
    end
end

function latent_vars(ram::RAMMatrices)
    if isnothing(ram.colnames)
        @warn "Your RAMMatrices do not contain column names. Please make sure the order of variables in your data is correct!"
        return nothing
    else
        return view(ram.colnames, setdiff(eachindex(ram.colnames), ram.F_ind))
    end
end

############################################################################################
### Constructor
############################################################################################

function RAMMatrices(;
    A::AbstractMatrix,
    S::AbstractMatrix,
    F::AbstractMatrix,
    M::Union{AbstractVector, Nothing} = nothing,
    params::AbstractVector{Symbol},
    colnames::Union{AbstractVector{Symbol}, Nothing} = nothing,
)
    ncols = size(A, 2)
    isnothing(colnames) || check_vars(colnames, ncols)

    size(A, 1) == size(A, 2) || throw(DimensionMismatch("A must be a square matrix"))
    size(S, 1) == size(S, 2) || throw(DimensionMismatch("S must be a square matrix"))
    size(A, 2) == ncols || throw(
        DimensionMismatch(
            "A should have as many rows and columns as colnames length ($ncols), $(size(A)) found",
        ),
    )
    size(S, 2) == ncols || throw(
        DimensionMismatch(
            "S should have as many rows and columns as colnames length ($ncols), $(size(S)) found",
        ),
    )
    size(F, 2) == ncols || throw(
        DimensionMismatch(
            "F should have as many columns as colnames length ($ncols), $(size(F, 2)) found",
        ),
    )
    if !isnothing(M)
        length(M) == ncols || throw(
            DimensionMismatch(
                "M should have as many elements as colnames length ($ncols), $(length(M)) found",
            ),
        )
    end

    check_params(params, nothing)

    A_indices = array_params_map(params, A)
    S_indices = array_params_map(params, S)
    M_indices = !isnothing(M) ? array_params_map(params, M) : nothing
    F_indices = [i for (i, col) in zip(axes(F, 2), eachcol(F)) if any(isone, col)]
    constants = Vector{RAMConstant}()
    append_RAMConstants!(constants, :A, A)
    append_RAMConstants!(constants, :S, S)
    isnothing(M) || append_RAMConstants!(constants, :M, M)
    return RAMMatrices(
        A_indices,
        S_indices,
        F_indices,
        M_indices,
        params,
        colnames,
        constants,
        size(F),
    )
end

############################################################################################
### get RAMMatrices from parameter table
############################################################################################

function RAMMatrices(
    partable::ParameterTable;
    params::Union{AbstractVector{Symbol}, Nothing} = nothing,
)
    params = copy(isnothing(params) ? SEM.params(partable) : params)
    check_params(params, partable.columns[:param])
    params_index = Dict(param => i for (i, param) in enumerate(params))

    n_observed = length(partable.observed_vars)
    n_latent = length(partable.latent_vars)
    n_node = n_observed + n_latent

    # F indices
    F_ind =
        length(partable.sorted_vars) != 0 ?
        findall(∈(Set(partable.observed_vars)), partable.sorted_vars) : 1:n_observed

    # indices of the colnames
    colnames =
        length(partable.sorted_vars) != 0 ? copy(partable.sorted_vars) :
        [
            partable.observed_vars
            partable.latent_vars
        ]
    col_indices = Dict(col => i for (i, col) in enumerate(colnames))

    # fill Matrices
    # known_labels = Dict{Symbol, Int64}()

    A_ind = [Vector{Int64}() for _ in 1:length(params)]
    S_ind = [Vector{Int64}() for _ in 1:length(params)]

    # is there a meanstructure?
    M_ind =
        any(==(Symbol("1")), partable.columns[:from]) ?
        [Vector{Int64}() for _ in 1:length(params)] : nothing

    # handle constants
    constants = Vector{RAMConstant}()

    for r in partable
        row_ind = col_indices[r.to]
        col_ind = r.from != Symbol("1") ? col_indices[r.from] : nothing

        if !r.free
            if (r.relation == :→) && (r.from == Symbol("1"))
                push!(constants, RAMConstant(:M, row_ind, r.value_fixed))
            elseif r.relation == :→
                push!(
                    constants,
                    RAMConstant(:A, CartesianIndex(row_ind, col_ind), r.value_fixed),
                )
            elseif r.relation == :↔
                push!(
                    constants,
                    RAMConstant(:S, CartesianIndex(row_ind, col_ind), r.value_fixed),
                )
            else
                error("Unsupported parameter type: $(r.relation)")
            end
        else
            par_ind = params_index[r.param]
            if (r.relation == :→) && (r.from == Symbol("1"))
                push!(M_ind[par_ind], row_ind)
            elseif r.relation == :→
                push!(A_ind[par_ind], row_ind + (col_ind - 1) * n_node)
            elseif r.relation == :↔
                push!(S_ind[par_ind], row_ind + (col_ind - 1) * n_node)
                if row_ind != col_ind
                    push!(S_ind[par_ind], col_ind + (row_ind - 1) * n_node)
                end
            else
                error("Unsupported parameter type: $(r.relation)")
            end
        end
    end

    return RAMMatrices(
        A_ind,
        S_ind,
        F_ind,
        M_ind,
        params,
        colnames,
        constants,
        (n_observed, n_node),
    )
end

Base.convert(
    ::Type{RAMMatrices},
    partable::ParameterTable;
    params::Union{AbstractVector{Symbol}, Nothing} = nothing,
) = RAMMatrices(partable; params)

############################################################################################
### get parameter table from RAMMatrices
############################################################################################

function ParameterTable(
    ram_matrices::RAMMatrices;
    params::Union{AbstractVector{Symbol}, Nothing} = nothing,
    observed_var_prefix::Symbol = :obs,
    latent_var_prefix::Symbol = :var,
)
    # defer parameter checks until we know which ones are used
    if !isnothing(ram_matrices.colnames)
        colnames = ram_matrices.colnames
        observed_vars = colnames[ram_matrices.F_ind]
        latent_vars = colnames[setdiff(eachindex(colnames), ram_matrices.F_ind)]
    else
        observed_vars =
            [Symbol("$(observed_var_prefix)_$i") for i in 1:nobserved_vars(ram_matrices)]
        latent_vars =
            [Symbol("$(latent_var_prefix)_$i") for i in 1:nlatent_vars(ram_matrices)]
        colnames = vcat(observed_vars, latent_vars)
    end

    # construct an empty table
    partable = ParameterTable(
        observed_vars = observed_vars,
        latent_vars = latent_vars,
        params = isnothing(params) ? SEM.params(ram_matrices) : params,
    )

    # constants
    for c in ram_matrices.constants
        push!(partable, partable_row(c, colnames))
    end

    # parameters
    for (i, par) in enumerate(ram_matrices.params)
        append_partable_rows!(
            partable,
            colnames,
            par,
            i,
            ram_matrices.A_ind,
            ram_matrices.S_ind,
            ram_matrices.M_ind,
            ram_matrices.size_F[2],
        )
    end
    check_params(SEM.params(partable), partable.columns[:param])

    return partable
end

Base.convert(
    ::Type{<:ParameterTable},
    ram::RAMMatrices;
    params::Union{AbstractVector{Symbol}, Nothing} = nothing,
) = ParameterTable(ram; params)

############################################################################################
### Pretty Printing
############################################################################################

function Base.show(io::IO, ram_matrices::RAMMatrices)
    print_type_name(io, ram_matrices)
    print_field_types(io, ram_matrices)
end

############################################################################################
### Additional Functions
############################################################################################

# return the `from □ to` variables relation symbol (□) given the name of the source RAM matrix
function matrix_to_relation(matrix::Symbol)
    if matrix == :A
        return :→
    elseif matrix == :S
        return :↔
    elseif matrix == :M
        return :→
    else
        throw(
            ArgumentError(
                "Unsupported matrix $matrix, supported matrices are :A, :S and :M",
            ),
        )
    end
end

partable_row(c::RAMConstant, varnames::AbstractVector{Symbol}) = (
    from = varnames[c.index[2]],
    relation = matrix_to_relation(c.matrix),
    to = varnames[c.index[1]],
    free = false,
    value_fixed = c.value,
    start = 0.0,
    estimate = 0.0,
    param = :const,
)

function partable_row(
    par::Symbol,
    varnames::AbstractVector{Symbol},
    index::Integer,
    matrix::Symbol,
    n_nod::Integer,
)

    # variable names
    if matrix == :M
        from = Symbol("1")
        to = varnames[index]
    else
        cart_index = linear2cartesian(index, (n_nod, n_nod))

        from = varnames[cart_index[2]]
        to = varnames[cart_index[1]]
    end

    return (
        from = from,
        relation = matrix_to_relation(matrix),
        to = to,
        free = true,
        value_fixed = 0.0,
        start = 0.0,
        estimate = 0.0,
        param = par,
    )
end

function append_partable_rows!(
    partable::ParameterTable,
    varnames::AbstractVector{Symbol},
    par::Symbol,
    par_index::Integer,
    A_ind,
    S_ind,
    M_ind,
    n_nod::Integer,
)
    for ind in A_ind[par_index]
        push!(partable, partable_row(par, varnames, ind, :A, n_nod))
    end

    visited_S_indices = Set{Int}()
    for ind in S_ind[par_index]
        if ind ∉ visited_S_indices
            push!(partable, partable_row(par, varnames, ind, :S, n_nod))
            # mark index and its symmetric as visited
            push!(visited_S_indices, ind)
            cart_index = linear2cartesian(ind, (n_nod, n_nod))
            push!(
                visited_S_indices,
                cartesian2linear(
                    CartesianIndex(cart_index[2], cart_index[1]),
                    (n_nod, n_nod),
                ),
            )
        end
    end

    if !isnothing(M_ind)
        for ind in M_ind[par_index]
            push!(partable, partable_row(par, varnames, ind, :M, n_nod))
        end
    end

    return nothing
end

function Base.:(==)(mat1::RAMMatrices, mat2::RAMMatrices)
    res = (
        (mat1.A_ind == mat2.A_ind) &&
        (mat1.S_ind == mat2.S_ind) &&
        (mat1.F_ind == mat2.F_ind) &&
        (mat1.M_ind == mat2.M_ind) &&
        (mat1.params == mat2.params) &&
        (mat1.colnames == mat2.colnames) &&
        (mat1.size_F == mat2.size_F) &&
        (mat1.constants == mat2.constants)
    )
    return res
end
