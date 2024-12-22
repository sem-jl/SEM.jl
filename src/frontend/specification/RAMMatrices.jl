
############################################################################################
### Type
############################################################################################

struct RAMMatrices <: SemSpecification
    A::ParamsMatrix{Float64}
    S::ParamsMatrix{Float64}
    F::SparseMatrixCSC{Float64}
    M::Union{ParamsVector{Float64}, Nothing}
    params::Vector{Symbol}
    vars::Union{Vector{Symbol}, Nothing}    # better call it "variables": it's a mixture of observed and latent (and it gets confusing with get_vars())
end

nparams(ram::RAMMatrices) = nparams(ram.A)
nvars(ram::RAMMatrices) = size(ram.F, 2)
nobserved_vars(ram::RAMMatrices) = size(ram.F, 1)
nlatent_vars(ram::RAMMatrices) = nvars(ram) - nobserved_vars(ram)

vars(ram::RAMMatrices) = ram.vars

isobserved_var(ram::RAMMatrices, i::Integer) = ram.F.colptr[i+1] > ram.F.colptr[i]
islatent_var(ram::RAMMatrices, i::Integer) = ram.F.colptr[i+1] == ram.F.colptr[i]

# indices of observed variables, for order=:rows (default), the order is as they appear in ram.F rows
# if order=:columns, the order is as they appear in the comined variables list (ram.F columns)
observed_var_indices(ram::RAMMatrices; order::Symbol = :rows) =
    nzcols_eachrow_to_col(ram.F; order)

latent_var_indices(ram::RAMMatrices) = [i for i in axes(ram.F, 2) if islatent_var(ram, i)]

# observed variables, if order=:rows, the order is as they appear in ram.F rows
# if order=:columns, the order is as they appear in the comined variables list (ram.F columns)
function observed_vars(ram::RAMMatrices; order::Symbol = :rows)
    order ∈ [:rows, :columns] ||
        throw(ArgumentError("order kwarg should be :rows or :cols"))
    if isnothing(ram.vars)
        @warn "Your RAMMatrices do not contain variable names. Please make sure the order of variables in your data is correct!"
        return nothing
    else
        return nzcols_eachrow_to_col(Base.Fix1(getindex, vars(ram)), ram.F; order = order)
    end
end

function latent_vars(ram::RAMMatrices)
    if isnothing(ram.vars)
        @warn "Your RAMMatrices do not contain variable names. Please make sure the order of variables in your data is correct!"
        return nothing
    else
        return [col for (i, col) in enumerate(ram.vars) if islatent_var(ram, i)]
    end
end

function variance_params(ram::RAMMatrices)
    S_diaginds = Set(diagind(ram.S))
    varparams = Vector{Symbol}()
    for (i, param) in enumerate(ram.params)
        if any(∈(S_diaginds), param_occurences(ram.S, i))
            push!(varparams, param)
        end
    end
    return unique!(varparams)
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
    vars::Union{AbstractVector{Symbol}, Nothing} = nothing,
)
    ncols = size(A, 2)
    isnothing(vars) || check_vars(vars, ncols)

    size(A, 1) == size(A, 2) || throw(DimensionMismatch("A must be a square matrix"))
    size(S, 1) == size(S, 2) || throw(DimensionMismatch("S must be a square matrix"))
    size(A, 2) == ncols || throw(
        DimensionMismatch(
            "A should have as many rows and columns as vars length ($ncols), $(size(A)) found",
        ),
    )
    size(S, 2) == ncols || throw(
        DimensionMismatch(
            "S should have as many rows and columns as vars length ($ncols), $(size(S)) found",
        ),
    )
    size(F, 2) == ncols || throw(
        DimensionMismatch(
            "F should have as many columns as vars length ($ncols), $(size(F, 2)) found",
        ),
    )
    if !isnothing(M)
        length(M) == ncols || throw(
            DimensionMismatch(
                "M should have as many elements as vars length ($ncols), $(length(M)) found",
            ),
        )
    end
    check_params(params, nothing)

    A = ParamsMatrix{Float64}(A, params)
    S = ParamsMatrix{Float64}(S, params)
    M = !isnothing(M) ? ParamsVector{Float64}(M, params) : nothing
    spF = sparse(F)
    if any(!isone, spF.nzval)
        throw(ArgumentError("F should contain only 0s and 1s"))
    end
    return RAMMatrices(A, S, F, M, copy(params), vars)
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
    n_vars = n_observed + n_latent

    if length(partable.sorted_vars) != 0
        @assert length(partable.sorted_vars) == nvars(partable)
        vars_sorted = copy(partable.sorted_vars)
    else
        vars_sorted = [
            partable.observed_vars
            partable.latent_vars
        ]
    end

    # indices of the vars (A/S/M rows or columns)
    vars_index = Dict(col => i for (i, col) in enumerate(vars_sorted))

    # fill Matrices
    # known_labels = Dict{Symbol, Int64}()

    T = nonmissingtype(eltype(partable.columns[:value_fixed]))
    A_inds = [Vector{Int64}() for _ in 1:length(params)]
    A_lin_ixs = LinearIndices((n_vars, n_vars))
    S_inds = [Vector{Int64}() for _ in 1:length(params)]
    S_lin_ixs = LinearIndices((n_vars, n_vars))
    A_consts = Vector{Pair{Int, T}}()
    S_consts = Vector{Pair{Int, T}}()
    # is there a meanstructure?
    M_inds =
        any(==(Symbol("1")), partable.columns[:from]) ?
        [Vector{Int64}() for _ in 1:length(params)] : nothing
    M_consts = !isnothing(M_inds) ? Vector{Pair{Int, T}}() : nothing

    for r in partable
        row_ind = vars_index[r.to]
        col_ind = r.from != Symbol("1") ? vars_index[r.from] : nothing

        if !r.free
            if (r.relation == :→) && (r.from == Symbol("1"))
                push!(M_consts, row_ind => r.value_fixed)
            elseif r.relation == :→
                push!(
                    A_consts,
                    A_lin_ixs[CartesianIndex(row_ind, col_ind)] => r.value_fixed,
                )
            elseif r.relation == :↔
                push!(
                    S_consts,
                    S_lin_ixs[CartesianIndex(row_ind, col_ind)] => r.value_fixed,
                )
                if row_ind != col_ind # symmetric
                    push!(
                        S_consts,
                        S_lin_ixs[CartesianIndex(col_ind, row_ind)] => r.value_fixed,
                    )
                end
            else
                error("Unsupported relation: $(r.relation)")
            end
        else
            par_ind = params_index[r.param]
            if (r.relation == :→) && (r.from == Symbol("1"))
                push!(M_inds[par_ind], row_ind)
            elseif r.relation == :→
                push!(A_inds[par_ind], A_lin_ixs[CartesianIndex(row_ind, col_ind)])
            elseif r.relation == :↔
                push!(S_inds[par_ind], S_lin_ixs[CartesianIndex(row_ind, col_ind)])
                if row_ind != col_ind # symmetric
                    push!(S_inds[par_ind], S_lin_ixs[CartesianIndex(col_ind, row_ind)])
                end
            else
                error("Unsupported relation: $(r.relation)")
            end
        end
    end
    # sort linear indices
    for A_ind in A_inds
        sort!(A_ind)
    end
    for S_ind in S_inds
        unique!(sort!(S_ind)) # also symmetric duplicates
    end
    if !isnothing(M_inds)
        for M_ind in M_inds
            sort!(M_ind)
        end
    end
    sort!(A_consts, by = first)
    sort!(S_consts, by = first)
    if !isnothing(M_consts)
        sort!(M_consts, by = first)
    end

    return RAMMatrices(
        ParamsMatrix{T}(A_inds, A_consts, (n_vars, n_vars)),
        ParamsMatrix{T}(S_inds, S_consts, (n_vars, n_vars)),
        eachrow_to_col(T, [vars_index[var] for var in partable.observed_vars], n_vars),
        !isnothing(M_inds) ? ParamsVector{T}(M_inds, M_consts, (n_vars,)) : nothing,
        params,
        vars_sorted,
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
    ram::RAMMatrices;
    params::Union{AbstractVector{Symbol}, Nothing} = nothing,
    observed_var_prefix::Symbol = :obs,
    latent_var_prefix::Symbol = :var,
)
    # defer parameter checks until we know which ones are used

    if !isnothing(ram.vars)
        latent_vars = SEM.latent_vars(ram)
        observed_vars = SEM.observed_vars(ram)
        vars = ram.vars
    else
        observed_vars = [Symbol("$(observed_var_prefix)_$i") for i in 1:nobserved_vars(ram)]
        latent_vars = [Symbol("$(latent_var_prefix)_$i") for i in 1:nlatent_vars(ram)]
        vars = vcat(observed_vars, latent_vars)
    end

    # construct an empty table
    partable = ParameterTable(
        observed_vars = observed_vars,
        latent_vars = latent_vars,
        params = isnothing(params) ? SEM.params(ram) : params,
    )

    # fill the table
    append_rows!(partable, ram.S, :S, ram.params, vars, skip_symmetric = true)
    append_rows!(partable, ram.A, :A, ram.params, vars)
    if !isnothing(ram.M)
        append_rows!(partable, ram.M, :M, ram.params, vars)
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

# generates a ParTable row NamedTuple for a given element of RAM matrix
function partable_row(
    val,
    index,
    matrix::Symbol,
    varnames::AbstractVector{Symbol};
    free::Bool = true,
)

    # variable names
    if matrix == :M
        from = Symbol("1")
        to = varnames[index]
    else
        from = varnames[index[2]]
        to = varnames[index[1]]
    end

    return (
        from = from,
        relation = matrix_to_relation(matrix),
        to = to,
        free = free,
        value_fixed = free ? 0.0 : val,
        start = 0.0,
        estimate = 0.0,
        param = free ? val : :const,
    )
end

function append_rows!(
    partable::ParameterTable,
    arr::ParamsArray,
    arr_name::Symbol,
    params::AbstractVector,
    varnames::AbstractVector{Symbol};
    skip_symmetric::Bool = false,
)
    nparams(arr) == length(params) || throw(
        ArgumentError(
            "Length of parameters vector ($(length(params))) does not match the number of parameters in the matrix ($(nparams(arr)))",
        ),
    )
    arr_ixs = eachindex(arr)

    # add parameters
    visited_indices = Set{eltype(arr_ixs)}()
    for (i, par) in enumerate(params)
        for j in param_occurences_range(arr, i)
            arr_ix = arr_ixs[arr.linear_indices[j]]
            skip_symmetric && (arr_ix ∈ visited_indices) && continue

            push!(partable, partable_row(par, arr_ix, arr_name, varnames, free = true))
            if skip_symmetric
                # mark index and its symmetric as visited
                push!(visited_indices, arr_ix)
                push!(visited_indices, CartesianIndex(arr_ix[2], arr_ix[1]))
            end
        end
    end

    # add constants
    for (i, _, val) in arr.constants
        arr_ix = arr_ixs[i]
        skip_symmetric && (arr_ix ∈ visited_indices) && continue
        push!(partable, partable_row(val, arr_ix, arr_name, varnames, free = false))
        if skip_symmetric
            # mark index and its symmetric as visited
            push!(visited_indices, arr_ix)
            push!(visited_indices, CartesianIndex(arr_ix[2], arr_ix[1]))
        end
    end

    return nothing
end

function Base.:(==)(mat1::RAMMatrices, mat2::RAMMatrices)
    res = (
        (mat1.A == mat2.A) &&
        (mat1.S == mat2.S) &&
        (mat1.F == mat2.F) &&
        (mat1.M == mat2.M) &&
        (mat1.params == mat2.params) &&
        (mat1.vars == mat2.vars)
    )
    return res
end
