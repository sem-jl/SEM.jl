# check if params vector correctly matches the parameter references (from the ParameterTable)
function check_params(
    params::AbstractVector{Symbol},
    param_refs::Union{AbstractVector{Symbol}, Nothing},
)
    dup_params = nonunique(params)
    isempty(dup_params) ||
        throw(ArgumentError("Duplicate parameters detected: $(join(dup_params, ", "))"))
    any(==(:const), params) &&
        throw(ArgumentError("Parameters constain reserved :const name"))

    if !isnothing(param_refs)
        # check if all references parameters are present
        all_refs = Set(id for id in param_refs if id != :const)
        undecl_params = setdiff(all_refs, params)
        if !isempty(undecl_params)
            throw(
                ArgumentError(
                    "The following $(length(undecl_params)) parameters present in the table, but are not declared: " *
                    join(sort!(collect(undecl_params))),
                ),
            )
        end
    end

    return nothing
end

function check_vars(vars::AbstractVector{Symbol}, nvars::Union{Integer, Nothing})
    isnothing(nvars) ||
        length(vars) == nvars ||
        throw(
            DimensionMismatch(
                "variables length ($(length(vars))) does not match the number of columns in A matrix ($nvars)",
            ),
        )
    dup_vars = nonunique(vars)
    isempty(dup_vars) ||
        throw(ArgumentError("Duplicate variables detected: $(join(dup_vars, ", "))"))

    return nothing
end
