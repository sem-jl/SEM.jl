function sem_summary(
    sem_fit::SemFit;
    show_fitmeasures = false,
    color = :light_cyan,
    digits = 2,
)
    print("\n")
    println("Fitted Structural Equation Model")
    print("\n")
    printstyled(
        "--------------------------------- Properties --------------------------------- \n";
        color = color,
    )
    print("\n")
    println("Optimization algorithm:      $(optimizer(sem_fit))")
    println("Convergence:                 $(convergence(sem_fit))")
    println("No. iterations/evaluations:  $(n_iterations(sem_fit))")
    print("\n")
    println("Number of parameters:        $(nparams(sem_fit))")
    println("Number of data samples:      $(nsamples(sem_fit))")
    print("\n")
    printstyled(
        "----------------------------------- Model ----------------------------------- \n";
        color = color,
    )
    print("\n")
    print(sem_fit.model)
    print("\n")
    if show_fitmeasures
        printstyled(
            "--------------------------------- Fitmeasures --------------------------------- \n";
            color = color,
        )
        print("\n")
        a = fit_measures(sem_fit)
        goal_length = maximum(length.(string.(keys(a)))) + 4
        for k in keys(a)
            key_length = length(string(k))
            print(k)
            print(repeat(" ", goal_length - key_length))
            print(round(a[k]; digits = 2))
            print("\n")
        end
    end
    print("\n")
end

function sem_summary(
    partable::ParameterTable;
    color = :light_cyan,
    secondary_color = :light_yellow,
    digits = 2,
    show_variables = true,
    show_columns = nothing
)
    if show_variables
        print("\n")
        printstyled(
            "--------------------------------- Variables --------------------------------- \n";
            color = color,
        )
        print("\n")
        printstyled("Latent variables:    "; color = color)
        for var in partable.latent_vars
            print("$var ")
        end
        print("\n")
        printstyled("Observed variables:  "; color = color)
        for var in partable.observed_vars
            print("$var ")
        end
        print("\n")
        if length(partable.sorted_vars) > 0
            printstyled("Sorted variables:    "; color = color)
            for var in partable.sorted_vars
                print("$var ")
            end
            print("\n")
        end
    end

    print("\n")
    printstyled(
        "---------------------------- Parameter Estimates ----------------------------- \n";
        color = color,
    )
    print("\n")

    columns = keys(partable.columns)
    show_columns = isnothing(show_columns) ? nothing : intersect(show_columns, columns)

    printstyled("Loadings: \n"; color = color)
    print("\n")

    if isnothing(show_columns)
        sorted_columns = [:to, :estimate, :param, :value_fixed, :start]
        loading_columns = sort_partially(sorted_columns, columns)
        header_cols = copy(loading_columns)
    else
        loading_columns = copy(show_columns)
        header_cols = copy(loading_columns)
    end

    for var in partable.latent_vars
        indicator_indices = findall(
            r ->
                (r.from == var) && (r.relation == :→) && (r.to ∈ partable.observed_vars),
            partable,
        )
        loading_array = reduce(
            hcat,
            check_round(partable.columns[c][indicator_indices]; digits = digits) for
            c in loading_columns
        )

        printstyled(var; color = secondary_color)
        print("\n")
        print("\n")
        pretty_table(
            loading_array;
            header = header_cols,
            tf = PrettyTables.tf_borderless,
            alignment = :l,
            formatters = (v, i, j) -> isa(v, Number) && isnan(v) ? "" : v,
        )
        print("\n")
    end

    printstyled("Directed Effects: \n"; color = color)

    regression_indices = findall(
        r ->
            (r.relation == :→) && (
                ((r.to ∈ partable.observed_vars) && (r.from ∈ partable.observed_vars)) ||
                ((r.to ∈ partable.latent_vars) && (r.from ∈ partable.observed_vars)) ||
                ((r.to ∈ partable.latent_vars) && (r.from ∈ partable.latent_vars))
            ),
        partable,
    )

    if isnothing(show_columns)
        sorted_columns = [:from, :relation, :to, :estimate, :param, :value_fixed, :start]
        regression_columns = sort_partially(sorted_columns, columns)
    else
        regression_columns = copy(show_columns)
    end

    regression_array = reduce(
        hcat,
        check_round(partable.columns[c][regression_indices]; digits = digits) for
        c in regression_columns
    )
    regression_columns[2] = regression_columns[2] == :relation ? Symbol("") : regression_columns[2]

    print("\n")
    pretty_table(
        regression_array;
        header = regression_columns,
        tf = PrettyTables.tf_borderless,
        alignment = :l,
        formatters = (v, i, j) -> isa(v, Number) && isnan(v) ? "" : v,
    )
    print("\n")

    printstyled("Variances: \n"; color = color)

    var_indices = findall(r -> r.relation == :↔ && r.to == r.from, partable)

    if isnothing(show_columns)
        sorted_columns = [:from, :relation, :to, :estimate, :param, :value_fixed, :start]
        var_columns = sort_partially(sorted_columns, columns)
    else
        var_columns = copy(show_columns)
    end

    var_array = reduce(
        hcat,
        check_round(partable.columns[c][var_indices]; digits) for c in var_columns
    )
    var_columns[2] = var_columns[2] == :relation ? Symbol("") : var_columns[2]

    print("\n")
    pretty_table(
        var_array;
        header = var_columns,
        tf = PrettyTables.tf_borderless,
        alignment = :l,
        formatters = (v, i, j) -> isa(v, Number) && isnan(v) ? "" : v,
    )
    print("\n")

    printstyled("Covariances: \n"; color = color)

    covar_indices = findall(r -> r.relation == :↔ && r.to != r.from, partable)

    if isnothing(show_columns)
        covar_columns = sort_partially(sorted_columns, columns)
    else
        covar_columns = copy(show_columns)
    end

    covar_array = reduce(
        hcat,
        check_round(partable.columns[c][covar_indices]; digits = digits) for
        c in covar_columns
    )
    covar_columns[2] = covar_columns[2] == :relation ? Symbol("") : covar_columns[2]

    print("\n")
    pretty_table(
        covar_array;
        header = covar_columns,
        tf = PrettyTables.tf_borderless,
        alignment = :l,
        formatters = (v, i, j) -> isa(v, Number) && isnan(v) ? "" : v,
    )
    print("\n")

    mean_indices = findall(r -> (r.relation == :→) && (r.from == Symbol("1")), partable)

    if length(mean_indices) > 0
        printstyled("Means: \n"; color = color)

        if isnothing(show_columns)
            sorted_columns = [:from, :relation, :to, :estimate, :param, :value_fixed, :start]
            mean_columns = sort_partially(sorted_columns, columns)
        else
            mean_columns = copy(show_columns)
        end

        mean_array = reduce(
            hcat,
            check_round(partable.columns[c][mean_indices]; digits = digits) for
            c in mean_columns
        )
        mean_columns[2] = mean_columns[2] == :relation ? Symbol("") : mean_columns[2]

        print("\n")
        pretty_table(
            mean_array;
            header = mean_columns,
            tf = PrettyTables.tf_borderless,
            alignment = :l,
            formatters = (v, i, j) -> isa(v, Number) && isnan(v) ? "" : v,
        )
        print("\n")
    end

    #printstyled("""No need to copy and paste results, you can use CSV.write(DataFrame(my_partable), "myfile.csv")"""; hidden = true)

end

function sem_summary(
    partable::EnsembleParameterTable;
    color = :light_cyan,
    secondary_color = :light_yellow,
    digits = 2,
    show_variables = true,
    show_columns = nothing
)
    if show_variables
        print("\n")
        printstyled(
            "--------------------------------- Variables --------------------------------- \n";
            color = color,
        )
        print("\n")
        let partable = partable.tables[[keys(partable.tables)...][1]]
            printstyled("Latent variables:    "; color = color)
            for var in partable.latent_vars
                print("$var ")
            end
            print("\n")
            printstyled("Observed variables:  "; color = color)
            for var in partable.observed_vars
                print("$var ")
            end
            print("\n")
            if length(partable.sorted_vars) > 0
                printstyled("Sorted variables:    "; color = color)
                for var in partable.sorted_vars
                    print("$var ")
                end
                print("\n")
            end
        end
    end
    print("\n")

    for k in keys(partable.tables)
        print("\n")
        printstyled(rpad(" Group: $k", 78), reverse = true)
        print("\n")
        sem_summary(
            partable.tables[k];
            color = color,
            secondary_color = secondary_color,
            digits = digits,
            show_variables = false,
            show_columns = show_columns
        )
    end

    # printstyled("""No need to copy and paste results, you can use CSV.write(DataFrame(my_partable), "myfile.csv")"""; hidden = true)

end

function check_round(vec; digits)
    if eltype(vec) <: Number
        return round.(vec; digits = digits)
    end
    return vec
end

function sort_partially(sorted, to_sort)
    out = Vector{eltype(to_sort)}()
    for el in sorted
        if el ∈ to_sort
            push!(out, el)
        end
    end
    remaining = setdiff(to_sort, sorted)
    append!(out, sort(collect(remaining)))
    return out
end

function Base.findall(fun::Function, partable::ParameterTable)
    rows = Int[]
    for (i, r) in enumerate(partable)
        fun(r) ? push!(rows, i) : nothing
    end
    return rows
end

"""
    (1) sem_summary(sem_fit::SemFit; show_fitmeasures = false)

    (2) sem_summary(partable::AbstractParameterTable; ...)

Print information about (1) a fitted SEM or (2) a parameter table to stdout.

# Extended help
## Addition keyword arguments
- `digits = 2`: controls precision of printed estimates, standard errors, etc.
- `color = :light_cyan`: color of some parts of the printed output. Can be adjusted for readability.
- `secondary_color = :light_yellow`
- `show_variables = true`
- `show_columns = nothing`: columns names to include in the output e.g.`[:from, :to, :estimate]`)
"""
function sem_summary end
