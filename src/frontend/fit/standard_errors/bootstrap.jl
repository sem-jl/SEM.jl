"""
    se_bootstrap(semfit::SemFit; n_boot = 3000, data = nothing, kwargs...)

Return boorstrap standard errors.
Only works for single models.

# Arguments
- `n_boot`: number of boostrap samples
- `data`: data to sample from. Only needed if different than the data from `sem_fit`
- `kwargs...`: passed down to `swap_observed`
"""
function se_bootstrap(
    semfit::SemFit;
    n_boot = 3000,
    data = nothing,
    specification = nothing,
    kwargs...,
)
    if model(semfit) isa AbstractSemCollection
        throw(
            ArgumentError(
                "bootstrap standard errors for ensemble models are not available yet",
            ),
        )
    end

    if isnothing(data)
        data = samples(observed(model(semfit)))
    end

    data = prepare_data_bootstrap(data)

    start = solution(semfit)

    new_solution = zero(start)
    sum = zero(start)
    squared_sum = zero(start)

    n_failed = 0.0

    converged = true

    for _ in 1:n_boot
        sample_data = bootstrap_sample(data)
        new_model = swap_observed(
            model(semfit);
            data = sample_data,
            specification = specification,
            kwargs...,
        )

        new_solution .= 0.0

        try
            new_solution = solution(sem_fit(new_model; start_val = start))
        catch
            n_failed += 1
        end

        @. sum += new_solution
        @. squared_sum += new_solution^2

        converged = true
    end

    n_conv = n_boot - n_failed
    sd = sqrt.(squared_sum / n_conv - (sum / n_conv) .^ 2)
    print("Number of nonconverged models: ", n_failed, "\n")
    return sd
end

function prepare_data_bootstrap(data)
    return Matrix(data)
end

function bootstrap_sample(data)
    nobs = size(data, 1)
    index_new = rand(1:nobs, nobs)
    data_new = data[index_new, :]
    return data_new
end
