"""
    observed(loss::SemLoss) -> SemObserved

Returns the [*observed*](@ref SemObserved) part of a model.
"""
observed(loss::SemLoss) = loss.observed

"""
    implied(loss::SemLoss) -> SemImplied

Returns the [*implied*](@ref SemImplied) part of a model.
"""
implied(loss::SemLoss) = loss.implied

for f in (:nsamples, :obs_cov, :obs_mean)
    @eval $f(loss::SemLoss) = $f(observed(loss))
end

for f in (
    :vars,
    :nvars,
    :latent_vars,
    :nlatent_vars,
    :observed_vars,
    :nobserved_vars,
    :params,
    :nparams,
)
    @eval $f(loss::SemLoss) = $f(implied(loss))
end

function check_observed_vars(observed::SemObserved, implied::SemImplied)
    isnothing(observed_vars(implied)) ||
        observed_vars(observed) == observed_vars(implied) ||
        throw(
            ArgumentError(
                "Observed variables defined for \"observed\" and \"implied\" do not match.",
            ),
        )
end

check_observed_vars(sem::SemLoss) = check_observed_vars(observed(sem), implied(sem))
