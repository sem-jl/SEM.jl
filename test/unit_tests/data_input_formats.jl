using StructuralEquationModels, Test, Statistics

### model specification --------------------------------------------------------------------

spec = ParameterTable(
    observed_vars = [:x1, :x2, :x3, :y1, :y2, :y3, :y4, :y5, :y6, :y7, :y8],
    latent_vars = [:ind60, :dem60, :dem65],
)

# specification with non-existent observed var z1
wrong_spec = ParameterTable(
    observed_vars = [:x1, :x2, :x3, :y1, :y2, :y3, :y4, :y5, :y6, :y7, :y8, :z1],
    latent_vars = [:ind60, :dem60, :dem65],
)

### data -----------------------------------------------------------------------------------

dat = example_data("political_democracy")
dat_missing = example_data("political_democracy_missing")[:, names(dat)]

@assert Symbol.(names(dat)) == observed_vars(spec)

dat_matrix = Matrix(dat)
dat_missing_matrix = Matrix(dat_missing)

dat_cov = Statistics.cov(dat_matrix)
dat_mean = vcat(Statistics.mean(dat_matrix, dims = 1)...)

# shuffle variables
new_order = [3, 2, 7, 8, 5, 6, 9, 11, 1, 10, 4]

shuffle_names = names(dat)[new_order]

shuffle_spec = ParameterTable(
    observed_vars = Symbol.(shuffle_names),
    latent_vars = [:ind60, :dem60, :dem65],
)

shuffle_dat = dat[:, new_order]
shuffle_dat_missing = dat_missing[:, new_order]

shuffle_dat_matrix = dat_matrix[:, new_order]
shuffle_dat_missing_matrix = dat_missing_matrix[:, new_order]

shuffle_dat_cov = cov(shuffle_dat_matrix)
shuffle_dat_mean = vec(mean(shuffle_dat_matrix, dims = 1))

# common tests for SemObserved subtypes
function test_observed(
    observed::SemObserved,
    dat,
    dat_matrix,
    dat_cov,
    dat_mean;
    meanstructure::Bool,
    approx_cov::Bool = false,
)
    if !isnothing(dat)
        @test @inferred(nsamples(observed)) == size(dat, 1)
        @test @inferred(nobserved_vars(observed)) == size(dat, 2)
        @test @inferred(observed_vars(observed)) == Symbol.(names(dat))
    end

    if !isnothing(dat_matrix)
        @test @inferred(nsamples(observed)) == size(dat_matrix, 1)

        if any(ismissing, dat_matrix)
            @test isequal(@inferred(samples(observed)), dat_matrix)
        else
            @test @inferred(samples(observed)) == dat_matrix
        end
    end

    if !isnothing(dat_cov)
        if any(ismissing, dat_cov)
            @test isequal(@inferred(obs_cov(observed)), dat_cov)
        else
            if approx_cov
                @test @inferred(obs_cov(observed)) ≈ dat_cov
            else
                @test @inferred(obs_cov(observed)) == dat_cov
            end
        end
    end

    # FIXME actually, SemObserved should not use meanstructure and always provide obs_mean()
    # since meanstructure belongs to the implied part of a SEM model
    if meanstructure
        if !isnothing(dat_mean)
            if any(ismissing, dat_mean)
                @test isequal(@inferred(obs_mean(observed)), dat_mean)
            else
                @test @inferred(obs_mean(observed)) == dat_mean
            end
        else
            @test @inferred(obs_mean(observed)) isa AbstractVector{Float64} # EM-based means
        end
    else
        @test @inferred(obs_mean(observed)) === nothing skip = true
    end
end

############################################################################################
@testset "SemObservedData" begin

    # errors
    obs_data_redundant = SemObservedData(
        specification = spec,
        data = dat,
        observed_vars = Symbol.(names(dat)),
    )
    @test observed_vars(obs_data_redundant) == Symbol.(names(dat))
    @test observed_vars(obs_data_redundant) == observed_vars(spec)

    obs_data_spec = SemObservedData(specification = spec, data = dat_matrix)
    @test observed_vars(obs_data_spec) == observed_vars(spec)

    obs_data_strnames =
        SemObservedData(specification = spec, data = dat_matrix, observed_vars = names(dat))
    @test observed_vars(obs_data_strnames) == Symbol.(names(dat))

    @test_throws UndefKeywordError(:data) SemObservedData(specification = spec)

    obs_data_nonames = SemObservedData(data = dat_matrix)
    @test observed_vars(obs_data_nonames) == Symbol.("obs", 1:size(dat_matrix, 2))

    obs_data_nonames2 =
        SemObservedData(data = dat_matrix, observed_var_prefix = "observed_")
    @test observed_vars(obs_data_nonames2) == Symbol.("observed_", 1:size(dat_matrix, 2))

    @testset "meanstructure=$meanstructure" for meanstructure in (false, true)
        observed = SemObservedData(specification = spec, data = dat; meanstructure)

        test_observed(observed, dat, dat_matrix, dat_cov, dat_mean; meanstructure)

        observed_nospec =
            SemObservedData(specification = nothing, data = dat_matrix; meanstructure)

        test_observed(
            observed_nospec,
            nothing,
            dat_matrix,
            dat_cov,
            dat_mean;
            meanstructure,
        )

        observed_matrix = SemObservedData(
            specification = spec,
            data = dat_matrix,
            observed_vars = Symbol.(names(dat));
            meanstructure,
        )

        test_observed(observed_matrix, dat, dat_matrix, dat_cov, dat_mean; meanstructure)

        # detect non-existing column
        @test_throws "ArgumentError: column name \"z1\"" SemObservedData(
            specification = wrong_spec,
            data = shuffle_dat,
        )

        # detect non-existing observed_var
        @test_throws "ArgumentError: observed_var \"z1\"" SemObservedData(
            specification = wrong_spec,
            data = shuffle_dat_matrix,
            observed_vars = shuffle_names,
        )

        # cannot infer observed_vars
        @test_throws "No data, specification or observed_vars provided" SemObservedData(
            data = nothing,
        )

        if false # FIXME data = nothing is for simulation studies
            # no data, just observed_vars
            observed_nodata =
                SemObservedData(data = nothing, observed_vars = Symbol.(names(dat)))
            @test observed_nodata isa SemObservedData
            @test @inferred(samples(observed_nodata)) === nothing
            @test observed_vars(observed_nodata) == Symbol.(names(dat))
        end

        @test_warn "The order of variables in observed_vars" SemObservedData(
            specification = spec,
            data = shuffle_dat,
            observed_vars = shuffle_names,
        )

        # spec takes precedence in obs_vars order
        observed_spec = SemObservedData(
            specification = spec,
            data = shuffle_dat,
            observed_vars = shuffle_names,
        )

        test_observed(
            observed_spec,
            dat,
            dat_matrix,
            dat_cov,
            meanstructure ? dat_mean : nothing;
            meanstructure,
        )

        observed_shuffle =
            SemObservedData(specification = shuffle_spec, data = shuffle_dat; meanstructure)

        test_observed(
            observed_shuffle,
            shuffle_dat,
            shuffle_dat_matrix,
            shuffle_dat_cov,
            meanstructure ? shuffle_dat_mean : nothing;
            meanstructure,
        )

        observed_matrix_shuffle = SemObservedData(
            specification = shuffle_spec,
            data = shuffle_dat_matrix,
            observed_vars = shuffle_names;
            meanstructure,
        )

        test_observed(
            observed_matrix_shuffle,
            shuffle_dat,
            shuffle_dat_matrix,
            shuffle_dat_cov,
            meanstructure ? shuffle_dat_mean : nothing;
            meanstructure,
        )
    end # meanstructure
end # SemObservedData

############################################################################################

@testset "SemObservedCovariance" begin

    # errors

    @test_throws UndefKeywordError(:nsamples) SemObservedCovariance(obs_cov = dat_cov)

    @testset "meanstructure=$meanstructure" for meanstructure in (false, true)

        # errors
        @test_throws UndefKeywordError SemObservedCovariance(
            obs_cov = dat_cov;
            meanstructure,
        )

        @test_throws UndefKeywordError SemObservedCovariance(
            data = dat_matrix;
            meanstructure,
        )

        # default vars
        observed_nonames = SemObservedCovariance(
            obs_cov = dat_cov,
            obs_mean = meanstructure ? dat_mean : nothing,
            nsamples = size(dat, 1),
        )
        @test observed_vars(observed_nonames) == Symbol.("obs", 1:size(dat_cov, 2))

        @test_throws DimensionMismatch SemObservedCovariance(
            obs_cov = dat_cov,
            observed_vars = Symbol.("obs", 1:(size(dat_cov, 2)+1)),
            nsamples = size(dat, 1),
        )

        observed = SemObservedCovariance(
            specification = spec,
            obs_cov = dat_cov,
            obs_mean = meanstructure ? dat_mean : nothing,
            observed_vars = Symbol.(names(dat)),
            nsamples = size(dat, 1),
            meanstructure = meanstructure,
        )

        test_observed(
            observed,
            dat,
            nothing,
            dat_cov,
            dat_mean;
            meanstructure,
            approx_cov = true,
        )

        @test @inferred(samples(observed)) === nothing

        observed_nospec = SemObservedCovariance(
            specification = nothing,
            obs_cov = dat_cov,
            obs_mean = meanstructure ? dat_mean : nothing,
            nsamples = size(dat, 1);
            meanstructure,
        )

        test_observed(
            observed_nospec,
            nothing,
            nothing,
            dat_cov,
            dat_mean;
            meanstructure,
            approx_cov = true,
        )

        @test @inferred(samples(observed_nospec)) === nothing

        # detect non-existing observed_var
        @test_throws "ArgumentError: observed_var \"z1\"" SemObservedCovariance(
            specification = wrong_spec,
            obs_cov = shuffle_dat_cov,
            observed_vars = shuffle_names,
            nsamples = size(dat, 1),
        )

        # spec takes precedence in obs_vars order
        observed_spec = SemObservedCovariance(
            specification = spec,
            obs_cov = shuffle_dat_cov,
            obs_mean = meanstructure ? shuffle_dat_mean : nothing,
            observed_vars = shuffle_names,
            nsamples = size(dat, 1),
        )

        test_observed(
            observed_spec,
            dat,
            nothing,
            dat_cov,
            meanstructure ? dat_mean : nothing;
            meanstructure,
            approx_cov = true,
        )

        observed_shuffle = SemObservedCovariance(
            specification = shuffle_spec,
            obs_cov = shuffle_dat_cov,
            obs_mean = meanstructure ? shuffle_dat_mean : nothing,
            observed_vars = shuffle_names,
            nsamples = size(dat, 1);
            meanstructure,
        )

        test_observed(
            observed_shuffle,
            shuffle_dat,
            nothing,
            shuffle_dat_cov,
            meanstructure ? shuffle_dat_mean : nothing;
            meanstructure,
            approx_cov = true,
        )
    end # meanstructure
end # SemObservedCovariance

############################################################################################

@testset "SemObservedMissing" begin

    # errors
    observed_redundant_names = SemObservedMissing(
        specification = spec,
        data = dat_missing,
        observed_vars = Symbol.(names(dat)),
    )
    @test observed_vars(observed_redundant_names) == Symbol.(names(dat))

    observed_spec_only = SemObservedMissing(specification = spec, data = dat_missing_matrix)
    @test observed_vars(observed_spec_only) == observed_vars(spec)

    observed_str_colnames = SemObservedMissing(
        specification = spec,
        data = dat_missing_matrix,
        observed_vars = names(dat),
    )
    @test observed_vars(observed_str_colnames) == Symbol.(names(dat))

    @test_throws UndefKeywordError(:data) SemObservedMissing(specification = spec)

    observed_no_names = SemObservedMissing(data = dat_missing_matrix)
    @test observed_vars(observed_no_names) == Symbol.(:obs, 1:size(dat_missing_matrix, 2))

    observed_no_names2 =
        SemObservedMissing(data = dat_missing_matrix, observed_var_prefix = "observed_")
    @test observed_vars(observed_no_names2) == Symbol.("observed_", 1:size(dat_matrix, 2))

    @testset "meanstructure=$meanstructure" for meanstructure in (false, true)
        observed =
            SemObservedMissing(specification = spec, data = dat_missing; meanstructure)

        test_observed(
            observed,
            dat_missing,
            dat_missing_matrix,
            nothing,
            nothing;
            meanstructure,
        )

        @test @inferred(length(observed.patterns)) == 55
        @test sum(@inferred(nsamples(pat)) for pat in observed.patterns) ==
              size(dat_missing, 1)
        @test all(nsamples(pat) <= size(dat_missing, 2) for pat in observed.patterns)

        observed_nospec = SemObservedMissing(
            specification = nothing,
            data = dat_missing_matrix;
            meanstructure,
        )

        test_observed(
            observed_nospec,
            nothing,
            dat_missing_matrix,
            nothing,
            nothing;
            meanstructure,
        )

        observed_matrix = SemObservedMissing(
            specification = spec,
            data = dat_missing_matrix,
            observed_vars = Symbol.(names(dat)),
        )

        test_observed(
            observed_matrix,
            dat_missing,
            dat_missing_matrix,
            nothing,
            nothing;
            meanstructure,
        )

        # detect non-existing column
        @test_throws "ArgumentError: column name \"z1\"" SemObservedMissing(
            specification = wrong_spec,
            data = shuffle_dat,
        )

        # detect non-existing observed_var
        @test_throws "ArgumentError: observed_var \"z1\"" SemObservedMissing(
            specification = wrong_spec,
            data = shuffle_dat_missing_matrix,
            observed_vars = shuffle_names,
        )

        # spec takes precedence in obs_vars order
        observed_spec = SemObservedMissing(
            specification = spec,
            observed_vars = shuffle_names,
            data = shuffle_dat_missing,
        )

        test_observed(
            observed_spec,
            dat_missing,
            dat_missing_matrix,
            nothing,
            nothing;
            meanstructure,
        )

        observed_shuffle =
            SemObservedMissing(specification = shuffle_spec, data = shuffle_dat_missing)

        test_observed(
            observed_shuffle,
            shuffle_dat_missing,
            shuffle_dat_missing_matrix,
            nothing,
            nothing;
            meanstructure,
        )

        observed_matrix_shuffle = SemObservedMissing(
            specification = shuffle_spec,
            data = shuffle_dat_missing_matrix,
            observed_vars = shuffle_names,
        )

        test_observed(
            observed_matrix_shuffle,
            shuffle_dat_missing,
            shuffle_dat_missing_matrix,
            nothing,
            nothing;
            meanstructure,
        )
    end # meanstructure
end # SemObservedMissing
