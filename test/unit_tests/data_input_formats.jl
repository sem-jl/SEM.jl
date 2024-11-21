using StructuralEquationModels, Test, Statistics

### model specification --------------------------------------------------------------------

spec = ParameterTable(
    observed_vars = [:x1, :x2, :x3, :y1, :y2, :y3, :y4, :y5, :y6, :y7, :y8],
    latent_vars = [:ind60, :dem60, :dem65],
)

### data -----------------------------------------------------------------------------------

dat = example_data("political_democracy")
dat_missing = example_data("political_democracy_missing")[:, names(dat)]

dat_matrix = Matrix(dat)
dat_missing_matrix = Matrix(dat_missing)

dat_cov = Statistics.cov(dat_matrix)
dat_mean = vcat(Statistics.mean(dat_matrix, dims = 1)...)

# shuffle variables
new_order = [3, 2, 7, 8, 5, 6, 9, 11, 1, 10, 4]

shuffle_names = Symbol.(names(dat))[new_order]

shuffle_dat = dat[:, new_order]
shuffle_dat_missing = dat_missing[:, new_order]

shuffle_dat_matrix = dat_matrix[:, new_order]
shuffle_dat_missing_matrix = dat_missing_matrix[:, new_order]

shuffle_dat_cov = Statistics.cov(shuffle_dat_matrix)
shuffle_dat_mean = vcat(Statistics.mean(shuffle_dat_matrix, dims = 1)...)

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
    @test @inferred(nobserved_vars(observed)) == size(dat, 2)
    # FIXME observed should provide names of observed variables
    @test @inferred(observed_vars(observed)) == names(dat) broken = true
    @test @inferred(nsamples(observed)) == size(dat, 1)

    hasmissing =
        !isnothing(dat_matrix) && any(ismissing, dat_matrix) ||
        !isnothing(dat_cov) && any(ismissing, dat_cov)

    if !isnothing(dat_matrix)
        if hasmissing
            @test isequal(@inferred(samples(observed)), dat_matrix)
        else
            @test @inferred(samples(observed)) == dat_matrix
        end
    end

    if !isnothing(dat_cov)
        if hasmissing
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
    # meanstructure is a part of SEM model
    if meanstructure
        if !isnothing(dat_mean)
            if hasmissing
                @test isequal(@inferred(obs_mean(observed)), dat_mean)
            else
                @test @inferred(obs_mean(observed)) == dat_mean
            end
        else
            # FIXME if meanstructure is present, obs_mean() should provide something (currently Missing don't support it)
            @test (@inferred(obs_mean(observed)) isa AbstractVector{Float64}) broken = true
        end
    else
        @test @inferred(obs_mean(observed)) === nothing skip = true
    end
end

############################################################################################
@testset "SemObservedData" begin

    # errors
    @test_throws ArgumentError(
        "You passed your data as a `DataFrame`, but also specified `obs_colnames`. " *
        "Please make sure the column names of your data frame indicate the correct variables " *
        "or pass your data in a different format.",
    ) begin
        SemObservedData(
            specification = spec,
            data = dat,
            obs_colnames = Symbol.(names(dat)),
        )
    end

    @test_throws ArgumentError(
        "Your `data` can not be indexed by symbols. " *
        "Maybe you forgot to provide column names via the `obs_colnames = ...` argument.",
    ) begin
        SemObservedData(specification = spec, data = dat_matrix)
    end

    @test_throws ArgumentError("please specify `obs_colnames` as a vector of Symbols") begin
        SemObservedData(specification = spec, data = dat_matrix, obs_colnames = names(dat))
    end

    @test_throws UndefKeywordError(:data) SemObservedData(specification = spec)

    @test_throws UndefKeywordError(:specification) SemObservedData(data = dat_matrix)

    @testset "meanstructure=$meanstructure" for meanstructure in (false, true)
        observed = SemObservedData(specification = spec, data = dat; meanstructure)

        test_observed(observed, dat, dat_matrix, dat_cov, dat_mean; meanstructure)

        observed_nospec =
            SemObservedData(specification = nothing, data = dat_matrix; meanstructure)

        test_observed(observed_nospec, dat, dat_matrix, dat_cov, dat_mean; meanstructure)

        observed_matrix = SemObservedData(
            specification = spec,
            data = dat_matrix,
            obs_colnames = Symbol.(names(dat)),
            meanstructure = meanstructure,
        )

        test_observed(observed_matrix, dat, dat_matrix, dat_cov, dat_mean; meanstructure)

        observed_shuffle =
            SemObservedData(specification = spec, data = shuffle_dat; meanstructure)

        test_observed(observed_shuffle, dat, dat_matrix, dat_cov, dat_mean; meanstructure)

        observed_matrix_shuffle = SemObservedData(
            specification = spec,
            data = shuffle_dat_matrix,
            obs_colnames = shuffle_names;
            meanstructure,
        )

        test_observed(
            observed_matrix_shuffle,
            dat,
            dat_matrix,
            dat_cov,
            dat_mean;
            meanstructure,
        )
    end # meanstructure
end # SemObservedData

############################################################################################

@testset "SemObservedCovariance" begin

    # errors

    @test_throws UndefKeywordError(:nsamples) SemObservedCovariance(obs_cov = dat_cov)

    @test_throws ArgumentError("no `obs_colnames` were specified") begin
        SemObservedCovariance(
            specification = spec,
            obs_cov = dat_cov,
            nsamples = size(dat, 1),
        )
    end

    @test_throws ArgumentError("observed means were passed, but `meanstructure = false`") begin
        SemObservedCovariance(
            specification = nothing,
            obs_cov = dat_cov,
            obs_mean = dat_mean,
            nsamples = size(dat, 1),
        )
    end

    @test_throws ArgumentError("please specify `obs_colnames` as a vector of Symbols") begin
        SemObservedCovariance(
            specification = spec,
            obs_cov = dat_cov,
            obs_colnames = names(dat),
            nsamples = size(dat, 1),
            meanstructure = false,
        )
    end

    @test_throws ArgumentError("`meanstructure = true`, but no observed means were passed") begin
        SemObservedCovariance(
            specification = spec,
            obs_cov = dat_cov,
            obs_colnames = Symbol.(names(dat)),
            meanstructure = true,
            nsamples = size(dat, 1),
        )
    end

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

        # should work
        observed = SemObservedCovariance(
            specification = spec,
            obs_cov = dat_cov,
            obs_mean = meanstructure ? dat_mean : nothing,
            obs_colnames = obs_colnames = Symbol.(names(dat)),
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

        @test_throws ErrorException samples(observed)

        observed_nospec = SemObservedCovariance(
            specification = nothing,
            obs_cov = dat_cov,
            obs_mean = meanstructure ? dat_mean : nothing,
            nsamples = size(dat, 1);
            meanstructure,
        )

        test_observed(
            observed_nospec,
            dat,
            nothing,
            dat_cov,
            dat_mean;
            meanstructure,
            approx_cov = true,
        )

        @test_throws ErrorException samples(observed_nospec)

        observed_shuffle = SemObservedCovariance(
            specification = spec,
            obs_cov = shuffle_dat_cov,
            obs_mean = meanstructure ? dat_mean[new_order] : nothing,
            obs_colnames = shuffle_names,
            nsamples = size(dat, 1);
            meanstructure,
        )

        test_observed(
            observed_shuffle,
            dat,
            nothing,
            dat_cov,
            dat_mean;
            meanstructure,
            approx_cov = true,
        )

        @test_throws ErrorException samples(observed_shuffle)

        # respect specification order
        @test @inferred(obs_cov(observed_shuffle)) ≈ obs_cov(observed)
        @test @inferred(observed_vars(observed_shuffle)) == shuffle_names broken = true
    end # meanstructure
end # SemObservedCovariance

############################################################################################

@testset "SemObservedMissing" begin

    # errors
    @test_throws ArgumentError(
        "You passed your data as a `DataFrame`, but also specified `obs_colnames`. " *
        "Please make sure the column names of your data frame indicate the correct variables " *
        "or pass your data in a different format.",
    ) begin
        SemObservedMissing(
            specification = spec,
            data = dat_missing,
            obs_colnames = Symbol.(names(dat)),
        )
    end

    @test_throws ArgumentError(
        "Your `data` can not be indexed by symbols. " *
        "Maybe you forgot to provide column names via the `obs_colnames = ...` argument.",
    ) begin
        SemObservedMissing(specification = spec, data = dat_missing_matrix)
    end

    @test_throws ArgumentError("please specify `obs_colnames` as a vector of Symbols") begin
        SemObservedMissing(
            specification = spec,
            data = dat_missing_matrix,
            obs_colnames = names(dat),
        )
    end

    @test_throws UndefKeywordError(:data) SemObservedMissing(specification = spec)

    @test_throws UndefKeywordError(:specification) SemObservedMissing(
        data = dat_missing_matrix,
    )

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

        @test @inferred(length(StructuralEquationModels.patterns(observed))) == 55
        @test sum(@inferred(StructuralEquationModels.pattern_nsamples(observed))) ==
              size(dat_missing, 1)
        @test all(
            <=(size(dat_missing, 2)),
            @inferred(StructuralEquationModels.pattern_nsamples(observed))
        )

        observed_nospec = SemObservedMissing(
            specification = nothing,
            data = dat_missing_matrix;
            meanstructure,
        )

        test_observed(
            observed_nospec,
            dat_missing,
            dat_missing_matrix,
            nothing,
            nothing;
            meanstructure,
        )

        observed_matrix = SemObservedMissing(
            specification = spec,
            data = dat_missing_matrix,
            obs_colnames = Symbol.(names(dat)),
        )

        test_observed(
            observed_matrix,
            dat_missing,
            dat_missing_matrix,
            nothing,
            nothing;
            meanstructure,
        )

        observed_shuffle =
            SemObservedMissing(specification = spec, data = shuffle_dat_missing)

        test_observed(
            observed_shuffle,
            dat_missing,
            dat_missing_matrix,
            nothing,
            nothing;
            meanstructure,
        )

        observed_matrix_shuffle = SemObservedMissing(
            specification = spec,
            data = shuffle_dat_missing_matrix,
            obs_colnames = shuffle_names,
        )

        test_observed(
            observed_matrix_shuffle,
            dat_missing,
            dat_missing_matrix,
            nothing,
            nothing;
            meanstructure,
        )
    end # meanstructure
end # SemObservedMissing
