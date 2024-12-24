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

############################################################################################
### tests - SemObservedData
############################################################################################

# w.o. means -------------------------------------------------------------------------------

# errors
@test_throws MethodError SemObservedData(specification = spec)

# should work
observed_nospec = SemObservedData(dat_matrix)

observed = SemObservedData(dat, specification = spec)

observed2 = SemObservedData(dat, specification = spec, obs_colnames = Symbol.(names(dat)))

observed_nospec = SemObservedData(dat_matrix, specification = nothing)

observed_matrix =
    SemObservedData(dat_matrix, specification = spec, obs_colnames = Symbol.(names(dat)))

observed_matrix2 =
    SemObservedData(dat_matrix, specification = spec, obs_colnames = names(dat))

observed_matrix3 = SemObservedData(dat_matrix, specification = spec)

@testset "unit tests | SemObservedData | input formats" begin
    @test obs_cov(observed2) == obs_cov(observed)
    @test obs_cov(observed_nospec) == obs_cov(observed)
    @test obs_cov(observed_matrix) == obs_cov(observed)
    @test get_data(observed_nospec) == get_data(observed)
    @test get_data(observed_matrix) == get_data(observed)
    @test get_data(observed_matrix2) == get_data(observed)
    @test get_data(observed_matrix3) == get_data(observed)
end

# shuffle variables
new_order = [3, 2, 7, 8, 5, 6, 9, 11, 1, 10, 4]

shuffle_names = Symbol.(names(dat))[new_order]

shuffle_dat = dat[:, new_order]

shuffle_dat_matrix = dat_matrix[:, new_order]

observed_shuffle = SemObservedData(shuffle_dat, specification = spec)

observed_matrix_shuffle =
    SemObservedData(shuffle_dat_matrix, specification = spec, obs_colnames = shuffle_names)

@testset "unit tests | SemObservedData | input formats shuffled " begin
    @test obs_cov(observed_shuffle) == obs_cov(observed)
    @test obs_cov(observed_matrix_shuffle) == obs_cov(observed)[new_order, new_order]

    @test get_data(observed_shuffle) == get_data(observed)
    @test get_data(observed_matrix_shuffle) == get_data(observed)[:, new_order]
end

# with means -------------------------------------------------------------------------------

# errors
@test_throws MethodError SemObservedData(specification = spec, meanstructure = true)

# should work
observed = SemObservedData(dat, specification = spec, meanstructure = true)

observed2 = SemObservedData(
    dat,
    specification = spec,
    obs_colnames = Symbol.(names(dat)),
    meanstructure = true,
)

observed_matrix = SemObservedData(
    dat_matrix,
    specification = spec,
    obs_colnames = names(dat),
    meanstructure = true,
)

observed_matrix_spec =
    SemObservedData(dat_matrix, specification = spec, meanstructure = true)

observed_nospec = SemObservedData(dat_matrix, specification = nothing, meanstructure = true)

observed_matrix2 = SemObservedData(
    dat_matrix,
    specification = spec,
    obs_colnames = Symbol.(names(dat)),
    meanstructure = true,
)

@testset "unit tests | SemObservedData | input formats - means" begin
    @test obs_mean(observed2) == obs_mean(observed)
    @test obs_mean(observed_matrix) == obs_mean(observed)
    @test obs_mean(observed_matrix_spec) == obs_mean(observed)
    @test obs_mean(observed_nospec) == obs_mean(observed)
    @test obs_mean(observed_matrix) == obs_mean(observed)
    @test obs_mean(observed_matrix2) == obs_mean(observed)
end

# shuffle variables
new_order = [3, 2, 7, 8, 5, 6, 9, 11, 1, 10, 4]

shuffle_names = Symbol.(names(dat))[new_order]

shuffle_dat = dat[:, new_order]

shuffle_dat_matrix = dat_matrix[:, new_order]

observed_shuffle = SemObservedData(shuffle_dat, specification = spec, meanstructure = true)

observed_matrix_shuffle = SemObservedData(
    shuffle_dat_matrix,
    specification = spec,
    obs_colnames = shuffle_names,
    meanstructure = true,
)

@testset "unit tests | SemObservedData | input formats shuffled - mean" begin
    @test obs_mean(observed_shuffle) == obs_mean(observed)
    @test obs_mean(observed_matrix_shuffle) == obs_mean(observed)[new_order]
end

############################################################################################
### tests - SemObservedCovariance
############################################################################################

# w.o. means -------------------------------------------------------------------------------

# errors

#@test_throws UndefKeywordError(:specification) SemObservedCovariance(obs_cov = dat_cov)

# should work
observed = SemObservedCovariance(
    dat_cov,
    specification = spec,
    obs_colnames = Symbol.(names(dat)),
    n_obs = 75,
)

observed_str = SemObservedCovariance(
    dat_cov,
    specification = spec,
    obs_colnames = names(dat),
    n_obs = 75,
)

observed_nospec2 = SemObservedCovariance(dat_cov, n_obs = 75)

@testset "unit tests | SemObservedCovariance | input formats" begin
    @test n_obs(observed) == 75
    @test n_obs(observed_nospec) == 75
    @test obs_cov(observed_nospec) == obs_cov(observed)

    @test n_obs(observed_str) == 75
    @test obs_cov(observed_str) == obs_cov(observed)

    @test n_obs(observed_nospec2) == 75
    @test obs_cov(observed_nospec2) == obs_cov(observed_nospec)
end

# shuffle variables
new_order = [3, 2, 7, 8, 5, 6, 9, 11, 1, 10, 4]

shuffle_names = Symbol.(names(dat))[new_order]

shuffle_dat_matrix = dat_matrix[:, new_order]

shuffle_dat_cov = Statistics.cov(shuffle_dat_matrix)

observed_shuffle = SemObservedCovariance(
    shuffle_dat_cov,
    specification = spec,
    obs_colnames = shuffle_names,
    n_obs = 75,
)

@testset "unit tests | SemObservedCovariance | input formats shuffled " begin
    @test obs_cov(observed)[new_order, new_order] ≈ obs_cov(observed_shuffle)
end

# with means -------------------------------------------------------------------------------

#@test_throws UndefKeywordError SemObservedCovariance(data = dat_matrix, meanstructure = true)

#@test_throws UndefKeywordError SemObservedCovariance(obs_cov = dat_cov, meanstructure = true)

# should work
observed = SemObservedCovariance(
    dat_cov,
    dat_mean,
    specification = spec,
    obs_colnames = Symbol.(names(dat)),
    n_obs = 75,
)

observed_nospec =
    SemObservedCovariance(dat_cov, dat_mean, specification = nothing, n_obs = 75)

@testset "unit tests | SemObservedCovariance | input formats - means" begin
    @test obs_mean(observed) == obs_mean(observed_nospec)
end

# shuffle variables
new_order = [3, 2, 7, 8, 5, 6, 9, 11, 1, 10, 4]

shuffle_names = Symbol.(names(dat))[new_order]

shuffle_dat = dat[:, new_order]

shuffle_dat_matrix = dat_matrix[:, new_order]

shuffle_dat_cov = Statistics.cov(shuffle_dat_matrix)
shuffle_dat_mean = vcat(Statistics.mean(shuffle_dat_matrix, dims = 1)...)

observed_shuffle = SemObservedCovariance(
    shuffle_dat_cov,
    shuffle_dat_mean,
    specification = spec,
    obs_colnames = shuffle_names,
    n_obs = 75,
)

@testset "unit tests | SemObservedCovariance | input formats shuffled - mean" begin
    @test obs_mean(observed)[new_order] == obs_mean(observed_shuffle)
end

############################################################################################
### tests - SemObservedMissing
############################################################################################

# errors

@test_throws MethodError SemObservedMissing(specification = spec)

# should work
observed =
    SemObservedMissing(dat_missing_matrix, specification = spec, obs_colnames = names(dat))

observed = SemObservedMissing(dat_missing, specification = spec)

observed_nospec = SemObservedMissing(dat_missing_matrix, specification = nothing)

observed_nospec2 = SemObservedMissing(dat_missing_matrix)

observed_matrix = SemObservedMissing(
    dat_missing_matrix,
    specification = spec,
    obs_colnames = Symbol.(names(dat)),
)

observed_matrix2 = SemObservedMissing(dat_missing_matrix, specification = spec)

@testset "unit tests | SemObservedMissing | input formats" begin
    @test isequal(get_data(observed), get_data(observed_nospec))
    @test isequal(get_data(observed_nospec2), get_data(observed_nospec))
    @test isequal(get_data(observed), get_data(observed_matrix))
    @test isequal(get_data(observed), get_data(observed_matrix2))
end

# shuffle variables
new_order = [3, 2, 7, 8, 5, 6, 9, 11, 1, 10, 4]

shuffle_names = Symbol.(names(dat))[new_order]

shuffle_dat_missing = dat_missing[:, new_order]

shuffle_dat_missing_matrix = dat_missing_matrix[:, new_order]

observed_shuffle = SemObservedMissing(shuffle_dat_missing, specification = spec)

observed_matrix_shuffle = SemObservedMissing(
    shuffle_dat_missing_matrix,
    specification = spec,
    obs_colnames = shuffle_names,
)

@testset "unit tests | SemObservedMissing | input formats shuffled " begin
    @test isequal(get_data(observed), get_data(observed_shuffle))
    @test isequal(get_data(observed)[:, new_order], get_data(observed_matrix_shuffle))
end
