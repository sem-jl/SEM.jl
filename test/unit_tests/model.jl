using StructuralEquationModels, Test, Statistics

dat = example_data("political_democracy")
dat_missing = example_data("political_democracy_missing")[:, names(dat)]

obs_vars = [Symbol.("x", 1:3); Symbol.("y", 1:8)]
lat_vars = [:ind60, :dem60, :dem65]

graph = @StenoGraph begin
    # loadings
    ind60 → fixed(1) * x1 + x2 + x3
    dem60 → fixed(1) * y1 + y2 + y3 + y4
    dem65 → fixed(1) * y5 + y6 + y7 + y8
    # latent regressions
    label(:a) * dem60 ← ind60
    dem65 ← dem60
    dem65 ← ind60
    # variances
    _(obs_vars) ↔ _(obs_vars)
    _(lat_vars) ↔ _(lat_vars)
    # covariances
    y1 ↔ y5
    y2 ↔ y4 + y6
    y3 ↔ y7
    y8 ↔ y4 + y6
end

ram_matrices =
    RAMMatrices(ParameterTable(graph, observed_vars = obs_vars, latent_vars = lat_vars))

obs = SemObservedData(specification = ram_matrices, data = dat)

function test_vars_api(semobj, spec::SemSpecification)
    @test @inferred(nobserved_vars(semobj)) == nobserved_vars(spec)
    @test observed_vars(semobj) == observed_vars(spec)

    @test @inferred(nlatent_vars(semobj)) == nlatent_vars(spec)
    @test latent_vars(semobj) == latent_vars(spec)

    @test @inferred(nvars(semobj)) == nvars(spec)
    @test vars(semobj) == vars(spec)
end

function test_params_api(semobj, spec::SemSpecification)
    @test @inferred(nparams(semobj)) == nparams(spec)
    @test @inferred(params(semobj)) == params(spec)
end

@testset "Sem(imply=$implytype, loss=$losstype)" for implytype in (RAM, RAMSymbolic),
    losstype in (SemML, SemWLS)

    model = Sem(
        specification = ram_matrices,
        observed = obs,
        imply = implytype,
        loss = losstype,
    )

    @test model isa Sem
    @test @inferred(imply(model)) isa implytype
    @test @inferred(observed(model)) isa SemObserved
    @test @inferred(optimizer(model)) isa SemOptimizer

    test_vars_api(model, ram_matrices)
    test_params_api(model, ram_matrices)

    test_vars_api(imply(model), ram_matrices)
    test_params_api(imply(model), ram_matrices)

    @test @inferred(loss(model)) isa SemLoss
    semloss = loss(model).functions[1]
    @test semloss isa losstype

    @test @inferred(nsamples(model)) == nsamples(obs)
end
