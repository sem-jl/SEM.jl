using StructuralEquationModels

obs_vars = Symbol.("x", 1:9)
lat_vars = [:visual, :textual, :speed]

graph = @StenoGraph begin
    # measurement model
    visual → fixed(1.0) * x1 + fixed(0.5) * x2 + fixed(0.6) * x3
    textual → fixed(1.0) * x4 + x5 + label(:a₁) * x6
    speed → fixed(1.0) * x7 + fixed(1.0) * x8 + label(:λ₉) * x9
    # variances and covariances
    _(obs_vars) ↔ _(obs_vars)
    _(lat_vars) ↔ _(lat_vars)
    visual ↔ textual + speed
    textual ↔ speed
end

ens_graph = @StenoGraph begin
    # measurement model
    visual → fixed(1.0, 1.0) * x1 + fixed(0.5, 0.5) * x2 + fixed(0.6, 0.8) * x3
    textual → fixed(1.0, 1.0) * x4 + x5 + label(:a₁, :a₂) * x6
    speed → fixed(1.0, 1.0) * x7 + fixed(1.0, NaN) * x8 + label(:λ₉, :λ₉) * x9
    # variances and covariances
    _(obs_vars) ↔ _(obs_vars)
    _(lat_vars) ↔ _(lat_vars)
    visual ↔ textual + speed
    textual ↔ speed
end

fixed_and_labeled_graph = @StenoGraph begin
    # measurement model
    visual → fixed(1.0) * label(:λ) * x1
end

@testset "ParameterTable" begin
    @testset "from StenoGraph" begin
        @test_throws UndefKeywordError(:observed_vars) ParameterTable(graph)
        @test_throws UndefKeywordError(:latent_vars) ParameterTable(
            graph,
            observed_vars = obs_vars,
        )
        @test_throws ArgumentError("It is not allowed to label fixed parameters.") ParameterTable(
            fixed_and_labeled_graph,
            observed_vars = obs_vars,
            latent_vars = lat_vars,
        )
        partable = @inferred(
            ParameterTable(graph, observed_vars = obs_vars, latent_vars = lat_vars)
        )

        @test partable isa ParameterTable

        # vars API
        @test observed_vars(partable) == obs_vars
        @test nobserved_vars(partable) == length(obs_vars)
        @test latent_vars(partable) == lat_vars
        @test nlatent_vars(partable) == length(lat_vars)
        @test nvars(partable) == length(obs_vars) + length(lat_vars)
        @test issetequal(vars(partable), [obs_vars; lat_vars])

        # params API
        @test params(partable) == [[:θ_1, :a₁, :λ₉]; Symbol.("θ_", 2:16)]
        @test nparams(partable) == 18

        # don't allow constructing ParameterTable from a graph for an ensemble
        @test_throws ArgumentError ParameterTable(
            ens_graph,
            observed_vars = obs_vars,
            latent_vars = lat_vars,
        )
    end

    @testset "from RAMMatrices" begin
        partable_orig =
            ParameterTable(graph, observed_vars = obs_vars, latent_vars = lat_vars)
        ram_matrices = RAMMatrices(partable_orig)

        partable = @inferred(ParameterTable(ram_matrices))
        @test partable isa ParameterTable
        @test issetequal(keys(partable.columns), keys(partable_orig.columns))
        # FIXME nrow()?
        @test length(partable.columns[:from]) == length(partable_orig.columns[:from])
        @test partable == partable_orig broken = true
    end
end

@testset "EnsembleParameterTable" begin
    groups = [:Pasteur, :Grant_White],
    @test_throws UndefKeywordError(:observed_vars) EnsembleParameterTable(ens_graph)
    @test_throws UndefKeywordError(:latent_vars) EnsembleParameterTable(
        ens_graph,
        observed_vars = obs_vars,
    )
    @test_throws UndefKeywordError(:groups) EnsembleParameterTable(
        ens_graph,
        observed_vars = obs_vars,
        latent_vars = lat_vars,
    )

    enspartable = @inferred(
        EnsembleParameterTable(
            ens_graph,
            observed_vars = obs_vars,
            latent_vars = lat_vars,
            groups = [:Pasteur, :Grant_White],
        )
    )
    @test enspartable isa EnsembleParameterTable

    @test nobserved_vars(enspartable) == length(obs_vars) broken = true
    @test observed_vars(enspartable) == obs_vars broken = true
    @test nlatent_vars(enspartable) == length(lat_vars) broken = true
    @test latent_vars(enspartable) == lat_vars broken = true
    @test nvars(enspartable) == length(obs_vars) + length(lat_vars) broken = true
    @test issetequal(vars(enspartable), [obs_vars; lat_vars]) broken = true

    @test nparams(enspartable) == 36
    @test issetequal(
        params(enspartable),
        [Symbol.("gPasteur_", 1:16); Symbol.("gGrant_White_", 1:17); [:a₁, :a₂, :λ₉]],
    )
end

@testset "RAMMatrices" begin
    partable = ParameterTable(graph, observed_vars = obs_vars, latent_vars = lat_vars)

    ram_matrices = @inferred(RAMMatrices(partable))
    @test ram_matrices isa RAMMatrices

    # vars API
    @test nobserved_vars(ram_matrices) == length(obs_vars)
    @test observed_vars(ram_matrices) == obs_vars
    @test nlatent_vars(ram_matrices) == length(lat_vars)
    @test latent_vars(ram_matrices) == lat_vars
    @test nvars(ram_matrices) == length(obs_vars) + length(lat_vars)
    @test issetequal(vars(ram_matrices), [obs_vars; lat_vars])

    # params API
    @test nparams(ram_matrices) == nparams(partable)
    @test params(ram_matrices) == params(partable)
end
