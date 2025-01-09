using StructuralEquationModels, Test, ProximalAlgorithms, ProximalOperators

# load data
dat = example_data("political_democracy")

############################################################################
### define models
############################################################################

observed_vars = [:x1, :x2, :x3, :y1, :y2, :y3, :y4, :y5, :y6, :y7, :y8]
latent_vars = [:ind60, :dem60, :dem65]

graph = @StenoGraph begin
    ind60 → fixed(1) * x1 + x2 + x3
    dem60 → fixed(1) * y1 + y2 + y3 + y4
    dem65 → fixed(1) * y5 + y6 + y7 + y8

    dem60 ← ind60
    dem65 ← dem60
    dem65 ← ind60

    _(observed_vars) ↔ _(observed_vars)
    _(latent_vars) ↔ _(latent_vars)

    y1 ↔ label(:cov_15) * y5
    y2 ↔ label(:cov_24) * y4 + label(:cov_26) * y6
    y3 ↔ label(:cov_37) * y7
    y4 ↔ label(:cov_48) * y8
    y6 ↔ label(:cov_68) * y8
end

partable = ParameterTable(graph, latent_vars = latent_vars, observed_vars = observed_vars)

ram_mat = RAMMatrices(partable)

model = Sem(specification = partable, data = dat, loss = SemML)

fit = sem_fit(model)

# use lasso from ProximalSEM
λ = zeros(31)

model_prox = Sem(specification = partable, data = dat, loss = SemML)

fit_prox = sem_fit(model_prox, engine = :Proximal, operator_g = NormL1(λ))

@testset "lasso | solution_unregularized" begin
    @test fit_prox.optimization_result.result[:iterations] < 1000
    @test maximum(abs.(solution(fit) - solution(fit_prox))) < 0.002
end

λ = zeros(31);
λ[16:20] .= 0.02;

model_prox = Sem(specification = partable, data = dat, loss = SemML)

fit_prox = sem_fit(model_prox, engine = :Proximal, operator_g = NormL1(λ))

@testset "lasso | solution_regularized" begin
    @test fit_prox.optimization_result.result[:iterations] < 1000
    @test all(solution(fit_prox)[16:20] .< solution(fit)[16:20])
    @test StructuralEquationModels.minimum(fit_prox) -
          StructuralEquationModels.minimum(fit) < 0.03
end
