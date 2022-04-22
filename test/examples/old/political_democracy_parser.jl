using StructuralEquationModels, CSV, DataFrames, SparseArrays, Symbolics, LineSearches, Optim, Test, FiniteDiff, LinearAlgebra,
    StenoGraphs
import StructuralEquationModels as SEM
include("test_helpers.jl")

############################################################################
### observed data
############################################################################

dat = DataFrame(CSV.File("examples/data/data_dem.csv"))
par_ml = DataFrame(CSV.File("examples/data/par_dem_ml.csv"))
par_ls = DataFrame(CSV.File("examples/data/par_dem_ls.csv"))

############################################################################
### define models
############################################################################

observed_vars = [:x1, :x2, :x3, :y1, :y2, :y3, :y4, :y5, :y6, :y7, :y8]
latent_vars = [:ind60, :dem60, :dem65]

graph = @StenoGraph begin
    # loadings
    ind60 → fixed(1)*x1 + x2 + x3
    dem60 → fixed(1)*y1 + y2 + y3 + y4
    dem65 → fixed(1)*y5 + y6 + y7 + y8
    # latent regressions
    label(:a)*dem60 ← ind60
    dem65 ← dem60
    dem65 ← ind60
    # variances
    _(observed_vars) ↔ _(observed_vars)
    _(latent_vars) ↔ _(latent_vars)
    # covariances
    y1 ↔ y5
    y2 ↔ y4 + y6
    y3 ↔ y7
    y8 ↔ y4 + y6
end

partable = ParameterTable(
    latent_vars = latent_vars,
    observed_vars = observed_vars,
    graph = graph)

sort!(partable)

# models
model_ml = Sem(
    specification = partable,
    data = dat,
    #algorithm = LBFGS(linesearch = LineSearches.BackTracking())
)

model_ml_weighted = Sem(
    specification = partable,
    data = dat,
    loss_weights = (n_obs(model_ml),)
)

model_ls_sym = Sem(
    specification = partable,
    data = dat,
    imply = RAMSymbolic,
    loss = SemWLS,
    start_val = start_simple
)

############################################################################
### test starting values
############################################################################

test_start_val = [fill(0.5, 8); fill(0.05, 3); fill(0.1, 3); fill(1.0, 11); fill(0.05, 6)]
start_val_fabin3 = start_val(model_ml)

############################################################################
### test parameter index retrieval
############################################################################

@testset "get_identifier_indices" begin
    pars = [:θ_1, :θ_7, :θ_21]
    @test get_identifier_indices(pars, model_ml) == get_identifier_indices(pars, partable)
    @test get_identifier_indices(pars, model_ml) == get_identifier_indices(pars, RAMMatrices(partable))
end

############################################################################
### test gradients
############################################################################

@testset "ml_gradients" begin
    @test test_gradient(model_ml, test_start_val)
end

@testset "ls_gradients" begin
    @test test_gradient(model_ls_sym, test_start_val)
end

@testset "ml_gradients_weighted" begin
    @test test_gradient(model_ml_weighted, test_start_val)
end

############################################################################
### test solution
############################################################################

@testset "ml_solution" begin
    solution_ml = sem_fit(model_ml)
    update_estimate!(partable, solution_ml)
    @test SEM.compare_estimates(par_ml, partable, 0.01)
end

solution_ml = sem_fit(model_ml)
bs = se_bootstrap(solution_ml; n_boot = 20)
se = se_hessian(solution_ml)

update_partable!(partable, solution_ml, se, :se)
update_partable!(partable, solution_ml, bs, :se_boot)

@testset "ls_solution" begin
    solution_ls = sem_fit(model_ls_sym)
    update_estimate!(partable, solution_ls)
    @test SEM.compare_estimates(par_ls, partable, 0.01)
end

@testset "ml_solution_weighted" begin
    solution_ml = sem_fit(model_ml)
    solution_ml_weighted = sem_fit(model_ml_weighted)
    @test isapprox(solution_ml.solution, solution_ml_weighted.solution, rtol = 1e-3)
    @test isapprox(n_obs(model_ml)*solution_ml.minimum, solution_ml_weighted.minimum, rtol = 1e-6)
end

############################################################################
### test sorting
############################################################################

sort!(partable)

model_ml_sorted = Sem(
    specification = partable,
    data = dat
)

@testset "graph sorting" begin
    @test model_ml_sorted.imply.I_A isa LowerTriangular
end

@testset "ml_solution_sorted" begin
    solution_ml_sorted = sem_fit(model_ml_sorted)
    update_estimate!(partable, solution_ml_sorted)
    @test SEM.compare_estimates(par_ml, partable, 0.01)
end