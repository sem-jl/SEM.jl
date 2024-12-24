using Statistics: cov, mean
using Random, NLopt

############################################################################################
### models w.o. meanstructure
############################################################################################

semoptimizer = SemOptimizer(engine = opt_engine)

model_ml = Sem(SemML(SemObservedData(dat), RAM(spec)))
@test SEM.params(model_ml) == SEM.params(spec)
@test SEM.nloss_terms(model_ml) == 1
@test SEM.sem_terms(model_ml) isa Tuple{SEM.LossTerm{<:SemML, Nothing, Nothing}}

model_ml_cov = Sem(
    SemML(
        SemObservedCovariance(
            cov(Matrix(dat)),
            obs_colnames = Symbol.(names(dat)),
            n_obs = 75,
        ),
        RAM(spec),
    ),
)

model_ls_sym = Sem(SemWLS(SemObservedData(dat), RAMSymbolic(spec, vech = true)))

model_ml_sym = Sem(SemML(SemObservedData(dat), RAMSymbolic(spec)))

model_ml_ridge = Sem(SemML(SemObservedData(dat), RAM(spec)), SemRidge(spec, 16:20) => 0.001)

@test SEM.nloss_terms(model_ml_ridge) == 2

model_ml_const = Sem(SemML(SemObservedData(dat), RAM(spec)), SemConstant(3.465))

model_ml_weighted = Sem(SemML(SemObservedData(dat), RAM(partable)) => size(dat, 1))

############################################################################################
### test gradients
############################################################################################

models = Dict(
    "ml" => model_ml,
    "ml_cov" => model_ml_cov,
    "ls_sym" => model_ls_sym,
    "ridge" => model_ml_ridge,
    "ml_const" => model_ml_const,
    "ml_sym" => model_ml_sym,
    "ml_weighted" => model_ml_weighted,
)

@testset "$(id)_gradient" for (id, model) in pairs(models)
    test_gradient(model, start_test; rtol = 1e-9)
end

############################################################################################
### test solution
############################################################################################

@testset "$(id)_solution" for id in ["ml", "ml_cov", "ls_sym", "ml_sym", "ml_const"]
    model = models[id]
    solution = sem_fit(semoptimizer, model)
    sol_name = Symbol("parameter_estimates_", replace(id, r"_.+$" => ""))
    update_estimate!(partable, solution)
    test_estimates(partable, solution_lav[sol_name]; atol = 1e-2)
end

@testset "ridge_solution" begin
    solution_ridge = sem_fit(semoptimizer, model_ml_ridge)
    solution_ml = sem_fit(semoptimizer, model_ml)
    # solution_ridge_id = sem_fit(model_ridge_id)
    @test abs(solution_ridge.minimum - solution_ml.minimum) < 1
end

# test constant objective value
@testset "constant_objective_and_gradient" begin
    @test (objective!(model_ml_const, start_test) - 3.465) ≈
          objective!(model_ml, start_test)
    grad = similar(start_test)
    grad2 = similar(start_test)
    gradient!(grad, model_ml_const, start_test)
    gradient!(grad2, model_ml, start_test)
    @test grad ≈ grad2
end

@testset "ml_solution_weighted" begin
    solution_ml = sem_fit(semoptimizer, model_ml)
    solution_ml_weighted = sem_fit(semoptimizer, model_ml_weighted)
    @test solution(solution_ml) ≈ solution(solution_ml_weighted) rtol = 1e-3
    @test n_obs(model_ml) * StructuralEquationModels.minimum(solution_ml) ≈
          StructuralEquationModels.minimum(solution_ml_weighted) rtol = 1e-6
end

############################################################################################
### test fit assessment
############################################################################################

@testset "fitmeasures/se_ml" begin
    solution_ml = sem_fit(semoptimizer, model_ml)
    test_fitmeasures(fit_measures(solution_ml), solution_lav[:fitmeasures_ml]; atol = 1e-3)

    update_se_hessian!(partable, solution_ml)
    test_estimates(
        partable,
        solution_lav[:parameter_estimates_ml];
        atol = 1e-3,
        col = :se,
        lav_col = :se,
    )
end

@testset "fitmeasures/se_ls" begin
    solution_ls = sem_fit(semoptimizer, model_ls_sym)
    fm = fit_measures(solution_ls)
    test_fitmeasures(
        fm,
        solution_lav[:fitmeasures_ls];
        atol = 1e-3,
        fitmeasure_names = fitmeasure_names_ls,
    )
    @test ismissing(fm[:AIC]) && ismissing(fm[:BIC]) && ismissing(fm[:minus2ll])

    update_se_hessian!(partable, solution_ls)
    test_estimates(
        partable,
        solution_lav[:parameter_estimates_ls];
        atol = 1e-2,
        col = :se,
        lav_col = :se,
    )
end

############################################################################################
### data simulation
############################################################################################

@testset "data_simulation_wo_mean" begin
    # parameters to recover
    params = start_simple(
        model_ml;
        start_loadings = 0.5,
        start_regressions = 0.5,
        start_variances_observed = 0.5,
        start_variances_latent = 1.0,
        start_covariances_observed = 0.2,
    )
    # set seed for simulation
    Random.seed!(83472834)
    colnames = Symbol.(names(example_data("political_democracy")))
    # simulate data
    model_ml_new = swap_observed(
        model_ml,
        data = rand(model_ml, params, 1_000_000),
        specification = spec,
        obs_colnames = colnames,
    )
    model_ml_sym_new = swap_observed(
        model_ml_sym,
        data = rand(model_ml_sym, params, 1_000_000),
        specification = spec,
        obs_colnames = colnames,
    )
    # fit models
    sol_ml = solution(sem_fit(semoptimizer, model_ml_new))
    sol_ml_sym = solution(sem_fit(semoptimizer, model_ml_sym_new))
    # check solution
    @test maximum(abs.(sol_ml - params)) < 0.01
    @test maximum(abs.(sol_ml_sym - params)) < 0.01
end

############################################################################################
### test hessians
############################################################################################

if opt_engine == :Optim
    using Optim, LineSearches

    model_ls =
        Sem(SemWLS(SemObservedData(dat), RAMSymbolic(spec, vech = true, hessian = true)))

    model_ml = Sem(SemML(SemObservedData(dat), RAMSymbolic(spec, hessian = true)))

    @testset "ml_hessians" begin
        test_hessian(model_ml, start_test; atol = 1e-4)
    end

    @testset "ls_hessians" begin
        test_hessian(model_ls, start_test; atol = 1e-4)
    end

    @testset "ml_solution_hessian" begin
        solution = sem_fit(SemOptimizer(engine = :Optim, algorithm = Newton()), model_ml)

        update_estimate!(partable, solution)
        test_estimates(partable, solution_lav[:parameter_estimates_ml]; atol = 1e-3)
    end

    @testset "ls_solution_hessian" begin
        solution = sem_fit(
            SemOptimizer(
                engine = :Optim,
                algorithm = Newton(
                    linesearch = BackTracking(order = 3),
                    alphaguess = InitialHagerZhang(),
                ),
            ),
            model_ls,
        )
        update_estimate!(partable, solution)
        test_estimates(
            partable,
            solution_lav[:parameter_estimates_ls];
            atol = 0.002,
            rtol = 0.0,
            skip = true,
        )
    end
end

############################################################################################
### meanstructure
############################################################################################

# models
model_ls = Sem(SemWLS(SemObservedData(dat), RAMSymbolic(spec_mean, vech = true)))

model_ml = Sem(SemML(SemObservedData(dat), RAM(spec_mean)))

model_ml_cov = Sem(
    SemML(
        SemObservedCovariance(
            cov(Matrix(dat)),
            vec(mean(Matrix(dat), dims = 1)),
            obs_colnames = names(dat),
            n_obs = 75,
        ),
        RAM(spec_mean),
    ),
)

model_ml_sym = Sem(SemML(SemObservedData(dat), RAMSymbolic(spec_mean)))

############################################################################################
### test gradients
############################################################################################

models = Dict(
    "ml" => model_ml,
    "ml_cov" => model_ml_cov,
    "ls_sym" => model_ls,
    "ml_sym" => model_ml_sym,
)

@testset "$(id)_gradient_mean" for (id, model) in pairs(models)
    test_gradient(model, start_test_mean; rtol = 1e-9)
end

############################################################################################
### test solution
############################################################################################

@testset "$(id)_solution_mean" for (id, model) in pairs(models)
    solution = sem_fit(semoptimizer, model, start_val = start_test_mean)
    update_estimate!(partable_mean, solution)
    sol_name = Symbol("parameter_estimates_", replace(id, r"_.+$" => ""), "_mean")
    test_estimates(partable_mean, solution_lav[sol_name]; atol = 1e-2)
end

############################################################################################
### test fit assessment
############################################################################################

@testset "fitmeasures/se_ml_mean" begin
    solution_ml = sem_fit(semoptimizer, model_ml)
    test_fitmeasures(
        fit_measures(solution_ml),
        solution_lav[:fitmeasures_ml_mean];
        atol = 0.002,
    )

    update_se_hessian!(partable_mean, solution_ml)
    test_estimates(
        partable_mean,
        solution_lav[:parameter_estimates_ml_mean];
        atol = 0.002,
        col = :se,
        lav_col = :se,
    )
end

@testset "fitmeasures/se_ls_mean" begin
    solution_ls = sem_fit(semoptimizer, model_ls)
    fm = fit_measures(solution_ls)
    test_fitmeasures(
        fm,
        solution_lav[:fitmeasures_ls_mean];
        atol = 1e-3,
        fitmeasure_names = fitmeasure_names_ls,
    )
    @test ismissing(fm[:AIC]) && ismissing(fm[:BIC]) && ismissing(fm[:minus2ll])

    update_se_hessian!(partable_mean, solution_ls)
    test_estimates(
        partable_mean,
        solution_lav[:parameter_estimates_ls_mean];
        atol = 1e-2,
        col = :se,
        lav_col = :se,
    )
end

############################################################################################
### data simulation
############################################################################################

@testset "data_simulation_with_mean" begin
    # parameters to recover
    params = start_simple(
        model_ml;
        start_loadings = 0.5,
        start_regressions = 0.5,
        start_variances_observed = 0.5,
        start_variances_latent = 1.0,
        start_covariances_observed = 0.2,
        start_means = 0.5,
    )
    # set seed for simulation
    Random.seed!(83472834)
    colnames = Symbol.(names(example_data("political_democracy")))
    # simulate data
    model_ml_new = swap_observed(
        model_ml,
        data = rand(model_ml, params, 1_000_000),
        specification = spec,
        obs_colnames = colnames,
        meanstructure = true,
    )
    model_ml_sym_new = swap_observed(
        model_ml_sym,
        data = rand(model_ml_sym, params, 1_000_000),
        specification = spec,
        obs_colnames = colnames,
        meanstructure = true,
    )
    # fit models
    sol_ml = solution(sem_fit(semoptimizer, model_ml_new))
    sol_ml_sym = solution(sem_fit(semoptimizer, model_ml_sym_new))
    # check solution
    @test maximum(abs.(sol_ml - params)) < 0.01
    @test maximum(abs.(sol_ml_sym - params)) < 0.01
end

############################################################################################
### fiml
############################################################################################

# models
obs_missing = SemObservedMissing(dat_missing, obs_colnames = SEM.observed_vars(spec_mean))

model_ml = Sem(SemFIML(obs_missing, RAM(spec_mean)))

model_ml_sym = Sem(SemFIML(obs_missing, RAMSymbolic(spec_mean)))

############################################################################################
### test gradients
############################################################################################

@testset "fiml_gradient" begin
    test_gradient(model_ml, start_test_mean; atol = 1e-6)
end

@testset "fiml_gradient_symbolic" begin
    test_gradient(model_ml_sym, start_test_mean; atol = 1e-6)
end

############################################################################################
### test solution
############################################################################################

@testset "fiml_solution" begin
    solution = sem_fit(semoptimizer, model_ml)
    update_estimate!(partable_mean, solution)
    test_estimates(partable_mean, solution_lav[:parameter_estimates_fiml]; atol = 1e-2)
end

@testset "fiml_solution_symbolic" begin
    solution = sem_fit(semoptimizer, model_ml_sym, start_val = start_test_mean)
    update_estimate!(partable_mean, solution)
    test_estimates(partable_mean, solution_lav[:parameter_estimates_fiml]; atol = 1e-2)
end

############################################################################################
### test fit measures
############################################################################################

@testset "fitmeasures/se_fiml" begin
    solution_ml = sem_fit(semoptimizer, model_ml)
    test_fitmeasures(
        fit_measures(solution_ml),
        solution_lav[:fitmeasures_fiml];
        atol = 1e-3,
    )

    update_se_hessian!(partable_mean, solution_ml)
    test_estimates(
        partable_mean,
        solution_lav[:parameter_estimates_fiml];
        atol = 0.002,
        col = :se,
        lav_col = :se,
    )
end
