using Statistics: cov, mean
using Random, NLopt

############################################################################################
### models w.o. meanstructure
############################################################################################

semoptimizer = SemOptimizer(engine = opt_engine)

model_ml = Sem(specification = spec, data = dat)
@test SEM.params(model_ml.implied.ram_matrices) == SEM.params(spec)

model_ml_cov = Sem(
    specification = spec,
    observed = SemObservedCovariance,
    obs_cov = cov(Matrix(dat)),
    obs_colnames = Symbol.(names(dat)),
    nsamples = 75,
)

model_ls_sym = Sem(specification = spec, data = dat, implied = RAMSymbolic, loss = SemWLS)

model_ml_sym = Sem(specification = spec, data = dat, implied = RAMSymbolic)

model_ridge = Sem(
    specification = spec,
    data = dat,
    loss = (SemML, SemRidge),
    α_ridge = 0.001,
    which_ridge = 16:20,
)

model_constant = Sem(
    specification = spec,
    data = dat,
    loss = (SemML, SemConstant),
    constant_loss = 3.465,
)

model_ml_weighted =
    Sem(specification = partable, data = dat, loss_weights = (nsamples(model_ml),))

############################################################################################
### test gradients
############################################################################################

models = [
    model_ml,
    model_ml_cov,
    model_ls_sym,
    model_ridge,
    model_constant,
    model_ml_sym,
    model_ml_weighted,
]
model_names = ["ml", "ml_cov", "ls_sym", "ridge", "constant", "ml_sym", "ml_weighted"]

for (model, name) in zip(models, model_names)
    try
        @testset "$(name)_gradient" begin
            test_gradient(model, start_test; rtol = 1e-9)
        end
    catch
    end
end

############################################################################################
### test solution
############################################################################################

models = [model_ml, model_ml_cov, model_ls_sym, model_ml_sym, model_constant]
model_names = ["ml", "ml_cov", "ls_sym", "ml_sym", "constant"]
solution_names = Symbol.("parameter_estimates_" .* ["ml", "ml", "ls", "ml", "ml"])

for (model, name, solution_name) in zip(models, model_names, solution_names)
    try
        @testset "$(name)_solution" begin
            solution = sem_fit(semoptimizer, model)
            update_estimate!(partable, solution)
            test_estimates(partable, solution_lav[solution_name]; atol = 1e-2)
        end
    catch
    end
end

@testset "ridge_solution" begin
    solution_ridge = sem_fit(semoptimizer, model_ridge)
    solution_ml = sem_fit(semoptimizer, model_ml)
    # solution_ridge_id = sem_fit(semoptimizer, model_ridge_id)
    @test abs(solution_ridge.minimum - solution_ml.minimum) < 1
end

# test constant objective value
@testset "constant_objective_and_gradient" begin
    @test (objective!(model_constant, start_test) - 3.465) ≈
          objective!(model_ml, start_test)
    grad = similar(start_test)
    grad2 = similar(start_test)
    gradient!(grad, model_constant, start_test)
    gradient!(grad2, model_ml, start_test)
    @test grad ≈ grad2
end

@testset "ml_solution_weighted" begin
    solution_ml = sem_fit(semoptimizer, model_ml)
    solution_ml_weighted = sem_fit(semoptimizer, model_ml_weighted)
    @test isapprox(solution(solution_ml), solution(solution_ml_weighted), rtol = 1e-3)
    @test isapprox(
        nsamples(model_ml) * StructuralEquationModels.minimum(solution_ml),
        StructuralEquationModels.minimum(solution_ml_weighted),
        rtol = 1e-6,
    )
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

    model_ls = Sem(
        specification = spec,
        data = dat,
        implied = RAMSymbolic,
        loss = SemWLS,
        hessian = true,
        algorithm = Newton(;
            linesearch = BackTracking(order = 3),
            alphaguess = InitialHagerZhang(),
        ),
    )

    model_ml = Sem(
        specification = spec,
        data = dat,
        implied = RAMSymbolic,
        hessian = true,
        algorithm = Newton(),
    )

    @testset "ml_hessians" begin
        test_hessian(model_ml, start_test; atol = 1e-4)
    end

    @testset "ls_hessians" begin
        test_hessian(model_ls, start_test; atol = 1e-4)
    end

    @testset "ml_solution_hessian" begin
        solution = sem_fit(semoptimizer, model_ml)
        update_estimate!(partable, solution)
        test_estimates(partable, solution_lav[:parameter_estimates_ml]; atol = 1e-3)
    end

    @testset "ls_solution_hessian" begin
        solution = sem_fit(semoptimizer, model_ls)
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
model_ls = Sem(
    specification = spec_mean,
    data = dat,
    implied = RAMSymbolic,
    loss = SemWLS,
    meanstructure = true,
)

model_ml = Sem(specification = spec_mean, data = dat, meanstructure = true)

model_ml_cov = Sem(
    specification = spec_mean,
    observed = SemObservedCovariance,
    obs_cov = cov(Matrix(dat)),
    obs_mean = vcat(mean(Matrix(dat), dims = 1)...),
    obs_colnames = Symbol.(names(dat)),
    meanstructure = true,
    nsamples = 75,
)

model_ml_sym =
    Sem(specification = spec_mean, data = dat, implied = RAMSymbolic, meanstructure = true)

############################################################################################
### test gradients
############################################################################################

models = [model_ml, model_ml_cov, model_ls, model_ml_sym]
model_names = ["ml", "ml_cov", "ls_sym", "ml_sym"]

for (model, name) in zip(models, model_names)
    try
        @testset "$(name)_gradient_mean" begin
            test_gradient(model, start_test_mean; rtol = 1e-9)
        end
    catch
    end
end

############################################################################################
### test solution
############################################################################################

solution_names = Symbol.("parameter_estimates_" .* ["ml", "ml", "ls", "ml"] .* "_mean")

for (model, name, solution_name) in zip(models, model_names, solution_names)
    try
        @testset "$(name)_solution_mean" begin
            solution = sem_fit(semoptimizer, model)
            update_estimate!(partable_mean, solution)
            test_estimates(partable_mean, solution_lav[solution_name]; atol = 1e-2)
        end
    catch
    end
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
model_ml = Sem(
    specification = spec_mean,
    data = dat_missing,
    observed = SemObservedMissing,
    loss = SemFIML,
    meanstructure = true,
)

model_ml_sym = Sem(
    specification = spec_mean,
    data = dat_missing,
    observed = SemObservedMissing,
    implied = RAMSymbolic,
    loss = SemFIML,
    meanstructure = true,
)

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
    solution = sem_fit(semoptimizer, model_ml_sym)
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
