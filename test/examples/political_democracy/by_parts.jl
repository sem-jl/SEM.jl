############################################################################################
### models w.o. meanstructure
############################################################################################

semoptimizer = SemOptimizer(engine = opt_engine)

model_ml = Sem(specification = spec, data = dat)
@test SEM.params(model_ml) == SEM.params(spec)

model_ls_sym =
    Sem(specification = spec, data = dat, implied = RAMSymbolic, vech = true, loss = SemWLS)

model_ml_sym = Sem(specification = spec, data = dat, implied = RAMSymbolic)

model_ml_ridge = Sem(
    specification = spec,
    data = dat,
    loss = (SemML, SemRidge),
    α_ridge = 0.001,
    which_ridge = 16:20,
)

model_ml_const = Sem(
    specification = spec,
    data = dat,
    loss = (SemML, SemConstant),
    constant_loss = 3.465,
)

model_ml_weighted = Sem(SemML(SemObservedData(data = dat), RAM(spec)) => nsamples(model_ml))

############################################################################################
### test gradients
############################################################################################

models = Dict(
    "ml" => model_ml,
    "ls_sym" => model_ls_sym,
    "ml_ridge" => model_ml_ridge,
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

@testset "$(id)_solution" for id in ["ml", "ls_sym", "ml_sym", "ml_const"]
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
    @test nsamples(model_ml) * StructuralEquationModels.minimum(solution_ml) ≈
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
### test hessians
############################################################################################

if opt_engine == :Optim
    using Optim, LineSearches

    model_ls = Sem(
        data = dat,
        specification = spec,
        implied = RAMSymbolic,
        loss = SemWLS,
        vech = true,
        hessian = true,
    )

    model_ml = Sem(
        data = dat,
        specification = spec,
        implied = RAMSymbolic,
        loss = SemML,
        hessian = true,
    )

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
            atol = 1e-3,
            skip = true,
        )
    end
end

############################################################################################
### meanstructure
############################################################################################

# models
model_ls = Sem(
    data = dat,
    specification = spec_mean,
    implied = RAMSymbolic,
    loss = SemWLS,
    vech = true,
)

model_ml = Sem(data = dat, specification = spec_mean, implied = RAM, loss = SemML)

model_ml_cov = Sem(
    specification = spec,
    observed = SemObservedCovariance,
    obs_cov = cov(Matrix(dat)),
    observed_vars = Symbol.(names(dat)),
    nsamples = 75,
)

model_ml_sym =
    Sem(data = dat, specification = spec_mean, implied = RAMSymbolic, loss = SemML)

############################################################################################
### test gradients
############################################################################################

models = Dict("ml" => model_ml, "ls_sym" => model_ls, "ml_sym" => model_ml_sym)

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
        atol = 1e-3,
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
### fiml
############################################################################################

# models
model_ml = Sem(
    data = dat_missing,
    observed = SemObservedMissing,
    specification = spec_mean,
    implied = RAM,
    loss = SemFIML,
)

model_ml_sym = Sem(
    data = dat_missing,
    observed = SemObservedMissing,
    specification = spec_mean,
    implied = RAMSymbolic,
    loss = SemFIML,
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
