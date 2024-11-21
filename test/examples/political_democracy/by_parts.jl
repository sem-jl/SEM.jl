############################################################################################
### models w.o. meanstructure
############################################################################################

# observed ---------------------------------------------------------------------------------
observed = SemObservedData(specification = spec, data = dat)

# imply
imply_ram = RAM(specification = spec)

imply_ram_sym = RAMSymbolic(specification = spec)

# loss functions ---------------------------------------------------------------------------
ml = SemML(observed = observed)

wls = SemWLS(observed = observed)

ridge = SemRidge(α_ridge = 0.001, which_ridge = 16:20, nparams = 31)

constant = SemConstant(constant_loss = 3.465)

# loss -------------------------------------------------------------------------------------
loss_ml = SemLoss(ml)

loss_wls = SemLoss(wls)

# optimizer -------------------------------------------------------------------------------------
optimizer_obj = semoptimizer()

# models -----------------------------------------------------------------------------------

model_ml = Sem(observed, imply_ram, loss_ml, optimizer_obj)

model_ls_sym =
    Sem(observed, RAMSymbolic(specification = spec, vech = true), loss_wls, optimizer_obj)

model_ml_sym = Sem(observed, imply_ram_sym, loss_ml, optimizer_obj)

model_ridge = Sem(observed, imply_ram, SemLoss(ml, ridge), optimizer_obj)

model_constant = Sem(observed, imply_ram, SemLoss(ml, constant), optimizer_obj)

model_ml_weighted = Sem(
    observed,
    imply_ram,
    SemLoss(ml; loss_weights = [nsamples(model_ml)]),
    optimizer_obj,
)

############################################################################################
### test gradients
############################################################################################

models =
    [model_ml, model_ls_sym, model_ridge, model_constant, model_ml_sym, model_ml_weighted]
model_names = ["ml", "ls_sym", "ridge", "constant", "ml_sym", "ml_weighted"]

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

models = [model_ml, model_ls_sym, model_ml_sym, model_constant]
model_names = ["ml", "ls_sym", "ml_sym", "constant"]
solution_names = Symbol.("parameter_estimates_" .* ["ml", "ls", "ml", "ml"])

for (model, name, solution_name) in zip(models, model_names, solution_names)
    try
        @testset "$(name)_solution" begin
            solution = sem_fit(model)
            update_estimate!(partable, solution)
            test_estimates(partable, solution_lav[solution_name]; atol = 1e-2)
        end
    catch
    end
end

@testset "ridge_solution" begin
    solution_ridge = sem_fit(model_ridge)
    solution_ml = sem_fit(model_ml)
    # solution_ridge_id = sem_fit(model_ridge_id)
    @test solution_ridge.minimum < solution_ml.minimum + 1
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
    solution_ml = sem_fit(model_ml)
    solution_ml_weighted = sem_fit(model_ml_weighted)
    @test solution(solution_ml) ≈ solution(solution_ml_weighted) rtol = 1e-3
    @test nsamples(model_ml) * StructuralEquationModels.minimum(solution_ml) ≈
          StructuralEquationModels.minimum(solution_ml_weighted) rtol = 1e-6
end

############################################################################################
### test fit assessment
############################################################################################

@testset "fitmeasures/se_ml" begin
    solution_ml = sem_fit(model_ml)
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
    solution_ls = sem_fit(model_ls_sym)
    fm = fit_measures(solution_ls)
    test_fitmeasures(
        fm,
        solution_lav[:fitmeasures_ls];
        atol = 1e-3,
        fitmeasure_names = fitmeasure_names_ls,
    )
    @test (fm[:AIC] === missing) & (fm[:BIC] === missing) & (fm[:minus2ll] === missing)

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

if semoptimizer == SemOptimizerOptim
    using Optim, LineSearches

    optimizer_obj = SemOptimizerOptim(
        algorithm = Newton(;
            linesearch = BackTracking(order = 3),
            alphaguess = InitialHagerZhang(),
        ),
    )

    imply_sym_hessian_vech = RAMSymbolic(specification = spec, vech = true, hessian = true)

    imply_sym_hessian = RAMSymbolic(specification = spec, hessian = true)

    model_ls = Sem(observed, imply_sym_hessian_vech, loss_wls, optimizer_obj)

    model_ml =
        Sem(observed, imply_sym_hessian, loss_ml, SemOptimizerOptim(algorithm = Newton()))

    @testset "ml_hessians" begin
        test_hessian(model_ml, start_test; atol = 1e-4)
    end

    @testset "ls_hessians" begin
        test_hessian(model_ls, start_test; atol = 1e-4)
    end

    @testset "ml_solution_hessian" begin
        solution = sem_fit(model_ml)
        update_estimate!(partable, solution)
        test_estimates(partable, solution_lav[:parameter_estimates_ml]; atol = 1e-3)
    end

    @testset "ls_solution_hessian" begin
        solution = sem_fit(model_ls)
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

# observed ---------------------------------------------------------------------------------
observed = SemObservedData(specification = spec_mean, data = dat, meanstructure = true)

# imply
imply_ram = RAM(specification = spec_mean, meanstructure = true)

imply_ram_sym = RAMSymbolic(specification = spec_mean, meanstructure = true)

# loss functions ---------------------------------------------------------------------------
ml = SemML(observed = observed, meanstructure = true)

wls = SemWLS(observed = observed, meanstructure = true)

# loss -------------------------------------------------------------------------------------
loss_ml = SemLoss(ml)

loss_wls = SemLoss(wls)

# optimizer -------------------------------------------------------------------------------------
optimizer_obj = semoptimizer()

# models -----------------------------------------------------------------------------------
model_ml = Sem(observed, imply_ram, loss_ml, optimizer_obj)

model_ls = Sem(
    observed,
    RAMSymbolic(specification = spec_mean, meanstructure = true, vech = true),
    loss_wls,
    optimizer_obj,
)

model_ml_sym = Sem(observed, imply_ram_sym, loss_ml, optimizer_obj)

############################################################################################
### test gradients
############################################################################################

models = [model_ml, model_ls, model_ml_sym]
model_names = ["ml", "ls_sym", "ml_sym"]

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

solution_names = Symbol.("parameter_estimates_" .* ["ml", "ls", "ml"] .* "_mean")

for (model, name, solution_name) in zip(models, model_names, solution_names)
    try
        @testset "$(name)_solution_mean" begin
            solution = sem_fit(model)
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
    solution_ml = sem_fit(model_ml)
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
    solution_ls = sem_fit(model_ls)
    fm = fit_measures(solution_ls)
    test_fitmeasures(
        fm,
        solution_lav[:fitmeasures_ls_mean];
        atol = 1e-3,
        fitmeasure_names = fitmeasure_names_ls,
    )
    @test (fm[:AIC] === missing) & (fm[:BIC] === missing) & (fm[:minus2ll] === missing)

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

observed = SemObservedMissing(specification = spec_mean, data = dat_missing)

fiml = SemFIML(observed = observed, specification = spec_mean)

loss_fiml = SemLoss(fiml)

model_ml = Sem(observed, imply_ram, loss_fiml, optimizer_obj)

model_ml_sym = Sem(observed, imply_ram_sym, loss_fiml, optimizer_obj)

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
    solution = sem_fit(model_ml)
    update_estimate!(partable_mean, solution)
    test_estimates(partable_mean, solution_lav[:parameter_estimates_fiml]; atol = 1e-2)
end

@testset "fiml_solution_symbolic" begin
    solution = sem_fit(model_ml_sym)
    update_estimate!(partable_mean, solution)
    test_estimates(partable_mean, solution_lav[:parameter_estimates_fiml]; atol = 1e-2)
end

############################################################################################
### test fit measures
############################################################################################

@testset "fitmeasures/se_fiml" begin
    solution_ml = sem_fit(model_ml)
    test_fitmeasures(
        fit_measures(solution_ml),
        solution_lav[:fitmeasures_fiml];
        atol = 1e-3,
    )

    update_se_hessian!(partable_mean, solution_ml)
    test_estimates(
        partable_mean,
        solution_lav[:parameter_estimates_fiml];
        atol = 1e-3,
        col = :se,
        lav_col = :se,
    )
end
