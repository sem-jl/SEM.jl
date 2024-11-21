############################################################################################
# ML estimation
############################################################################################

model_g1 = Sem(specification = specification_g1, data = dat_g1, imply = RAMSymbolic)

model_g2 = Sem(specification = specification_g2, data = dat_g2, imply = RAM)

# test the different constructors
model_ml_multigroup = SemEnsemble(model_g1, model_g2; optimizer = semoptimizer)
model_ml_multigroup2 = SemEnsemble(
    specification = partable,
    data = dat,
    column = :school,
    groups = [:Pasteur, :Grant_White],
    loss = SemML
)


# gradients
@testset "ml_gradients_multigroup" begin
    test_gradient(model_ml_multigroup, start_test; atol = 1e-9)
    test_gradient(model_ml_multigroup2, start_test; atol = 1e-9)
end

# fit
@testset "ml_solution_multigroup" begin
    solution = sem_fit(model_ml_multigroup)
    update_estimate!(partable, solution)
    test_estimates(
        partable,
        solution_lav[:parameter_estimates_ml];
        atol = 1e-4,
        lav_groups = Dict(:Pasteur => 1, :Grant_White => 2),
    )
    solution = sem_fit(model_ml_multigroup2)
    update_estimate!(partable, solution)
    test_estimates(
        partable,
        solution_lav[:parameter_estimates_ml];
        atol = 1e-4,
        lav_groups = Dict(:Pasteur => 1, :Grant_White => 2),
    )
end

@testset "fitmeasures/se_ml" begin
    solution_ml = sem_fit(model_ml_multigroup)
    test_fitmeasures(
        fit_measures(solution_ml),
        solution_lav[:fitmeasures_ml];
        rtol = 1e-2,
        atol = 1e-7,
    )
    update_se_hessian!(partable, solution_ml)
    test_estimates(
        partable,
        solution_lav[:parameter_estimates_ml];
        atol = 1e-3,
        col = :se,
        lav_col = :se,
        lav_groups = Dict(:Pasteur => 1, :Grant_White => 2),
    )

    solution_ml = sem_fit(model_ml_multigroup2)
    test_fitmeasures(
        fit_measures(solution_ml),
        solution_lav[:fitmeasures_ml];
        rtol = 1e-2,
        atol = 1e-7,
    )
    update_se_hessian!(partable, solution_ml)
    test_estimates(
        partable,
        solution_lav[:parameter_estimates_ml];
        atol = 1e-3,
        col = :se,
        lav_col = :se,
        lav_groups = Dict(:Pasteur => 1, :Grant_White => 2),
    )
end

############################################################################################
# ML estimation - sorted
############################################################################################

partable_s = sort_vars(partable)

specification_s = convert(Dict{Symbol, RAMMatrices}, partable_s)

specification_g1_s = specification_s[:Pasteur]
specification_g2_s = specification_s[:Grant_White]

model_g1 = Sem(specification = specification_g1_s, data = dat_g1, imply = RAMSymbolic)

model_g2 = Sem(specification = specification_g2_s, data = dat_g2, imply = RAM)

model_ml_multigroup = SemEnsemble(model_g1, model_g2; optimizer = semoptimizer)

# gradients
@testset "ml_gradients_multigroup | sorted" begin
    test_gradient(model_ml_multigroup, start_test; atol = 1e-2)
end

grad = similar(start_test)
gradient!(grad, model_ml_multigroup, rand(36))
grad_fd = FiniteDiff.finite_difference_gradient(
    Base.Fix1(SEM.objective, model_ml_multigroup),
    start_test,
)

# fit
@testset "ml_solution_multigroup | sorted" begin
    solution = sem_fit(model_ml_multigroup)
    update_estimate!(partable_s, solution)
    test_estimates(
        partable_s,
        solution_lav[:parameter_estimates_ml];
        atol = 1e-4,
        lav_groups = Dict(:Pasteur => 1, :Grant_White => 2),
    )
end

@testset "fitmeasures/se_ml | sorted" begin
    solution_ml = sem_fit(model_ml_multigroup)
    test_fitmeasures(
        fit_measures(solution_ml),
        solution_lav[:fitmeasures_ml];
        rtol = 1e-2,
        atol = 1e-7,
    )

    update_se_hessian!(partable_s, solution_ml)
    test_estimates(
        partable_s,
        solution_lav[:parameter_estimates_ml];
        atol = 1e-3,
        col = :se,
        lav_col = :se,
        lav_groups = Dict(:Pasteur => 1, :Grant_White => 2),
    )
end

@testset "sorted | LowerTriangular A" begin
    @test imply(model_ml_multigroup.sems[2]).A isa LowerTriangular
end

############################################################################################
# ML estimation - user defined loss function
############################################################################################

struct UserSemML <: SemLossFunction
    hessianeval::ExactHessian

    UserSemML() = new(ExactHessian())
end

############################################################################################
### functors
############################################################################################

using LinearAlgebra: isposdef, logdet, tr, inv

function SEM.objective(ml::UserSemML, model::AbstractSem, params)
    Σ = imply(model).Σ
    Σₒ = SEM.obs_cov(observed(model))
    if !isposdef(Σ)
        return Inf
    else
        return logdet(Σ) + tr(inv(Σ) * Σₒ)
    end
end

# models
model_g1 = Sem(specification = specification_g1, data = dat_g1, imply = RAMSymbolic)

model_g2 = SemFiniteDiff(
    specification = specification_g2,
    data = dat_g2,
    imply = RAMSymbolic,
    loss = UserSemML(),
)

model_ml_multigroup = SemEnsemble(model_g1, model_g2; optimizer = semoptimizer)

@testset "gradients_user_defined_loss" begin
    test_gradient(model_ml_multigroup, start_test; atol = 1e-9)
end

# fit
@testset "solution_user_defined_loss" begin
    solution = sem_fit(model_ml_multigroup)
    update_estimate!(partable, solution)
    test_estimates(
        partable,
        solution_lav[:parameter_estimates_ml];
        atol = 1e-4,
        lav_groups = Dict(:Pasteur => 1, :Grant_White => 2),
    )
end

############################################################################################
# GLS estimation
############################################################################################

model_ls_g1 =
    Sem(specification = specification_g1, data = dat_g1, imply = RAMSymbolic, loss = SemWLS)

model_ls_g2 =
    Sem(specification = specification_g2, data = dat_g2, imply = RAMSymbolic, loss = SemWLS)

model_ls_multigroup = SemEnsemble(model_ls_g1, model_ls_g2; optimizer = semoptimizer)

@testset "ls_gradients_multigroup" begin
    test_gradient(model_ls_multigroup, start_test; atol = 1e-9)
end

@testset "ls_solution_multigroup" begin
    solution = sem_fit(model_ls_multigroup)
    update_estimate!(partable, solution)
    test_estimates(
        partable,
        solution_lav[:parameter_estimates_ls];
        atol = 1e-4,
        lav_groups = Dict(:Pasteur => 1, :Grant_White => 2),
    )
end

@testset "fitmeasures/se_ls" begin
    solution_ls = sem_fit(model_ls_multigroup)
    test_fitmeasures(
        fit_measures(solution_ls),
        solution_lav[:fitmeasures_ls];
        fitmeasure_names = fitmeasure_names_ls,
        rtol = 1e-2,
        atol = 1e-5,
    )

    update_se_hessian!(partable, solution_ls)
    test_estimates(
        partable,
        solution_lav[:parameter_estimates_ls];
        atol = 1e-2,
        col = :se,
        lav_col = :se,
        lav_groups = Dict(:Pasteur => 1, :Grant_White => 2),
    )
end

############################################################################################
# FIML estimation
############################################################################################

if !isnothing(specification_miss_g1)
    model_g1 = Sem(
        specification = specification_miss_g1,
        observed = SemObservedMissing,
        loss = SemFIML,
        data = dat_miss_g1,
        imply = RAM,
        optimizer = SemOptimizerEmpty(),
        meanstructure = true,
    )

    model_g2 = Sem(
        specification = specification_miss_g2,
        observed = SemObservedMissing,
        loss = SemFIML,
        data = dat_miss_g2,
        imply = RAM,
        optimizer = SemOptimizerEmpty(),
        meanstructure = true,
    )

    model_ml_multigroup = SemEnsemble(model_g1, model_g2; optimizer = semoptimizer)
    model_ml_multigroup2 = SemEnsemble(
        specification = partable_miss,
        data = dat_missing,
        column = :school,
        groups = [:Pasteur, :Grant_White],
        loss = SemFIML,
        observed = SemObservedMissing,
        meanstructure = true
    )

    ############################################################################################
    ### test gradients
    ############################################################################################

    start_test = [
        fill(0.5, 6)
        fill(1.0, 9)
        0.05
        0.01
        0.01
        0.05
        0.01
        0.05
        fill(0.01, 9)
        fill(1.0, 9)
        0.05
        0.01
        0.01
        0.05
        0.01
        0.05
        fill(0.01, 9)
    ]

    @testset "fiml_gradients_multigroup" begin
        test_gradient(model_ml_multigroup, start_test; atol = 1e-7)
        test_gradient(model_ml_multigroup2, start_test; atol = 1e-7)
    end

    @testset "fiml_solution_multigroup" begin
        solution = sem_fit(model_ml_multigroup)
        update_estimate!(partable_miss, solution)
        test_estimates(
            partable_miss,
            solution_lav[:parameter_estimates_fiml];
            atol = 1e-4,
            lav_groups = Dict(:Pasteur => 1, :Grant_White => 2),
        )
        solution = sem_fit(model_ml_multigroup2)
        update_estimate!(partable_miss, solution)
        test_estimates(
            partable_miss,
            solution_lav[:parameter_estimates_fiml];
            atol = 1e-4,
            lav_groups = Dict(:Pasteur => 1, :Grant_White => 2),
        )
    end

    @testset "fitmeasures/se_fiml" begin
        solution = sem_fit(model_ml_multigroup)
        test_fitmeasures(
            fit_measures(solution),
            solution_lav[:fitmeasures_fiml];
            rtol = 1e-3,
            atol = 0,
        )
        update_se_hessian!(partable_miss, solution)
        test_estimates(
            partable_miss,
            solution_lav[:parameter_estimates_fiml];
            atol = 1e-3,
            col = :se,
            lav_col = :se,
            lav_groups = Dict(:Pasteur => 1, :Grant_White => 2),
        )

        solution = sem_fit(model_ml_multigroup2)
        test_fitmeasures(
            fit_measures(solution),
            solution_lav[:fitmeasures_fiml];
            rtol = 1e-3,
            atol = 0,
        )
        update_se_hessian!(partable_miss, solution)
        test_estimates(
            partable_miss,
            solution_lav[:parameter_estimates_fiml];
            atol = 1e-3,
            col = :se,
            lav_col = :se,
            lav_groups = Dict(:Pasteur => 1, :Grant_White => 2),
        )
    end
end
