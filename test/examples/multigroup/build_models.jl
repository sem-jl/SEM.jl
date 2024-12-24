const SEM = StructuralEquationModels

############################################################################################
# ML estimation
############################################################################################

obs_g1 = SemObservedData(data = dat_g1, observed_vars = SEM.observed_vars(specification_g1))
obs_g2 = SemObservedData(data = dat_g2, observed_vars = SEM.observed_vars(specification_g2))

model_ml_multigroup =
    Sem(SemML(obs_g1, RAMSymbolic(specification_g1)), SemML(obs_g2, RAM(specification_g2)))

@testset "Sem API" begin
    @test SEM.nsamples(model_ml_multigroup) == nsamples(obs_g1) + nsamples(obs_g2)
    @test SEM.nsem_terms(model_ml_multigroup) == 2
    @test length(SEM.sem_terms(model_ml_multigroup)) == 2
end

# gradients
@testset "ml_gradients_multigroup" begin
    test_gradient(model_ml_multigroup, start_test; atol = 1e-9)
end

# fit
@testset "ml_solution_multigroup" begin
    solution = sem_fit(semoptimizer, model_ml_multigroup)
    update_estimate!(partable, solution)
    test_estimates(
        partable,
        solution_lav[:parameter_estimates_ml];
        atol = 1e-4,
        lav_groups = Dict(:Pasteur => 1, :Grant_White => 2),
    )
end

@testset "fitmeasures/se_ml" begin
    solution_ml = sem_fit(semoptimizer, model_ml_multigroup)
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
obs_g1_s = SemObservedData(
    data = dat_g1,
    observed_vars = SEM.observed_vars(specification_s[:Pasteur]),
)
obs_g2_s = SemObservedData(
    data = dat_g2,
    observed_vars = SEM.observed_vars(specification_s[:Grant_White]),
)

model_ml_multigroup = Sem(
    SemML(obs_g1_s, RAMSymbolic(specification_s[:Pasteur])),
    SemML(obs_g2_s, RAM(specification_s[:Grant_White])),
)

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
    solution = sem_fit(semoptimizer, model_ml_multigroup)
    update_estimate!(partable_s, solution)
    test_estimates(
        partable_s,
        solution_lav[:parameter_estimates_ml];
        atol = 1e-4,
        lav_groups = Dict(:Pasteur => 1, :Grant_White => 2),
    )
end

@testset "fitmeasures/se_ml | sorted" begin
    solution_ml = sem_fit(semoptimizer, model_ml_multigroup)
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
    @test implied(SEM.sem_terms(model_ml_multigroup)[2]).A isa LowerTriangular
end

############################################################################################
# ML estimation - user defined loss function
############################################################################################

struct UserSemML{O, I} <: SemLoss{O, I}
    hessianeval::ExactHessian

    observed::O
    implied::I

    UserSemML(observed::SemObserved, implied::SemImplied) =
        new{typeof(observed), typeof(implied)}(ExactHessian(), observed, implied)
end

function SEM.objective(ml::UserSemML, params)
    Σ = implied(ml).Σ
    Σₒ = SEM.obs_cov(observed(ml))
    if !isposdef(Σ)
        return Inf
    else
        return logdet(Σ) + tr(inv(Σ) * Σₒ)
    end
end

# models
model_ml_multigroup = Sem(
    SemML(obs_g1, RAMSymbolic(specification_g1)),
    SEM.FiniteDiffWrapper(UserSemML(obs_g2, RAMSymbolic(specification_g2))),
)

@testset "gradients_user_defined_loss" begin
    test_gradient(model_ml_multigroup, start_test; atol = 1e-9)
end

# fit
@testset "solution_user_defined_loss" begin
    solution = sem_fit(semoptimizer, model_ml_multigroup)
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

model_ls_multigroup = Sem(
    SemWLS(obs_g1, RAMSymbolic(specification_g1, vech = true)),
    SemWLS(obs_g2, RAMSymbolic(specification_g2, vech = true)),
)

@testset "ls_gradients_multigroup" begin
    test_gradient(model_ls_multigroup, start_test; atol = 1e-9)
end

@testset "ls_solution_multigroup" begin
    solution = sem_fit(semoptimizer, model_ls_multigroup)
    update_estimate!(partable, solution)
    test_estimates(
        partable,
        solution_lav[:parameter_estimates_ls];
        atol = 1e-4,
        lav_groups = Dict(:Pasteur => 1, :Grant_White => 2),
    )
end

@testset "fitmeasures/se_ls" begin
    solution_ls = sem_fit(semoptimizer, model_ls_multigroup)
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
    model_ml_multigroup = Sem(
        SemFIML(
            SemObservedMissing(
                data = dat_miss_g1,
                observed_vars = SEM.observed_vars(specification_miss_g1),
            ),
            RAM(specification_miss_g1),
        ),
        SemFIML(
            SemObservedMissing(
                data = dat_miss_g2,
                observed_vars = SEM.observed_vars(specification_miss_g2),
            ),
            RAM(specification_miss_g2),
        ),
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
    end

    @testset "fiml_solution_multigroup" begin
        solution = sem_fit(semoptimizer, model_ml_multigroup)
        update_estimate!(partable_miss, solution)
        test_estimates(
            partable_miss,
            solution_lav[:parameter_estimates_fiml];
            atol = 1e-4,
            lav_groups = Dict(:Pasteur => 1, :Grant_White => 2),
        )
    end

    @testset "fitmeasures/se_fiml" begin
        solution = sem_fit(semoptimizer, model_ml_multigroup)
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
