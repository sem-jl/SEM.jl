############################################################################
### test variables sorting
############################################################################

sort_vars!(partable)

model_ml_sorted = Sem(specification = partable, data = dat)

@testset "graph sorting" begin
    @test model_ml_sorted.imply.I_A isa LowerTriangular
end

@testset "ml_solution_sorted" begin
    solution_ml_sorted = sem_fit(model_ml_sorted)
    update_estimate!(partable, solution_ml_sorted)
    @test test_estimates(par_ml, partable, 0.01)
end
