using StructuralEquationModels, Test, ProximalCore, ProximalAlgorithms, ProximalOperators

@testset "Ridge" begin
    include("ridge.jl")
end
@testset "Lasso" begin
    include("lasso.jl")
end
@testset "L0" begin
    include("l0.jl")
end
