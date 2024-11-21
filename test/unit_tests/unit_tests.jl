using Test, SafeTestsets

@safetestset "Multithreading" begin
    include("multithreading.jl")
end

@safetestset "Matrix algebra helper functions" begin
    include("matrix_helpers.jl")
end

@safetestset "SemObserved" begin
    include("data_input_formats.jl")
end

@safetestset "SemSpecification" begin
    include("specification.jl")
end

@safetestset "Sem model" begin
    include("model.jl")
end
