using StructuralEquationModels, Test, FiniteDiff

SEM = StructuralEquationModels

include(
    joinpath(
        chop(dirname(pathof(StructuralEquationModels)), tail = 3),
        "test/examples/helper.jl",
    ),
)

############################################################################################
### data
############################################################################################

dat = example_data("political_democracy")
dat_missing = example_data("political_democracy_missing")
solution_lav = example_data("political_democracy_solution")

############################################################################################
### specification - RAMMatrices
############################################################################################

# w.o. meanstructure -----------------------------------------------------------------------

x = Symbol.("x" .* string.(1:31))

S = [
    :x1   0    0     0     0      0     0     0     0     0     0     0     0     0
    0     :x2  0     0     0      0     0     0     0     0     0     0     0     0
    0     0     :x3  0     0      0     0     0     0     0     0     0     0     0
    0     0     0     :x4  0      0     0     :x15  0     0     0     0     0     0
    0     0     0     0     :x5   0     :x16  0     :x17  0     0     0     0     0
    0     0     0     0     0     :x6  0      0     0     :x18  0     0     0     0
    0     0     0     0     :x16  0     :x7   0     0     0     :x19  0     0     0
    0     0     0     :x15 0      0     0     :x8   0     0     0     0     0     0
    0     0     0     0     :x17  0     0     0     :x9   0     :x20  0     0     0
    0     0     0     0     0     :x18 0      0     0     :x10  0     0     0     0
    0     0     0     0     0     0     :x19  0     :x20  0     :x11  0     0     0
    0     0     0     0     0     0     0     0     0     0     0     :x12  0     0
    0     0     0     0     0     0     0     0     0     0     0     0     :x13  0
    0     0     0     0     0     0     0     0     0     0     0     0     0     :x14
]

F = [
    1.0 0 0 0 0 0 0 0 0 0 0 0 0 0
    0 1 0 0 0 0 0 0 0 0 0 0 0 0
    0 0 1 0 0 0 0 0 0 0 0 0 0 0
    0 0 0 1 0 0 0 0 0 0 0 0 0 0
    0 0 0 0 1 0 0 0 0 0 0 0 0 0
    0 0 0 0 0 1 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 1 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 1 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 1 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 1 0 0 0 0
    0 0 0 0 0 0 0 0 0 0 1 0 0 0
]

A = [
    0  0  0  0  0  0  0  0  0  0  0     1.0   0     0
    0  0  0  0  0  0  0  0  0  0  0     :x21  0     0
    0  0  0  0  0  0  0  0  0  0  0     :x22  0     0
    0  0  0  0  0  0  0  0  0  0  0     0     1.0   0
    0  0  0  0  0  0  0  0  0  0  0     0     :x23  0
    0  0  0  0  0  0  0  0  0  0  0     0     :x24  0
    0  0  0  0  0  0  0  0  0  0  0     0     :x25  0
    0  0  0  0  0  0  0  0  0  0  0     0     0     1
    0  0  0  0  0  0  0  0  0  0  0     0     0     :x26
    0  0  0  0  0  0  0  0  0  0  0     0     0     :x27
    0  0  0  0  0  0  0  0  0  0  0     0     0     :x28
    0  0  0  0  0  0  0  0  0  0  0     0     0     0
    0  0  0  0  0  0  0  0  0  0  0     :x29  0     0
    0  0  0  0  0  0  0  0  0  0  0     :x30  :x31  0
]

spec = RAMMatrices(;
    A = A,
    S = S,
    F = F,
    params = x,
    vars = [:x1, :x2, :x3, :y1, :y2, :y3, :y4, :y5, :y6, :y7, :y8, :ind60, :dem60, :dem65],
)

partable = ParameterTable(spec)

@test SEM.params(spec) == SEM.params(partable)

# w. meanstructure -------------------------------------------------------------------------

M = [:x32; :x33; :x34; :x35; :x36; :x37; :x38; :x35; :x36; :x37; :x38; 0.0; 0.0; 0.0]

spec_mean = RAMMatrices(;
    A = A,
    S = S,
    F = F,
    M = M,
    params = [SEM.params(spec); Symbol.("x", string.(32:38))],
    vars = [:x1, :x2, :x3, :y1, :y2, :y3, :y4, :y5, :y6, :y7, :y8, :ind60, :dem60, :dem65],
)

partable_mean = ParameterTable(spec_mean)

@test SEM.params(partable_mean) == SEM.params(spec_mean)

start_test = [fill(1.0, 11); fill(0.05, 3); fill(0.05, 6); fill(0.5, 8); fill(0.05, 3)]
start_test_mean =
    [fill(1.0, 11); fill(0.05, 3); fill(0.05, 6); fill(0.5, 8); fill(0.05, 3); fill(0.1, 7)]

opt_engine = :Optim
@testset "RAMMatrices | constructor | Optim" begin
    include("constructor.jl")
end

opt_engine = :NLopt
@testset "RAMMatrices | constructor | NLopt" begin
    include("constructor.jl")
end

if is_extended_tests()
    opt_engine = :Optim
    @testset "RAMMatrices | parts | Optim" begin
        include("by_parts.jl")
    end
    opt_engine = :NLopt
    @testset "RAMMatrices | parts | NLopt" begin
        include("by_parts.jl")
    end
end

@testset "constraints | NLopt" begin
    include("constraints.jl")
end

############################################################################################
### specification - RAMMatrices → ParameterTable
############################################################################################

spec = ParameterTable(spec)
spec_mean = ParameterTable(spec_mean)

@test SEM.params(spec) == SEM.params(partable)

partable = spec
partable_mean = spec_mean

opt_engine = :Optim
@testset "RAMMatrices → ParameterTable | constructor | Optim" begin
    include("constructor.jl")
end
opt_engine = :NLopt
@testset "RAMMatrices → ParameterTable | constructor | NLopt" begin
    include("constructor.jl")
end

if is_extended_tests()
    opt_engine = :Optim
    @testset "RAMMatrices → ParameterTable | parts | Optim" begin
        include("by_parts.jl")
    end
    opt_engine = :NLopt
    @testset "RAMMatrices → ParameterTable | parts | NLopt" begin
        include("by_parts.jl")
    end
end

############################################################################################
### specification - Graph
############################################################################################

observed_vars = [:x1, :x2, :x3, :y1, :y2, :y3, :y4, :y5, :y6, :y7, :y8]
latent_vars = [:ind60, :dem60, :dem65]

graph = @StenoGraph begin
    # loadings
    ind60 → fixed(1) * x1 + x2 + x3
    dem60 → fixed(1) * y1 + y2 + y3 + y4
    dem65 → fixed(1) * y5 + y6 + y7 + y8
    # latent regressions
    label(:a) * dem60 ← ind60
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

spec = ParameterTable(graph, latent_vars = latent_vars, observed_vars = observed_vars)

sort_vars!(spec)

partable = spec

# meanstructure
mean_labels = label.([:m1, :m2, :m3, :m4, :m5, :m6, :m7, :m4, :m5, :m6, :m7])

graph = @StenoGraph begin
    # loadings
    ind60 → fixed(1) * x1 + x2 + x3
    dem60 → fixed(1) * y1 + y2 + y3 + y4
    dem65 → fixed(1) * y5 + y6 + y7 + y8
    # latent regressions
    label(:a) * dem60 ← ind60
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
    # means
    Symbol("1") → _(mean_labels) .* _(observed_vars)
    Symbol("1") → fixed(0) * ind60
end

spec_mean = ParameterTable(graph, latent_vars = latent_vars, observed_vars = observed_vars)

sort_vars!(spec_mean)

partable_mean = spec_mean

start_test = [fill(0.5, 8); fill(0.05, 3); fill(1.0, 11); fill(0.05, 9)]
start_test_mean =
    [fill(0.5, 8); fill(0.05, 3); fill(1.0, 11); fill(0.05, 3); fill(0.05, 13)]

opt_engine = :Optim
@testset "Graph → ParameterTable | constructor | Optim" begin
    include("constructor.jl")
end
opt_engine = :NLopt
@testset "Graph → ParameterTable | constructor | NLopt" begin
    include("constructor.jl")
end

if is_extended_tests()
    opt_engine = :Optim
    @testset "Graph → ParameterTable | parts | Optim" begin
        include("by_parts.jl")
    end
    opt_engine = :NLopt
    @testset "Graph → ParameterTable | parts | NLopt" begin
        include("by_parts.jl")
    end
end
