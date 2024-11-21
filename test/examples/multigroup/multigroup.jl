using StructuralEquationModels, Test, FiniteDiff
using LinearAlgebra: diagind, LowerTriangular

const SEM = StructuralEquationModels

include(joinpath(chop(dirname(pathof(SEM)), tail = 3), "test/examples/helper.jl"))

dat = example_data("holzinger_swineford")
dat_missing = example_data("holzinger_swineford_missing")
solution_lav = example_data("holzinger_swineford_solution")

dat_g1 = dat[dat.school.=="Pasteur", :]
dat_g2 = dat[dat.school.=="Grant-White", :]

dat_miss_g1 = dat_missing[dat_missing.school.=="Pasteur", :]
dat_miss_g2 = dat_missing[dat_missing.school.=="Grant-White", :]

dat.school = ifelse.(dat.school .== "Pasteur", :Pasteur, :Grant_White)
dat_missing.school = ifelse.(dat_missing.school .== "Pasteur", :Pasteur, :Grant_White)

############################################################################################
### specification - RAMMatrices
############################################################################################

x = Symbol.(:x, 1:36)

F = zeros(9, 12)
F[diagind(F)] .= 1.0

A = Matrix{Any}(zeros(12, 12))
A[1, 10] = 1.0;
A[4, 11] = 1.0;
A[7, 12] = 1.0;
A[2:3, 10] .= x[16:17];
A[5:6, 11] .= x[18:19];
A[8:9, 12] .= x[20:21];

# group 1
S1 = Matrix{Any}(zeros(12, 12))
S1[diagind(S1)] .= x[1:12]
S1[10, 11] = x[13];
S1[11, 10] = x[13];
S1[10, 12] = x[14];
S1[12, 10] = x[14];
S1[12, 11] = x[15];
S1[11, 12] = x[15];

# group 2
S2 = Matrix{Any}(zeros(12, 12))
S2[diagind(S2)] .= x[22:33]
S2[10, 11] = x[34];
S2[11, 10] = x[34];
S2[10, 12] = x[35];
S2[12, 10] = x[35];
S2[12, 11] = x[36];
S2[11, 12] = x[36];

specification_g1 = RAMMatrices(;
    A = A,
    S = S1,
    F = F,
    params = x,
    colnames = [:x1, :x2, :x3, :x4, :x5, :x6, :x7, :x8, :x9, :visual, :textual, :speed],
)

specification_g2 = RAMMatrices(;
    A = A,
    S = S2,
    F = F,
    params = x,
    colnames = [:x1, :x2, :x3, :x4, :x5, :x6, :x7, :x8, :x9, :visual, :textual, :speed],
)

partable = EnsembleParameterTable(
    :Pasteur => specification_g1,
    :Grant_White => specification_g2
)

specification_miss_g1 = nothing
specification_miss_g2 = nothing

start_test = [
    fill(1.0, 9)
    fill(0.05, 3)
    fill(0.01, 3)
    fill(0.5, 6)
    fill(1.0, 9)
    fill(0.05, 3)
    fill(0.01, 3)
]
semoptimizer = SemOptimizerOptim

@testset "RAMMatrices | constructor | Optim" begin
    include("build_models.jl")
end

############################################################################################
### specification - Graph
############################################################################################

# w.o. meanstructure -----------------------------------------------------------------------

latent_vars = [:visual, :textual, :speed]
observed_vars = Symbol.(:x, 1:9)

graph = @StenoGraph begin
    # measurement model
    visual → fixed(1.0, 1.0) * x1 + label(:λ₂, :λ₂) * x2 + label(:λ₃, :λ₃) * x3
    textual → fixed(1.0, 1.0) * x4 + label(:λ₅, :λ₅) * x5 + label(:λ₆, :λ₆) * x6
    speed → fixed(1.0, 1.0) * x7 + label(:λ₈, :λ₈) * x8 + label(:λ₉, :λ₉) * x9
    # variances and covariances
    _(observed_vars) ↔ _(observed_vars)
    _(latent_vars) ⇔ _(latent_vars)
end

partable = EnsembleParameterTable(
    graph;
    observed_vars = observed_vars,
    latent_vars = latent_vars,
    groups = [:Pasteur, :Grant_White],
)

specification = convert(Dict{Symbol, RAMMatrices}, partable)

specification_g1 = specification[:Pasteur]
specification_g2 = specification[:Grant_White]

# w. meanstructure (fiml) ------------------------------------------------------------------

latent_vars = [:visual, :textual, :speed]
observed_vars = Symbol.(:x, 1:9)

graph = @StenoGraph begin
    # measurement model
    visual → fixed(1.0, 1.0) * x1 + label(:λ₂, :λ₂) * x2 + label(:λ₃, :λ₃) * x3
    textual → fixed(1.0, 1.0) * x4 + label(:λ₅, :λ₅) * x5 + label(:λ₆, :λ₆) * x6
    speed → fixed(1.0, 1.0) * x7 + label(:λ₈, :λ₈) * x8 + label(:λ₉, :λ₉) * x9
    # variances and covariances
    _(observed_vars) ↔ _(observed_vars)
    _(latent_vars) ⇔ _(latent_vars)

    Symbol("1") → _(observed_vars)
end

partable_miss = EnsembleParameterTable(
    graph;
    observed_vars = observed_vars,
    latent_vars = latent_vars,
    groups = [:Pasteur, :Grant_White],
)

specification_miss = convert(Dict{Symbol, RAMMatrices}, partable_miss)

specification_miss_g1 = specification_miss[:Pasteur]
specification_miss_g2 = specification_miss[:Grant_White]

start_test = [
    fill(0.5, 6)
    fill(1.0, 9)
    0.05
    0.01
    0.01
    0.05
    0.01
    0.05
    fill(1.0, 9)
    0.05
    0.01
    0.01
    0.05
    0.01
    0.05
]
semoptimizer = SemOptimizerOptim

@testset "Graph → Partable → RAMMatrices | constructor | Optim" begin
    include("build_models.jl")
end
