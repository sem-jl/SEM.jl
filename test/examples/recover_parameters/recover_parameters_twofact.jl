using StructuralEquationModels, Distributions, Random, Optim, LineSearches

SEM = StructuralEquationModels

include(
    joinpath(
        chop(dirname(pathof(StructuralEquationModels)), tail = 3),
        "test/examples/helper.jl",
    ),
)

x = Symbol.("x", 1:13)

S = [
    :x1 0 0 0 0 0 0 0
    0 :x2 0 0 0 0 0 0
    0 0 :x3 0 0 0 0 0
    0 0 0 :x4 0 0 0 0
    0 0 0 0 :x5 0 0 0
    0 0 0 0 0 :x6 0 0
    0 0 0 0 0 0 :x7 :x9
    0 0 0 0 0 0 :x9 :x8
]

F = [
    1.0 0 0 0 0 0 0 0
    0 1.0 0 0 0 0 0 0
    0 0 1.0 0 0 0 0 0
    0 0 0 1.0 0 0 0 0
    0 0 0 0 1.0 0 0 0
    0 0 0 0 0 1.0 0 0
]

A = [
    0 0 0 0 0 0 1.0   0
    0 0 0 0 0 0 :x10 0
    0 0 0 0 0 0 :x11 0
    0 0 0 0 0 0 0    1.0
    0 0 0 0 0 0 0    :x12
    0 0 0 0 0 0 0    :x13
    0 0 0 0 0 0 0    0
    0 0 0 0 0 0 0    0
]

ram_matrices = RAMMatrices(; A = A, S = S, F = F, params = x, vars = nothing)

true_val = [
    repeat([1], 8)
    0.4
    repeat([0.8], 4)
]

start = [
    repeat([1], 9)
    repeat([0.5], 4)
]

start = [
    repeat([1], 9)
    repeat([0.5], 4)
]

implied_ml = RAMSymbolic(ram_matrices)

implied_ml.Σ_eval!(implied_ml.Σ, true_val)

true_dist = MultivariateNormal(implied_ml.Σ)

Random.seed!(1234)
x = permutedims(rand(true_dist, 10^5), (2, 1))

optimizer = SemOptimizerOptim(
    BFGS(; linesearch = BackTracking(order = 3), alphaguess = InitialHagerZhang()),# m = 100),
    Optim.Options(; f_tol = 1e-10, x_tol = 1.5e-8),
)

model_ml = Sem(SemML(SemObservedData(data = x, specification = ram_matrices), implied_ml))
objective!(model_ml, true_val)
solution_ml = sem_fit(optimizer, model_ml, start_val = start)

@test solution(solution_ml) ≈ true_val atol = 0.05
