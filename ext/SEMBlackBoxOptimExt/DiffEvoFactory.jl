"""
Base class for factories of optimizers for a specific problem.
"""
abstract type OptimizerFactory{P <: OptimizationProblem} end

problem(factory::OptimizerFactory) = factory.problem

const OptController_DefaultParameters = ParamsDict(
    :MaxTime => 60.0,
    :MaxSteps => 10^8,
    :TraceMode => :compact,
    :TraceInterval => 5.0,
    :RecoverResults => false,
    :SaveTrace => false,
)

function generate_opt_controller(alg::Optimizer, optim_factory::OptimizerFactory, params)
    return BlackBoxOptim.OptController(
        alg,
        problem(optim_factory),
        BlackBoxOptim.chain(
            BlackBoxOptim.DefaultParameters,
            OptController_DefaultParameters,
            params,
        ),
    )
end

function check_population(
    factory::OptimizerFactory,
    popmatrix::BlackBoxOptim.PopulationMatrix,
)
    ssp = factory |> problem |> search_space
    for i in 1:popsize(popmatrix)
        @assert popmatrix[:, i] ∈ ssp "Individual $i is out of space: $(popmatrix[:,i])" # fitness: $(fitness(population, i))"
    end
end

initial_search_space(factory::OptimizerFactory, id::Int) = search_space(factory.problem)

function initial_population_matrix(factory::OptimizerFactory, id::Int)
    #@info "Standard initial_population_matrix()"
    ini_ss = initial_search_space(factory, id)
    if !isempty(factory.initial_population)
        numdims(factory.initial_population) == numdims(factory.problem) || throw(
            DimensionMismatch(
                "Dimensions of :Population ($(numdims(factory.initial_population))) " *
                "are different from the problem dimensions ($(numdims(factory.problem)))",
            ),
        )
        res = factory.initial_population[
            :,
            StatsBase.sample(
                1:popsize(factory.initial_population),
                factory.population_size,
            ),
        ]
    else
        res = rand_individuals(ini_ss, factory.population_size, method = :latin_hypercube)
    end
    prj = RandomBound(ini_ss)
    if size(res, 2) > 1
        apply!(prj, view(res, :, 1), SEM.start_fabin3(factory.problem.model))
    end
    if size(res, 2) > 2
        apply!(prj, view(res, :, 2), SEM.start_simple(factory.problem.model))
    end
    return res
end

# convert individuals in the archive into population matrix
population_matrix(archive::Any) = population_matrix!(
    Matrix{Float64}(undef, length(BlackBoxOptim.params(first(archive))), length(archive)),
    archive,
)

function population_matrix!(pop::AbstractMatrix{<:Real}, archive::Any)
    npars = length(BlackBoxOptim.params(first(archive)))
    size(pop, 1) == npars || throw(
        DimensionMismatch(
            "Matrix rows count ($(size(pop, 1))) doesn't match the number of problem dimensions ($(npars))",
        ),
    )
    @inbounds for (i, indi) in enumerate(archive)
        (i <= size(pop, 2)) || break
        pop[:, i] .= BlackBoxOptim.params(indi)
    end
    if size(pop, 2) > length(archive)
        @warn "Matrix columns count ($(size(pop, 2))) is bigger than population size ($(length(archive))), last columns not set"
    end
    return pop
end

generate_embedder(factory::OptimizerFactory, id::Int, problem::OptimizationProblem) =
    RandomBound(search_space(problem))

abstract type DiffEvoFactory{P <: OptimizationProblem} <: OptimizerFactory{P} end

generate_selector(
    factory::DiffEvoFactory,
    id::Int,
    problem::OptimizationProblem,
    population,
) = RadiusLimitedSelector(get(factory.params, :selector_radius, popsize(population) ÷ 5))

function generate_modifier(factory::DiffEvoFactory, id::Int, problem::OptimizationProblem)
    ops = GeneticOperator[
        MutationClock(UniformMutation(search_space(problem)), 1 / numdims(problem)),
        BlackBoxOptim.AdaptiveDiffEvoRandBin1(
            BlackBoxOptim.AdaptiveDiffEvoParameters(
                factory.params[:fdistr],
                factory.params[:crdistr],
            ),
        ),
        SimplexCrossover{3}(1.05),
        SimplexCrossover{2}(1.1),
        #SimulatedBinaryCrossover(0.05, 16.0),
        #SimulatedBinaryCrossover(0.05, 3.0),
        #SimulatedBinaryCrossover(0.1, 5.0),
        #SimulatedBinaryCrossover(0.2, 16.0),
        UnimodalNormalDistributionCrossover{2}(
            chain(BlackBoxOptim.UNDX_DefaultOptions, factory.params),
        ),
        UnimodalNormalDistributionCrossover{3}(
            chain(BlackBoxOptim.UNDX_DefaultOptions, factory.params),
        ),
        ParentCentricCrossover{2}(chain(BlackBoxOptim.PCX_DefaultOptions, factory.params)),
        ParentCentricCrossover{3}(chain(BlackBoxOptim.PCX_DefaultOptions, factory.params)),
    ]
    if problem isa SemModelBlackBoxOptimProblem
        push!(
            ops,
            AdamMutation(problem.model, chain(AdamMutation_DefaultOptions, factory.params)),
        )
    end
    FAGeneticOperatorsMixture(ops)
end

function generate_optimizer(
    factory::DiffEvoFactory,
    id::Int,
    problem::OptimizationProblem,
    popmatrix,
)
    population = FitPopulation(popmatrix, nafitness(fitness_scheme(problem)))
    BlackBoxOptim.DiffEvoOpt(
        "AdaptiveDE/rand/1/bin/gradient",
        population,
        generate_selector(factory, id, problem, population),
        generate_modifier(factory, id, problem),
        generate_embedder(factory, id, problem),
    )
end

const Population_DefaultParameters = ParamsDict(
    :Population => BlackBoxOptim.PopulationMatrix(undef, 0, 0),
    :PopulationSize => 100,
)

const DE_DefaultParameters = chain(
    ParamsDict(
        :SelectorRadius => 0,
        :fdistr =>
            BlackBoxOptim.BimodalCauchy(0.65, 0.1, 1.0, 0.1, clampBelow0 = false),
        :crdistr =>
            BlackBoxOptim.BimodalCauchy(0.1, 0.1, 0.95, 0.1, clampBelow0 = false),
    ),
    Population_DefaultParameters,
)

struct DefaultDiffEvoFactory{P <: OptimizationProblem} <: DiffEvoFactory{P}
    problem::P
    initial_population::BlackBoxOptim.PopulationMatrix
    population_size::Int
    params::ParamsDictChain
end

DefaultDiffEvoFactory(problem::OptimizationProblem; kwargs...) =
    DefaultDiffEvoFactory(problem, BlackBoxOptim.kwargs2dict(kwargs))

function DefaultDiffEvoFactory(problem::OptimizationProblem, params::AbstractDict)
    params = chain(DE_DefaultParameters, params)
    DefaultDiffEvoFactory{typeof(problem)}(
        problem,
        params[:Population],
        params[:PopulationSize],
        params,
    )
end

function BlackBoxOptim.bbsetup(factory::OptimizerFactory; kwargs...)
    popmatrix = initial_population_matrix(factory, 1)
    check_population(factory, popmatrix)
    alg = generate_optimizer(factory, 1, problem(factory), popmatrix)
    return generate_opt_controller(alg, factory, BlackBoxOptim.kwargs2dict(kwargs))
end
