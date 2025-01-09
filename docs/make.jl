using Documenter, StructuralEquationModels

makedocs(
    sitename = "StructuralEquationModels.jl",
    pages = [
        "index.md",
        "Tutorials" => [
            "A first model" => "tutorials/first_model.md",
            "Our Concept of a Structural Equation Model" => "tutorials/concept.md",
            "Model Specification" => [
                "tutorials/specification/specification.md",
                "tutorials/specification/graph_interface.md",
                "tutorials/specification/ram_matrices.md",
                "tutorials/specification/parameter_table.md",
            ],
            "Model Construction" => [
                "tutorials/construction/construction.md",
                "tutorials/construction/outer_constructor.md",
                "tutorials/construction/build_by_parts.md",
            ],
            "Optimization Backends" =>
                ["tutorials/backends/optim.md", "tutorials/backends/nlopt.md"],
            "Model Fitting" => "tutorials/fitting/fitting.md",
            "Model Inspection" => "tutorials/inspection/inspection.md",
            "Mean Structures" => "tutorials/meanstructure.md",
            "Collections" => [
                "tutorials/collection/collection.md",
                "tutorials/collection/multigroup.md",
            ],
            "Constraints" => "tutorials/constraints/constraints.md",
        ],
        "Developer documentation" => [
            "Extending the package" => "developer/extending.md",
            "Custom loss functions" => "developer/loss.md",
            "Custom implied types" => "developer/implied.md",
            "Custom optimizer types" => "developer/optimizer.md",
            "Custom observed types" => "developer/observed.md",
            "Custom model types" => "developer/sem.md",
        ],
        "Advanced tutorials" =>
            ["Regularization" => "tutorials/regularization/regularization.md"],
        "Performance tips" => [
            "Model sorting" => "performance/sorting.md",
            "MKL" => "performance/mkl.md",
            "Simulation studies" => "performance/simulation.md",
            "Symbolic precomputation" => "performance/symbolic.md",
            "Starting values" => "performance/starting_values.md",
            "Parametric Types" => "performance/parametric.md",
        ],
        "Internals and design" => [
            "Internals and design" => "internals/internals.md",
            "files" => "internals/files.md",
            "types" => "internals/types.md",
        ],
        "Complementary material" => ["Mathematical appendix" => "complementary/maths.md"],
    ],
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        assets = ["assets/favicon_zeta.ico"],
        ansicolor = true,
        collapselevel = 1,
    ),
    doctest = false,
)

# doctest(StructuralEquationModels, fix=true)

deploydocs(
    repo = "github.com/StructuralEquationModels/StructuralEquationModels.jl",
    devbranch = "devel",
    push_preview = "push_preview=true" ∈ ARGS,
)
