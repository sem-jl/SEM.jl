module StructuralEquationModels

using LinearAlgebra,
    Optim,
    NLSolversBase,
    Statistics,
    StatsBase,
    SparseArrays,
    Symbolics,
    FiniteDiff,
    PrettyTables,
    Distributions,
    StenoGraphs,
    LazyArtifacts,
    DelimitedFiles,
    DataFrames

export StenoGraphs, @StenoGraph, meld

const SEM = StructuralEquationModels

# type hierarchy
include("types.jl")
include("objective_gradient_hessian.jl")

# helper objects and functions
include("additional_functions/commutation_matrix.jl")
include("additional_functions/params_array.jl")

# fitted objects
include("frontend/fit/SemFit.jl")
# specification of models
include("frontend/common.jl")
include("frontend/specification/checks.jl")
include("frontend/specification/ParameterTable.jl")
include("frontend/specification/RAMMatrices.jl")
include("frontend/specification/EnsembleParameterTable.jl")
include("frontend/specification/StenoGraphs.jl")
include("frontend/fit/summary.jl")
# pretty printing
include("frontend/pretty_printing.jl")
# observed
include("observed/abstract.jl")
include("observed/data.jl")
include("observed/covariance.jl")
include("observed/missing_pattern.jl")
include("observed/missing.jl")
include("observed/EM.jl")
# constructor
include("frontend/specification/Sem.jl")
include("frontend/specification/documentation.jl")
# implied
include("implied/abstract.jl")
include("implied/RAM/symbolic.jl")
include("implied/RAM/generic.jl")
include("implied/empty.jl")
# loss
include("loss/ML/ML.jl")
include("loss/ML/FIML.jl")
include("loss/regularization/ridge.jl")
include("loss/WLS/WLS.jl")
include("loss/constant/constant.jl")
# optimizer
include("optimizer/abstract.jl")
include("optimizer/Empty.jl")
include("optimizer/optim.jl")
# helper functions
include("additional_functions/helper.jl")
include("additional_functions/start_val/start_fabin3.jl")
include("additional_functions/start_val/start_simple.jl")
include("additional_functions/artifacts.jl")
include("additional_functions/simulation.jl")
# fit measures
include("frontend/fit/fitmeasures/AIC.jl")
include("frontend/fit/fitmeasures/BIC.jl")
include("frontend/fit/fitmeasures/chi2.jl")
include("frontend/fit/fitmeasures/df.jl")
include("frontend/fit/fitmeasures/minus2ll.jl")
include("frontend/fit/fitmeasures/p.jl")
include("frontend/fit/fitmeasures/RMSEA.jl")
include("frontend/fit/fitmeasures/fit_measures.jl")
# standard errors
include("frontend/fit/standard_errors/hessian.jl")
include("frontend/fit/standard_errors/bootstrap.jl")

export AbstractSem,
    AbstractSemSingle,
    AbstractSemCollection,
    Sem,
    SemFiniteDiff,
    SemEnsemble,
    MeanStruct,
    NoMeanStruct,
    HasMeanStruct,
    HessianEval,
    ExactHessian,
    ApproxHessian,
    SemImplied,
    RAMSymbolic,
    RAM,
    ImpliedEmpty,
    implied,
    start_val,
    start_fabin3,
    start_simple,
    SemLoss,
    SemLossFunction,
    SemML,
    SemFIML,
    em_mvn,
    SemRidge,
    SemConstant,
    SemWLS,
    loss,
    SemOptimizer,
    SemOptimizerEmpty,
    SemOptimizerOptim,
    optimizer,
    n_iterations,
    convergence,
    SemObserved,
    SemObservedData,
    SemObservedCovariance,
    SemObservedMissing,
    observed,
    obs_cov,
    obs_mean,
    nsamples,
    samples,
    sem_fit,
    SemFit,
    minimum,
    solution,
    sem_summary,
    objective!,
    gradient!,
    hessian!,
    objective_gradient!,
    objective_hessian!,
    gradient_hessian!,
    objective_gradient_hessian!,
    SemSpecification,
    RAMMatrices,
    ParameterTable,
    EnsembleParameterTable,
    update_partable!,
    update_estimate!,
    update_start!,
    update_se_hessian!,
    Fixed,
    fixed,
    Start,
    start,
    Label,
    label,
    nvars,
    vars,
    nlatent_vars,
    latent_vars,
    nobserved_vars,
    observed_vars,
    sort_vars!,
    sort_vars,
    params,
    nparams,
    param_indices,
    fit_measures,
    AIC,
    BIC,
    χ²,
    df,
    fit_measures,
    minus2ll,
    p_value,
    RMSEA,
    EmMVNModel,
    se_hessian,
    se_bootstrap,
    example_data,
    swap_observed,
    update_observed,
    @StenoGraph,
    →,
    ←,
    ↔,
    ⇔
end
