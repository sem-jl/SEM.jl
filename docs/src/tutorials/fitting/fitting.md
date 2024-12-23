# Model fitting

As we saw in [A first model](@ref), after you have build a model, you can fit it via

```julia
model_fit = sem_fit(model)

# output

Fitted Structural Equation Model 
=============================================== 
--------------------- Model ------------------- 

Structural Equation Model 
- Loss Functions 
   SemML
- Fields 
   observed:  SemObservedData 
   implied:   RAM 
   optimizer: SemOptimizerOptim 

------------- Optimization result ------------- 

 * Status: success

 * Candidate solution
    Final objective value:     2.120543e+01

 * Found with
    Algorithm:     L-BFGS

 * Convergence measures
    |x - x'|               = 6.13e-05 ≰ 1.5e-08
    |x - x'|/|x'|          = 8.21e-06 ≰ 0.0e+00
    |f(x) - f(x')|         = 1.05e-09 ≰ 0.0e+00
    |f(x) - f(x')|/|f(x')| = 4.94e-11 ≤ 1.0e-10
    |g(x)|                 = 2.48e-05 ≰ 1.0e-08

 * Work counters
    Seconds run:   0  (vs limit Inf)
    Iterations:    175
    f(x) calls:    524
    ∇f(x) calls:   524
```

You may optionally specify [Starting values](@ref).

# API - model fitting

```@docs
sem_fit
```