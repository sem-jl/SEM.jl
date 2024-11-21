# RAMMatrices interface

Models can also be specified by an object of type `RAMMatrices`. 
The RAM (reticular action model) specification corresponds to three matrices; the `A` matrix containing all directed parameters, the `S` matrix containing all undirected parameters, and the `F` matrix filtering out latent variables from the model implied covariance.

The model implied covariance matrix for the observed variables of a SEM is then computed as
```math
\Sigma = F(I-A)^{-1}S(I-A)^{-T}F^T
```
For [A first model](@ref), the corresponding specification looks like this:

```julia

S =[:θ1   0    0     0     0      0     0     0     0     0     0     0     0     0
    0     :θ2  0     0     0      0     0     0     0     0     0     0     0     0
    0     0     :θ3  0     0      0     0     0     0     0     0     0     0     0
    0     0     0     :θ4  0      0     0     :θ15  0     0     0     0     0     0
    0     0     0     0     :θ5   0     :θ16  0     :θ17  0     0     0     0     0
    0     0     0     0     0     :θ6  0      0     0     :θ18  0     0     0     0
    0     0     0     0     :θ16  0     :θ7   0     0     0     :θ19  0     0     0
    0     0     0     :θ15 0      0     0     :θ8   0     0     0     0     0     0
    0     0     0     0     :θ17  0     0     0     :θ9   0     :θ20  0     0     0
    0     0     0     0     0     :θ18 0      0     0     :θ10  0     0     0     0
    0     0     0     0     0     0     :θ19  0     :θ20  0     :θ11  0     0     0
    0     0     0     0     0     0     0     0     0     0     0     :θ12  0     0
    0     0     0     0     0     0     0     0     0     0     0     0     :θ13  0
    0     0     0     0     0     0     0     0     0     0     0     0     0     :θ14]

F =[1.0 0 0 0 0 0 0 0 0 0 0 0 0 0
    0 1 0 0 0 0 0 0 0 0 0 0 0 0
    0 0 1 0 0 0 0 0 0 0 0 0 0 0
    0 0 0 1 0 0 0 0 0 0 0 0 0 0
    0 0 0 0 1 0 0 0 0 0 0 0 0 0
    0 0 0 0 0 1 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 1 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 1 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 1 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 1 0 0 0 0
    0 0 0 0 0 0 0 0 0 0 1 0 0 0]

A =[0  0  0  0  0  0  0  0  0  0  0     1.0   0     0
    0  0  0  0  0  0  0  0  0  0  0     :θ21  0     0
    0  0  0  0  0  0  0  0  0  0  0     :θ22  0     0
    0  0  0  0  0  0  0  0  0  0  0     0     1.0   0
    0  0  0  0  0  0  0  0  0  0  0     0     :θ23  0
    0  0  0  0  0  0  0  0  0  0  0     0     :θ24  0
    0  0  0  0  0  0  0  0  0  0  0     0     :θ25  0
    0  0  0  0  0  0  0  0  0  0  0     0     0     1
    0  0  0  0  0  0  0  0  0  0  0     0     0     :θ26
    0  0  0  0  0  0  0  0  0  0  0     0     0     :θ27
    0  0  0  0  0  0  0  0  0  0  0     0     0     :θ28
    0  0  0  0  0  0  0  0  0  0  0     0     0     0
    0  0  0  0  0  0  0  0  0  0  0     :θ29  0     0
    0  0  0  0  0  0  0  0  0  0  0     :θ30  :θ31  0]

θ = Symbol.(:θ, 1:31)

spec = RAMMatrices(;
    A = A, 
    S = S, 
    F = F, 
    params = θ,
    colnames = [:x1, :x2, :x3, :y1, :y2, :y3, :y4, :y5, :y6, :y7, :y8, :ind60, :dem60, :dem65]
)

model = Sem(
    specification = spec,
    data = example_data("political_democracy")
)
```

Let's look at this step by step:

First, we specify the `A`, `S` and `F`-Matrices. 
For a free parameter, we write a `Symbol` like `:θ1` (or any other symbol we like) to the corresponding place in the respective matrix, to constrain parameters to be equal we just use the same `Symbol` in the respective entries. 
To fix a parameter (as in the `A`-Matrix above), we just write down the number we want to fix it to. 
All other entries are 0.

Second, we specify a vector of symbols containing our parameters:

```julia
θ = Symbol.(:θ, 1:31)
```

Third, we construct an object of type `RAMMatrices`, passing our matrices and parameters, as well as the column names of our matrices. 
Those are quite important, as they will be used to rearrange your data to match it to your `RAMMatrices` specification.

```julia
spec = RAMMatrices(;
    A = A, 
    S = S, 
    F = F, 
    params = θ,
    colnames = [:x1, :x2, :x3, :y1, :y2, :y3, :y4, :y5, :y6, :y7, :y8, :ind60, :dem60, :dem65]
)
```

Finally, we construct a model, passing our `RAMMatrices` as the `specification = ... ` argument.

```julia
model = Sem(
    specification = spec,
    data = example_data("political_democracy")
)
```
## Meanstructure

According to the RAM, model implied mean values of the observed variables are computed as
```math
\mu = F(I-A)^{-1}M
```
where `M` is a vector of mean parameters. To estimate the means of the observed variables in our example (and set the latent means to `0`), we would specify the model just as before but add 

```julia
...

M = [:x32; :x33; :x34; :x35; :x36; :x37; :x38; :x39; :x40; :x41; :x42; 0; 0; 0]

θ = Symbol.(:θ, 1:42)

spec = RAMMatrices(;
    ...,
    M = M)

...

```

## Convert from and to ParameterTables

To convert a RAMMatrices object to a ParameterTable, simply use `partable = ParameterTable(ram_matrices)`. 
To convert an object of type `ParameterTable` to RAMMatrices, you can use `ram_matrices = RAMMatrices(partable)`.