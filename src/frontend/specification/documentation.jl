params(spec::SemSpecification) = spec.params

"""
    vars(semobj) -> Vector{Symbol}

Return the vector of SEM model variables (both observed and latent)
in the order specified by the model.
"""
function vars end

vars(spec::SemSpecification) = error("vars(spec::$(typeof(spec))) is not implemented")

"""
    observed_vars(semobj) -> Vector{Symbol}

Return the vector of SEM model observed variable in the order specified by the
model, which also should match the order of variables in [`SemObserved`](@ref).
"""
function observed_vars end

observed_vars(spec::SemSpecification) =
    error("observed_vars(spec::$(typeof(spec))) is not implemented")

"""
    latent_vars(semobj) -> Vector{Symbol}

Return the vector of SEM model latent variable in the order specified by the
model.
"""
function latent_vars end

latent_vars(spec::SemSpecification) =
    error("latent_vars(spec::$(typeof(spec))) is not implemented")

"""
`ParameterTable`s contain the specification of a structural equation model.

# Constructor

    (1) ParameterTable(;graph, observed_vars, latent_vars, ...)

    (2) ParameterTable(ram_matrices)

Return a `ParameterTable` constructed from (1) a graph or (2) RAM matrices.

# Arguments
- `graph`: graph defined via `@StenoGraph`
- `observed_vars::Vector{Symbol}`: observed variable names
- `latent_vars::Vector{Symbol}`: latent variable names
- `ram_matrices::RAMMatrices`: a `RAMMatrices` object

# Examples
See the online documentation on [Model specification](@ref) and the [ParameterTable interface](@ref).

# Extended help
## Additional keyword arguments
- `parname::Symbol = :θ`: prefix for automatically generated parameter labels
"""
function ParameterTable end

"""
`EnsembleParameterTable`s contain the specification of an ensemble structural equation model.

# Constructor

    (1) EnsembleParameterTable(;graph, observed_vars, latent_vars, groups)

    (2) EnsembleParameterTable(ps::Pair...; params = nothing)

Return an `EnsembleParameterTable` constructed from (1) a graph or (2) multiple specifications.

# Arguments
- `graph`: graph defined via `@StenoGraph`
- `observed_vars::Vector{Symbol}`: observed variable names
- `latent_vars::Vector{Symbol}`: latent variable names
- `params::Vector{Symbol}`: (optional) a vector of parameter names
- `ps::Pair...`: `:group_name => specification`, where `specification` is either a `ParameterTable` or `RAMMatrices`

# Examples
See the online documentation on [Multigroup models](@ref).
"""
function EnsembleParameterTable end

"""
`RAMMatrices` contain the specification of a structural equation model.

# Constructor

    (1) RAMMatrices(partable::ParameterTable)

    (2) RAMMatrices(;A, S, F, M = nothing, params, vars)

    (3) RAMMatrices(partable::EnsembleParameterTable)

Return `RAMMatrices` constructed from (1) a parameter table or (2) individual matrices.

(3) Return a dictionary of `RAMMatrices` from an `EnsembleParameterTable` (keys are the group names).

# Arguments
- `partable`
- `A`: matrix of directed effects
- `S`: matrix of undirected effects
- `F`: filter matrix
- `M`: vector of mean effects
- `params::Vector{Symbol}`: parameter labels
- `vars::Vector{Symbol}`: variable names corresponding to the A, S and F matrix columns

# Examples
See the online documentation on [Model specification](@ref) and the [RAMMatrices interface](@ref).
"""
function RAMMatrices end

"""
    fixed(args...)

Fix parameters to a certain value.
For ensemble models, multiple values (one for each submodel/group) are needed.

# Examples
```julia
graph = @StenoGraph begin
    x → fixed(1)*y
end
```
"""
function fixed end

"""
    start(args...)

Define starting values for parameters.
For ensemble models, multiple values (one for each submodel/group) are needed.

# Examples
```julia
graph = @StenoGraph begin
    x → start(1)*y
end
```
"""
function start end

"""
    label(args...)

Label parameters.
For ensemble models, multiple values (one for each submodel/group) are needed.

# Examples
```julia
graph = @StenoGraph begin
    x → label(:a)*y
end
```
"""
function label end
