"""
    start_fabin3(model)

Return a vector of FABIN 3 starting values (see Hägglund 1982).
Not available for ensemble models.
"""
function start_fabin3 end

# splice model and loss functions
function start_fabin3(model::AbstractSemSingle; kwargs...)
    return start_fabin3(model.observed, model.implied, model.loss.functions..., kwargs...)
end

function start_fabin3(observed, implied, args...; kwargs...)
    return start_fabin3(implied.ram_matrices, obs_cov(observed), obs_mean(observed))
end

# SemObservedMissing
function start_fabin3(observed::SemObservedMissing, implied, args...; kwargs...)
    if !observed.em_model.fitted
        em_mvn(observed; kwargs...)
    end

    return start_fabin3(implied.ram_matrices, observed.em_model.Σ, observed.em_model.μ)
end

function start_fabin3(
    ram_matrices::RAMMatrices,
    Σ::AbstractMatrix,
    μ::Union{AbstractVector, Nothing},
)
    A, S, F, M, n_par = ram_matrices.A,
    ram_matrices.S,
    ram_matrices.F,
    ram_matrices.M,
    nparams(ram_matrices)

    if !isnothing(M) && isnothing(μ)
        throw(ArgumentError("RAM has meanstructure, but no observed means provided."))
    end

    start_val = zeros(n_par)
    F_var2obs = Dict(
        i => F.rowval[F.colptr[i]] for i in axes(F, 2) if isobserved_var(ram_matrices, i)
    )
    @assert length(F_var2obs) == size(F, 1)

    # check in which matrix each parameter appears

    #=     in_S = length.(S_ind) .!= 0
        in_A = length.(A_ind) .!= 0
        A_ind_c = [linear2cartesian(ind, (n_var, n_var)) for ind in A_ind]
        in_Λ = [any(ind[2] .∈ F_ind) for ind in A_ind_c]

        if !isnothing(M)
            in_M = length.(M_ind) .!= 0
            in_any = in_A .| in_S .| in_M
        else
            in_any = in_A .| in_S
        end

        if !all(in_any)
            @warn "Could not determine fabin3 starting values for some parameters, default to 0."
        end =#

    # set undirected parameters in S
    S_indices = CartesianIndices(S)
    for j in 1:nparams(S)
        for lin_ind in param_occurences(S, j)
            to, from = Tuple(S_indices[lin_ind])
            if (to == from) # covariances start with 0
                # half of observed variance for observed, 0.05 for latent
                obs = get(F_var2obs, to, nothing)
                start_val[j] = !isnothing(obs) ? Σ[obs, obs] / 2 : 0.05
                break # j-th parameter initialized
            end
        end
    end

    # set loadings
    A_indices = CartesianIndices(A)
    # ind_Λ = findall([is_in_Λ(ind_vec, F_ind) for ind_vec in A_ind_c])

    # collect latent variable indicators in A
    # maps latent parameter to the vector of dependent vars
    # the 2nd index in the pair specified the parameter index,
    # 0 if no parameter (constant), -1 if constant=1
    var2indicators = Dict{Int, Vector{Pair{Int, Int}}}()
    for j in 1:nparams(A)
        for lin_ind in param_occurences(A, j)
            to, from = Tuple(A_indices[lin_ind])
            haskey(F_var2obs, from) && continue # skip observed
            obs = get(F_var2obs, to, nothing)
            if !isnothing(obs)
                indicators = get!(() -> Vector{Pair{Int, Int}}(), var2indicators, from)
                push!(indicators, obs => j)
            end
        end
    end

    for (lin_ind, val) in A.constants
        iszero(val) && continue # only non-zero loadings
        to, from = Tuple(A_indices[lin_ind])
        haskey(F_var2obs, from) && continue # skip observed
        obs = get(F_var2obs, to, nothing)
        if !isnothing(obs)
            indicators = get!(() -> Vector{Pair{Int, Int}}(), var2indicators, from)
            push!(indicators, obs => ifelse(isone(val), -1, 0)) # no parameter associated, -1 = reference, 0 = indicator
        end
    end

    # calculate starting values for parameters of latent regression vars
    function calculate_lambda(ref::Integer, indicator::Integer, indicators::AbstractVector)
        instruments = filter(i -> (i != ref) && (i != indicator), indicators)
        if length(instruments) == 1
            s13 = Σ[ref, instruments[1]]
            s32 = Σ[instruments[1], indicator]
            return s32 / s13
        else
            s13 = Σ[ref, instruments]
            s32 = Σ[instruments, indicator]
            S33 = Σ[instruments, instruments]
            temp = S33 \ s13
            return dot(s32, temp) / dot(s13, temp)
        end
    end

    for (i, indicators) in pairs(var2indicators)
        reference = [obs for (obs, param) in indicators if param == -1]
        indicator_obs = first.(indicators)
        # is there at least one reference indicator?
        if length(reference) > 0
            if (length(reference) > 1) && any(((obs, param),) -> param > 0, indicators) # don't warn if entire column is fixed
                @warn "You have more than 1 scaling indicator for $(ram_matrices.vars[i])"
            end
            ref = reference[1]

            for (indicator, param) in indicators
                if (indicator != ref) && (param > 0)
                    start_val[param] = calculate_lambda(ref, indicator, indicator_obs)
                end
            end
            # no reference indicator:
        else
            ref = indicator_obs[1]
            λ = Vector{Float64}(undef, length(indicator_obs))
            λ[1] = 1.0
            for (j, indicator) in enumerate(indicator_obs)
                if indicator != ref
                    λ[j] = calculate_lambda(ref, indicator, indicator_obs)
                end
            end

            Σ_λ = Σ[indicator_obs, indicator_obs]
            l₂ = sum(abs2, λ)
            D = λ * λ' ./ l₂
            θ = (I - D .^ 2) \ (diag(Σ_λ - D * Σ_λ * D))

            # 3. psi
            Σ₁ = Σ_λ - Diagonal(θ)
            Ψ = dot(λ, Σ₁, λ) / l₂^2

            λ .*= sign(Ψ) * sqrt(abs(Ψ))

            for (j, (_, param)) in enumerate(indicators)
                if param > 0
                    start_val[param] = λ[j]
                end
            end
        end
    end

    if !isnothing(M)
        # set starting values of the observed means
        for j in 1:nparams(M)
            M_ind = param_occurences(M, j)
            if !isempty(M_ind)
                obs = get(F_var2obs, M_ind[1], nothing)
                if !isnothing(obs)
                    start_val[j] = μ[obs]
                end # latent means stay 0
            end
        end
    end

    return start_val
end

function is_in_Λ(ind_vec, F_ind)
    return any(ind -> !(ind[2] ∈ F_ind) && (ind[1] ∈ F_ind), ind_vec)
end
