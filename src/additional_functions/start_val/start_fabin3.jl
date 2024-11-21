"""
    start_fabin3(model)

Return a vector of FABIN 3 starting values (see Hägglund 1982).
Not available for ensemble models.
"""
function start_fabin3 end

# splice model and loss functions
function start_fabin3(model::AbstractSemSingle; kwargs...)
    return start_fabin3(
        model.observed,
        model.imply,
        model.optimizer,
        model.loss.functions...,
        kwargs...,
    )
end

function start_fabin3(observed, imply, optimizer, args...; kwargs...)
    return start_fabin3(imply.ram_matrices, obs_cov(observed), obs_mean(observed))
end

# SemObservedMissing
function start_fabin3(observed::SemObservedMissing, imply, optimizer, args...; kwargs...)
    if !observed.em_model.fitted
        em_mvn(observed; kwargs...)
    end

    return start_fabin3(imply.ram_matrices, observed.em_model.Σ, observed.em_model.μ)
end

function start_fabin3(ram_matrices::RAMMatrices, Σ, μ)
    A_ind, S_ind, F_ind, M_ind, n_par = ram_matrices.A_ind,
    ram_matrices.S_ind,
    ram_matrices.F_ind,
    ram_matrices.M_ind,
    nparams(ram_matrices)

    start_val = zeros(n_par)
    n_obs = nobserved_vars(ram_matrices)
    n_var = nvars(ram_matrices)
    n_latent = nlatent_vars(ram_matrices)

    C_indices = CartesianIndices((n_var, n_var))

    # check in which matrix each parameter appears

    indices = Vector{CartesianIndex{2}}(undef, n_par)

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
    for (i, S_ind) in enumerate(S_ind)
        for c_ind in C_indices[S_ind]
            (c_ind[1] == c_ind[2]) || continue # covariances stay 0
            pos = searchsortedfirst(F_ind, c_ind[1])
            start_val[i] =
                (pos <= length(F_ind)) && (F_ind[pos] == c_ind[1]) ? Σ[pos, pos] / 2 : 0.05
            break # i-th parameter initialized
        end
    end

    # set loadings
    constants = ram_matrices.constants
    A_ind_c = [linear2cartesian(ind, (n_var, n_var)) for ind in A_ind]
    # ind_Λ = findall([is_in_Λ(ind_vec, F_ind) for ind_vec in A_ind_c])

    function calculate_lambda(
        ref::Integer,
        indicator::Integer,
        indicators::AbstractVector{<:Integer},
    )
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

    for i in setdiff(1:n_var, F_ind)
        reference = Int64[]
        indicators = Int64[]
        indicator2parampos = Dict{Int, Int}()

        for (j, Aj_ind_c) in enumerate(A_ind_c)
            for ind_c in Aj_ind_c
                (ind_c[2] == i) || continue
                ind_pos = searchsortedfirst(F_ind, ind_c[1])
                if (ind_pos <= length(F_ind)) && (F_ind[ind_pos] == ind_c[1])
                    push!(indicators, ind_pos)
                    indicator2parampos[ind_pos] = j
                end
            end
        end

        for ram_const in constants
            if (ram_const.matrix == :A) && (ram_const.index[2] == i)
                ind_pos = searchsortedfirst(F_ind, ram_const.index[1])
                if (ind_pos <= length(F_ind)) && (F_ind[ind_pos] == ram_const.index[1])
                    if isone(ram_const.value)
                        push!(reference, ind_pos)
                    else
                        push!(indicators, ind_pos)
                        # no parameter associated
                    end
                end
            end
        end

        # is there at least one reference indicator?
        if length(reference) > 0
            if (length(reference) > 1) && isempty(indicator2parampos) # don't warn if entire column is fixed
                @warn "You have more than 1 scaling indicator for $(ram_matrices.colnames[i])"
            end
            ref = reference[1]

            for (j, indicator) in enumerate(indicators)
                if (indicator != ref) &&
                   (parampos = get(indicator2parampos, indicator, 0)) != 0
                    start_val[parampos] = calculate_lambda(ref, indicator, indicators)
                end
            end
            # no reference indicator:
        elseif length(indicators) > 0
            ref = indicators[1]
            λ = Vector{Float64}(undef, length(indicators))
            λ[1] = 1.0
            for (j, indicator) in enumerate(indicators)
                if indicator != ref
                    λ[j] = calculate_lambda(ref, indicator, indicators)
                end
            end

            Σ_λ = Σ[indicators, indicators]
            l₂ = sum(abs2, λ)
            D = λ * λ' ./ l₂
            θ = (I - D .^ 2) \ (diag(Σ_λ - D * Σ_λ * D))

            # 3. psi
            Σ₁ = Σ_λ - Diagonal(θ)
            Ψ = dot(λ, Σ₁, λ) / l₂^2

            λ .*= sign(Ψ) * sqrt(abs(Ψ))

            for (j, indicator) in enumerate(indicators)
                if (parampos = get(indicator2parampos, indicator, 0)) != 0
                    start_val[parampos] = λ[j]
                end
            end
        else
            @warn "No scaling indicators for $(ram_matrices.colnames[i])"
        end
    end

    # set means
    if !isnothing(M_ind)
        for (i, M_ind) in enumerate(M_ind)
            if length(M_ind) != 0
                ind = M_ind[1]
                pos = searchsortedfirst(F_ind, ind[1])
                if (pos <= length(F_ind)) && (F_ind[pos] == ind[1])
                    start_val[i] = μ[pos]
                end # latent means stay 0
            end
        end
    end

    return start_val
end

function is_in_Λ(ind_vec, F_ind)
    return any(ind -> !(ind[2] ∈ F_ind) && (ind[1] ∈ F_ind), ind_vec)
end
