# mutate by moving in the gradient direction
mutable struct AdamMutation{M <: AbstractSem, O, S} <: MutationOperator
    model::M
    optim::O
    opt_state::S
    params_fraction::Float64

    function AdamMutation(model::AbstractSem, params::AbstractDict)
        optim = RAdam(params[:AdamMutation_eta], params[:AdamMutation_beta])
        params_fraction = params[:AdamMutation_params_fraction]
        opt_state = Optimisers.init(optim, Vector{Float64}(undef, nparams(model)))

        new{typeof(model), typeof(optim), typeof(opt_state)}(
            model,
            optim,
            opt_state,
            params_fraction,
        )
    end
end

Base.show(io::IO, op::AdamMutation) =
    print(io, "AdamMutation(", op.optim, " state[3]=", op.opt_state[3], ")")

"""
Default parameters for `AdamMutation`.
"""
const AdamMutation_DefaultOptions = ParamsDict(
    :AdamMutation_eta => 1E-1,
    :AdamMutation_beta => (0.99, 0.999),
    :AdamMutation_params_fraction => 0.25,
)

function BlackBoxOptim.apply!(m::AdamMutation, v::AbstractVector{<:Real}, target_index::Int)
    grad = similar(v)
    obj = SEM.evaluate!(0.0, grad, nothing, m.model, v)
    @inbounds for i in eachindex(grad)
        (rand() > m.params_fraction) && (grad[i] = 0.0)
    end

    m.opt_state, dv = Optimisers.apply!(m.optim, m.opt_state, v, grad)
    if (m.opt_state[3][1] <= 1E-20) || !isfinite(obj) || any(!isfinite, dv)
        m.opt_state = Optimisers.init(m.optim, v)
    else
        v .-= dv
    end

    return v
end
