using FiniteDiff

## loss
abstract type SemLossFunction end

struct SemLoss{F <: Tuple}
    functions::F
end

## Diff
abstract type SemDiff end

## Obs
abstract type SemObs end

## Imply
abstract type SemImply end

struct Sem{O <: SemObs, I <: SemImply, L <: SemLoss, D <: SemDiff}
    observed::O
    imply::I 
    loss::L 
    diff::D
end

function (model::Sem)(par, F, G, H)
    if !isnothing(G) G .= 0.0 end
    if !isnothing(H) H .= 0.0 end
    model.imply(par, F, G, H, model)
    F = model.loss(par, F, G, H, model)
    return F
end

function (loss::SemLoss)(par, F, G, H, model)
    if !isnothing(F)
        F = zero(eltype(par))
        for lossfun in loss.functions
            F += lossfun(par, F, G, H, model)
        end
        return F
    end
    for lossfun in loss.functions lossfun(par, F, G, H, model) end
end

######################## example ############################

struct myobs <: SemObs end

obsinst = myobs()

struct mydiff <: SemDiff end

diffinst = mydiff()

struct myimply <: SemImply
    Σ
end

struct myml <: SemLossFunction
    Σ
end

struct myhell <: SemLossFunction
    Σ
end

mlinst = myml([0.0])
hellinst = myhell([0.0])
implyinst = myimply([0.0])

modelinst = Sem(obsinst, implyinst, SemLoss((mlinst,hellinst)), diffinst)

function (imply::myimply)(par, F, G, H, model)
    imply.Σ[1] = par^2
end

function (lossfun::myml)(par, F, G, H, model)
    # do common computations here
    if !isnothing(G)
        G[1] += 4*par
    end
    # if isnothing(H) end
    if !isnothing(F)
        F = 2*model.imply.Σ[1]
        return F
    end
end

function (lossfun::myhell)(par, F, G, H, model)
    if !isnothing(G)
        G .+= FiniteDiff.finite_difference_gradient(par -> lossfun(par, model), [par])
    end
    #if isnothing(H) end
    if !isnothing(F)
        F = model.imply.Σ[1]^2
        return F
    end
end

function (lossfun::myhell)(par, model)
    lossfun.Σ[1] = par[1]^2
    return lossfun.Σ[1]^2
end

par = 2.0

grad = [0.0]

2*par^2 + par^4

4*par + 4*par^3

modelinst(par, 0.0, grad, nothing)

grad