# Maximum Likelihood Estimation

struct SemML <: SemObjective end
function (objective::SemML)(parameters, model::model)
      obs_cov = model.obs.cov
      n_man = size(obs_cov, 1)
      matrices = model.ram(parameters)
      imp_cov = sem.imp_cov(matrices)
      F_ML = log(det(imp_cov)) + tr(obs_cov*inv(imp_cov)) - log(det(obs_cov)) - n_man
      if size(matrices, 1) == 4
          mean_diff = model.obs.mean - sem.imp_mean(matrices)
          F_ML = F_ML + transpose(mean_diff)*inv(imp_cov)*mean_diff
      end
      return F_ML
end

### RegSem
struct SemMLLasso{P, W} <: SemObjective
    penalty::P
    wich::W
end
function (objective::SemMLLasso)(parameters, model::model)
      obs_cov = model.obs.cov
      obs_mean = model.obs.mean
      reg_vec = objective.which
      penalty = objective.penalty
      n_man = size(obs_cov, 1)
      matrices = model.ram(parameters)
      Cov_Exp = imp_cov(model, parameters)
      F_ML = log(det(Cov_Exp)) + tr(obs_cov*inv(Cov_Exp)) -
                  log(det(obs_cov)) - n_man + penalty*sum(transpose(parameters)[reg_vec])
      return F_ML
end
# doesnt work
function ML_ridge(parameters; ram, obs_cov, reg_vec, penalty)
      obs_cov = model.obs_cov
      obs_mean = model.obs_mean
      reg_vec = model.rec_vec
      penalty = model.penalty
      n_man = size(obs_cov, 1)
      matrices = model.ram(parameters)
      Cov_Exp = matrices[2]*inv(I-matrices[3])*matrices[1]*transpose(inv(I-matrices[3]))*transpose(matrices[2])
      F_ML = log(det(Cov_Exp)) + tr(obs_cov*inv(Cov_Exp)) -
                  log(det(obs_cov)) - n_man + penalty*sum(transpose(parameters)[reg_vec].^2)
      return F_ML
end
# FIML
### to add
