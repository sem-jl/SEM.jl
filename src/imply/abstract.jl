
# vars and params API methods for SemImply
vars(imply::SemImply) = vars(imply.ram_matrices)
observed_vars(imply::SemImply) = observed_vars(imply.ram_matrices)
latent_vars(imply::SemImply) = latent_vars(imply.ram_matrices)

nvars(imply::SemImply) = nvars(imply.ram_matrices)
nobserved_vars(imply::SemImply) = nobserved_vars(imply.ram_matrices)
nlatent_vars(imply::SemImply) = nlatent_vars(imply.ram_matrices)

params(imply::SemImply) = params(imply.ram_matrices)
nparams(imply::SemImply) = nparams(imply.ram_matrices)
