
# vars and params API methods for SemImply
vars(imply::SemImply) = vars(imply.ram_matrices)
observed_vars(imply::SemImply) = observed_vars(imply.ram_matrices)
latent_vars(imply::SemImply) = latent_vars(imply.ram_matrices)

nvars(imply::SemImply) = nvars(imply.ram_matrices)
nobserved_vars(imply::SemImply) = nobserved_vars(imply.ram_matrices)
nlatent_vars(imply::SemImply) = nlatent_vars(imply.ram_matrices)

params(imply::SemImply) = params(imply.ram_matrices)
nparams(imply::SemImply) = nparams(imply.ram_matrices)

# checks if the A matrix is acyclic
# wraps A in LowerTriangular/UpperTriangular if it is triangular
function check_acyclic(A::AbstractMatrix; verbose::Bool = false)
    # check if A is lower or upper triangular
    if istril(A)
        verbose && @info "A matrix is lower triangular"
        return LowerTriangular(A)
    elseif istriu(A)
        verbose && @info "A matrix is upper triangular"
        return UpperTriangular(A)
    else
        # check if non-triangular matrix is acyclic
        acyclic = isone(det(I - A))
        if acyclic
            verbose && @info "The matrix is acyclic. Reordering variables in the model to make the A matrix either Upper or Lower Triangular can significantly improve performance.\n" maxlog =
                1
        end
        return A
    end
end
