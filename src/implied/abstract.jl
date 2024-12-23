
# vars and params API methods for SemImplied
vars(implied::SemImplied) = vars(implied.ram_matrices)
observed_vars(implied::SemImplied) = observed_vars(implied.ram_matrices)
latent_vars(implied::SemImplied) = latent_vars(implied.ram_matrices)

nvars(implied::SemImplied) = nvars(implied.ram_matrices)
nobserved_vars(implied::SemImplied) = nobserved_vars(implied.ram_matrices)
nlatent_vars(implied::SemImplied) = nlatent_vars(implied.ram_matrices)

params(implied::SemImplied) = params(implied.ram_matrices)
nparams(implied::SemImplied) = nparams(implied.ram_matrices)

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
