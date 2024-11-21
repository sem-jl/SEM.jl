using StructuralEquationModels, Test, Random, SparseArrays, LinearAlgebra
using StructuralEquationModels:
    CommutationMatrix, transpose_linear_indices, duplication_matrix, elimination_matrix

Random.seed!(73721)

n = 4
m = 5

@testset "Commutation matrix" begin
    # transpose linear indices
    A = rand(n, m)
    @test reshape(A[transpose_linear_indices(n, m)], m, n) == A'
    # commutation matrix multiplication
    K = CommutationMatrix(n)
    # test K array interface methods
    @test size(K) == (n^2, n^2)
    @test size(K, 1) == n^2
    @test length(K) == n^4
    nn_linind = LinearIndices((n, n))
    @test K[nn_linind[3, 2], nn_linind[2, 3]] == 1
    @test K[nn_linind[3, 2], nn_linind[3, 2]] == 0

    B = rand(n, n)
    @test_throws DimensionMismatch K * rand(n, m)
    @test K * vec(B) == vec(B')
    C = sprand(n, n, 0.5)
    @test K * vec(C) == vec(C')
    # lmul!
    D = sprand(n^2, n^2, 0.1)
    E = copy(D)
    F = Matrix(E)
    lmul!(K, D)
    @test D == K * E
    @test Matrix(D) == K * F
end

@testset "Duplication / elimination matrix" begin
    A = rand(m, m)
    A = A * A'

    # dupication
    D = duplication_matrix(m)
    @test D * A[tril(trues(size(A)))] == vec(A)

    # elimination
    E = elimination_matrix(m)
    @test E * vec(A) == A[tril(trues(size(A)))]
end
