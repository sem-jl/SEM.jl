# ParameterTable interface

Altough you can directly specify a parameter table, this is kind of tedious, so at the moment, we dont have a tutorial for this.
As lavaan also uses parameter tables to store model specifications, we are working on a way to convert lavaan parameter tables to StructuralEquationModels.jl parameter tables, but this is still WIP.

## Convert from and to RAMMatrices

To convert a RAMMatrices object to a ParameterTable, simply use `partable = ParameterTable(rammatrices)`.
To convert an object of type `ParameterTable` to RAMMatrices, you can use `ram_matrices = RAMMatrices(partable)`.