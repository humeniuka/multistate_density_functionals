#!/usr/bin/env python
# coding: utf-8
"""
compare

  tr(diag(D)⁴ᐟ³) = ∑ᵢ Dᵢᵢ⁴ᐟ³

with

  tr(D⁴ᐟ³) = ∑ᵢ λᵢ⁴ᐟ³

where λᵢ are the eigenvalues of the symmetric, positive-definite matrix D.
Because of Klein's inequality (see https://en.wikipedia.org/wiki/Trace_inequality)

  tr(D⁴ᐟ³) ≥ tr(diag(D)⁴ᐟ³)
"""
import numpy
import scipy.linalg

dim = 5

# Create a random antisymmetric matrix Xᵀ = -X
X = numpy.random.rand(dim, dim)
X = 0.5 * (X - X.T)

# random orthogonal matrix with eigenvectors
U = scipy.linalg.expm(X)

# random positive eigenvalues
eigvals = 5.0 * numpy.random.rand(dim)

# random symmetric, positive-definite matrix
D = numpy.einsum('a,ia,ja->ij', eigvals, U, U)

# The function f(x) = x⁴ᐟ³ is applied to each element of D
# before taking the trace, ∑ᵢ Dᵢᵢ⁴ᐟ³
element_wise = 0.0
# The function f(x) = x⁴ᐟ³ is applied as a matrix function
# f(D) = D⁴ᐟ³. The trace tr(D⁴ᐟ³) is computed in the basis
# of eigenvectors of D, where it is just ∑ᵢ λᵢ⁴ᐟ³.
matrix_wise = 0.0
for i in range(0, dim):
    element_wise += pow(D[i,i], 4.0/3.0)
    matrix_wise += pow(eigvals[i], 4.0/3.0)

print(f"tr(diag(D)⁴ᐟ³) = ∑ᵢ Dᵢᵢ⁴ᐟ³ = {element_wise}")
print(f"tr(D⁴ᐟ³)       = ∑ᵢ λᵢ⁴ᐟ³  = {matrix_wise}")

assert (matrix_wise > element_wise)
