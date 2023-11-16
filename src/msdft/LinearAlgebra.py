#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy


def eigensystem_derivatives(D, grad_D):
    """
    compute the eigenvalues Λ and eigenvectors U of the
    symmetric matrix D, as well as their gradients with
    respect to some external parameter, ∇Λ and ∇U.

    :param D: symmetric matrix
    :type D: numpy.ndarray of shape (n,n)

    :param grad_D: ∇D, gradient of D w/r/t some p external
        parameters.
    :type grad_D: numpy.ndarray of shape (n,n,p)

    :return:
        L, U, grad_L, grad_U
    :rtype: tuple of numpy.ndarray
        `L` has shape (n), L[j] is the j-th eigenvalue of D.
        `U` has shape (n,n), U[i,j] is the i-th component of the j-th
            eigenvector of D.
        `grad_L` has shape (n,p), grad_L[j,p] is the derivative of the
            j-th eigenvalue of D with respect to the p-th parameter.
        `grad_U` has shape (n,n,p), grad_U[i,j,p] is the derivative
            of the i-th component of the j-th eigenvector of D
            with respect to the p-th parameter.
    """
    # Check dimensions of inputs.
    dimension, _, num_parameters = grad_D.shape
    assert D.shape == (dimension, dimension), "Matrix D has to be square."
    assert grad_D.shape == (dimension, dimension, num_parameters)

    # Compute eigenvalues Λ and eigenvectors U of the symmetric
    # matrix D.
    L, U = numpy.linalg.eigh(D)

    # If uᵢ is a normalized eigenvector of D with eigenvalue λᵢ,
    #   D uᵢ = λᵢ uᵢ
    # then the gradient of the eigenvalue is given by
    #   ∇λᵢ = <uᵢ,∇D uᵢ>
    # where < , > is the scalar product.
    #
    # ∇Λ(r), gradients of eigenvalues of D(r)
    grad_L = numpy.einsum('ki,kld,li->id', U, grad_D, U)

    # Differentiating the eigenvalue equation
    #   D.U = Λ.U   with Λ = diag(λ₁,...,λₙ)
    # gives the matrix equation
    #   ∇U.Λ - D.∇U = ∇D.U - U.∇Λ,
    # which is written component-wise as
    #
    #  ∑ₖ (∇Uᵢₖ Λₖⱼ - Dᵢₖ ∇Uₖⱼ) = ∑ₖ (∇Dᵢₖ Uₖⱼ - Uᵢₖ ∇Λₖⱼ)     i,k,j = 1,...,n
    #
    # If one treats Xₖₗ = ∇Uₖₗ as a vector in ℝ^(n^2), this leads to a
    # system of linear equations for the gradients of the eigenvectors:
    #
    #   A.X = B      or      ∑ₖₗ Aᵢⱼ,ₖₗ Xₖₗ = Bᵢⱼ
    #
    # where
    #
    #   Aᵢⱼ,ₖₗ = δᵢₖ λⱼ δₗⱼ - Dᵢₖ δₗⱼ
    # 
    # and
    #
    #   Bᵢⱼ = ∑ₖ ∇Dᵢₖ Uₖⱼ - Uᵢⱼ ∇λⱼ
    #
    # with the multi indices ij,kl = 1,...,n^2.

    # Reserve space for gradient of eigenvectors
    grad_U = numpy.zeros((dimension, dimension, num_parameters))

    # identity matrix
    delta = numpy.eye(dimension)
    # The matrix A is the same for all partial derivatives.
    dim2 = dimension*dimension
    A = numpy.zeros((dim2,dim2))
    # ij is a multiindex that runs over all combinations of (i,j) (rows of A)
    ij = 0
    for i in range(0, dimension):
        for j in range(0, dimension):
            # kl is a multiindex that runs over all combinations of (k,l) (columns of A)
            kl = 0
            for k in range(0, dimension):
                for l in range(0, dimension):
                    # Aᵢⱼ,ₖₗ = δᵢₖ λⱼ δₗⱼ - Dᵢₖ δₗⱼ
                    A[ij,kl] = delta[i,k] * L[j] * delta[l,j] - D[i,k] * delta[l,j]
                    # increase column counter
                    kl += 1
            # increase row counter
            ij += 1

    # right-hand side of A.X = B
    # Bᵢⱼ = ∑ₖ ∇Dᵢₖ Uₖⱼ - Uᵢⱼ ∇λⱼ 
    B_matrix = numpy.einsum('ikp,kj->ijp', grad_D, U) - numpy.einsum('ij,jp->ijp', U, grad_L)

    # Each partial derivative w/r/t an external parameter can
    # be treated separately.
    for p in range(0, num_parameters):
        # Solve A.X = B for X in a least-square sense in case that A is singular.
        # Turn B into a vector in ℝ^(n^2)
        B = B_matrix[:,:,p].flatten()
        X, _, _, _ = numpy.linalg.lstsq(A, B, rcond=None)
        # Reinterpret the vector X as a square matrix.
        grad_U[:,:,p] = numpy.reshape(X, (dimension, dimension))

    return L, U, grad_L, grad_U
