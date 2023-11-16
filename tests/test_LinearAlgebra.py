#!/usr/bin/env python
# coding: utf-8
import unittest

import numpy
import numpy.testing

from msdft.LinearAlgebra import eigensystem_derivatives

class TestLinearAlgebra(unittest.TestCase):
    def check_eigensystem_derivatives(
          self, 
          matrix_function,
          matrix_function_derivative,
          x):
        """
        Compare numerical and analytical derivatives of eigenvalues and
        eigenvectors of a matrix function D(x) at the point x.

        :param matrix_function: matrix function D(x) that maps a scalar
            to a symmetric (n,n) matrix
        :type matrix_function: callable

        :param matrix_function_derivative: 
            function that evalutes the derivative dD/dx of the matrix function
        :type matrix_function: callable

        :param x: argument of matrix function for which gradients are
            evaluated.
        :type x: float
        """
        # D(x)
        D = matrix_function(x)
        # dD/dx(x)
        grad_D = matrix_function_derivative(x)
        L_ref, U_ref = numpy.linalg.eigh(D)
        # A) Compute derivatives of eigenvalues and eigenvectors numerically.
        # displacement for finite differences
        h = 0.001
        # D(x+h)
        D_plus = matrix_function(x + h)
        # eigenvalues and eigenvectors of D(x+h)
        L_plus, U_plus = numpy.linalg.eigh(D_plus)
        # D(x-h)
        D_minus = matrix_function(x - h)
        # eigenvalues and eigenvectors of D(x+h)
        L_minus, U_minus = numpy.linalg.eigh(D_minus)
        
        # finite difference derivatives
        grad_D_numerical = (D_plus - D_minus)/(2*h)
        grad_L_numerical = (L_plus - L_minus)/(2*h)
        grad_U_numerical = (U_plus - U_minus)/(2*h)

        # B) Compute derivatives of eigenvalues and eigenvectors analytically.
        L, U, grad_L, grad_U = eigensystem_derivatives(D, grad_D[:,:,numpy.newaxis])

        # Eigenvectors are not uniquely defined, since their phases (+1 or -1)
        # are arbitrary.
        def compare_eigenvectors(U_ref, U, **kwds):
            dim, _ = U_ref.shape
            # Check dimensions.
            self.assertEqual((dim, dim), U.shape)
            for i in range(0, dim):
                # Compute the phase by which the eigenvectors differ.
                sign = numpy.sign(numpy.dot(U_ref[:,i], U[:,i]))
                # Compare the aligned eigenvectors.
                numpy.testing.assert_almost_equal(U_ref[:,i], sign * U[:,i], **kwds)

        # Check that derivative of matrix function is implemented correctly.
        numpy.testing.assert_almost_equal(grad_D_numerical, grad_D, decimal=6)

        # Check that the matrix is diagonalized correctly.
        numpy.testing.assert_almost_equal(L_ref, L)
        # If there are repeated eigenvalues, the eigenvectors in the degenerate subspace
        # can be mixed arbitrarily. Eigenvectors are only compared, if all eigenvalues
        # are distinct.
        if len(numpy.unique(L)) == len(L):
            # All eigenvectors are distinct.
            compare_eigenvectors(U_ref, U)
        # Check that the derivatives agree.
        numpy.testing.assert_almost_equal(grad_L_numerical, grad_L[:,0], decimal=6)
        compare_eigenvectors(grad_U_numerical, grad_U[:,:,0], decimal=5)

    def test_eigensystem_derivatives_2x2_distinct_eigenvalues(self):
        """
        Test eigenvalue derivatives for a 2x2 matrix with distinct eigenvalues.
        """
        def matrix_function(x):
            s = numpy.sin(x)
            D = numpy.array([
                [1.0, s],
                [s,  2.0]
            ])
            return D

        def matrix_function_derivative(x):
            c = numpy.cos(x)
            grad_D = numpy.array([
                [0.0, c],
                [c,  0.0]
            ])
            return grad_D

        # At x=0, D = [[1, 0], [0, 2]]
        for x0 in [0.0, 0.8, numpy.pi/2.0-0.1]:
            with self.subTest(x0=x0):
                self.check_eigensystem_derivatives(
                    matrix_function, matrix_function_derivative, x0)

    def test_eigensystem_derivatives_2x2_repeated_eigenvalues(self):
        """
        Test eigenvalue derivatives for a 2x2 matrix with same eigenvalues.
        """
        def matrix_function(x):
            s = numpy.sin(x)
            D = numpy.array([
                [1.0, s],
                [s,  1.0]
            ])
            return D

        def matrix_function_derivative(x):
            c = numpy.cos(x)
            grad_D = numpy.array([
                [0.0, c],
                [c,  0.0]
            ])
            return grad_D

        # At x=0, D = [[1, 0], [0, 1]]
        # and ∇U = [[0, 0], [0, 0]]
        x0 = 0.0
        D = matrix_function(x0)
        grad_D = matrix_function_derivative(x0)
        # Compute derivatives of eigenvalues and eigenvectors analytically.
        L, U, grad_L, grad_U = eigensystem_derivatives(D, grad_D[:,:,numpy.newaxis])
        # Compare with expectd eigenvector derivatives.
        numpy.testing.assert_almost_equal(numpy.zeros((2,2)), grad_U[:,:,0])


if __name__ == "__main__":
    unittest.main()
