#!/usr/bin/env python
"""
Check that the coefficient vectors f_A(zeta) are normalized to 1 for all values of zeta.
"""
import numpy

def gaussian(sigma, mu, x):
    g = 1.0/(sigma * numpy.sqrt(2*numpy.pi)) * numpy.exp(-0.5 * pow(x - mu,2)/pow(sigma,2))
    return g

def coefficients_F(L,zeta):
    A = numpy.array(range(1,L+1))
    g = gaussian(zeta*L, L/2.0, A)
    f = numpy.sqrt(g) / numpy.sqrt(numpy.sum(g))
    return f

fA = coefficients_F(11, 0.1)
print(fA)
print(numpy.sum(fA**2))
fA = coefficients_F(11, 10.0)
print(fA)
print(numpy.sum(fA**2))
