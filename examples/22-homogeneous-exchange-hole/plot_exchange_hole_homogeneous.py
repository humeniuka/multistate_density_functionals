#!/usr/bin/env python
"""
Plot the exchange hole of the homogeneous electron gas

see
[Becke1983] Becke, A. D.
    "Hartree-Fock exchange energy of an inhomogeneous electron gas."
    International journal of quantum chemistry 23.6 (1983): 1915-1922.
    doi:10.1002/qua.560230605
"""
import matplotlib.pyplot as plt
import numpy

def exchange_hole_homogeneous(density, s):
    """
    Exchange hole of the homogeneous electron gas with density `density`
    as a function of the distance s from the reference point.
    """
    # first order Bessel function
    def j1(x):
        return numpy.sin(x)/pow(x,2) - numpy.cos(x)/x
    # Fermi momentum
    kF = pow(6 * numpy.pi**2 * density, 1.0/3.0)
    x = kF*s
    rho_x_hom = 9*density * (j1(x)/x)**2
    return -rho_x_hom

s = numpy.linspace(1.0e-5, 5.0, 1000)
for density in [0.1, 1.0, 2.0, 5.0]:
    rho_x_hom = exchange_hole_homogeneous(density, s)
    plt.plot(s, rho_x_hom, label=r"$\rho=%s$" % density)

plt.legend()
plt.xlabel("s / $a_0$")
plt.ylabel("Exchange hole (homogeneous)")

#plt.savefig("exchange_hole_homogeneous.png", dpi=300)
#plt.savefig("exchange_hole_homogeneous.svg")

plt.show()
