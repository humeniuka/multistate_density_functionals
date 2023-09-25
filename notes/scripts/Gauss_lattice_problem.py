#!/usr/bin/env python
"""
The Gauss lattice problem in 3 dimensions consists in counting the
number of integer lattice points (nx,ny,nz) that lie within a sphere
of radius r,

   N(r): number of points (nx,ny,nz) such that nx^2+ny^2+nz^2 <= r^2
"""
import numpy
import matplotlib.pyplot as plt

rmax = 10
radii = numpy.linspace(0, rmax, 1000)

# The integer coordinates of the lattice points
nx, ny, nz = numpy.mgrid[-rmax:rmax+1, -rmax:rmax+1, -rmax:rmax+1]

# Count the number of grid points inside a sphere of radius r
num_inside_points = numpy.zeros_like(radii)
kinetic_energy = numpy.zeros_like(radii, dtype=float)
for i,r in enumerate(radii):
    num_inside_points[i] = numpy.count_nonzero(nx**2+ny**2+nz**2 <= r**2)
    kinetic_energy[i] = numpy.sum(
        0.5 * numpy.pi**2 * (nx**2+ny**2+nz**2)[nx**2+ny**2+nz**2 <= r**2])
    
plt.xlabel(r"r")
plt.ylabel(r"N(r)")
plt.plot(radii, num_inside_points)
plt.plot(radii, 4.0/3.0 * numpy.pi * radii**3, ls="--", label=r"$4/3 \pi r^3$")

plt.plot(radii, kinetic_energy, label=r"kinetic energy")
# Kinetic energy density according to Thomas-Fermi model.
prefactor = 1.15 # pow(3.0/numpy.pi, 2.0/3.0)
plt.plot(radii, prefactor * pow(num_inside_points, 5.0/3.0), label=r"$n^{5/3}$")


plt.legend()
plt.show()

