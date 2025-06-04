#!/usr/bin/env python
# coding: utf-8
"""
Compute the exchange-correlation energy density matrix xcᵢⱼ(r)
from the 1-particle matrix density

    Dᵢⱼ(r) = n ∫ dr_2 ... ∫ dr_n Ψᵢ(r,r_2,...,r_n) Ψⱼ(r,r_2,...,r_n)

and the 2-particle matrix density

    D2ᵢⱼ(r,r') = n*(n-1) ∫ dr_3 ... ∫ dr_n Ψᵢ(r,r',r_3,...,r_n) Ψⱼ(r,r',r_3,...,r_n)

as

                    D2ᵢⱼ(r,r') - ∑ₖ  Dᵢₖ(r) Dₖⱼ(r')
    xcᵢⱼ(r) = 1/2 ∫ --------------------------------- dr'
                            |r-r'|
"""
import matplotlib.pyplot as plt
import numpy

import pyscf.gto

from msdft.MultistateMatrixDensity import MultistateMatrixDensityFCI

# hydrogen molecule, closed shell
mol = pyscf.gto.M(
    atom = 'H 0 0 -0.35; H 0 0 0.35',
    basis = '6-31g', #'cc-pvdz',
    # singlet
    spin = 0)
nstate = 3

msmd = MultistateMatrixDensityFCI.create_matrix_density(
    mol, nstate=nstate,
    # To compute the xc-energy density we need the pair-density matrix Dᵢⱼ(r,r').
    compute_pair_density=True
)

# Plot exchange-correlation energy density along z-axis
ncoord = 1000
r = numpy.linspace(-1.0, 1.0, ncoord)
coords = numpy.zeros((ncoord, 3))
coords[:,2] = r

xced = msmd.exchange_correlation_energy_density(coords)

for i in range(0, nstate):
    for j in range(i, nstate):
        if i == j:
            linestyle = "-"
        else:
            linestyle = "-."
        plt.plot(r, xced[i,j,:], ls=linestyle, label=r"XC$_{%d,%d}(r)$" % (i,j))

#plt.title("XC-energy density in hydrogen molecule")
plt.xlabel("r ($a_0$)")
plt.ylabel(r"XC$_{IJ}(r)$ ($E_h a_0^{-3}$)")
plt.legend()

#plt.savefig("xc_energy_density.svg")
#plt.savefig("xc_energy_density.png", dpi=300)

plt.show()
