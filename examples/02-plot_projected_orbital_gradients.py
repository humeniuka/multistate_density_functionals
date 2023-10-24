#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
After adding additional contractions with angular momenta L-1 and L+1,
the gradients of atomic orbitals with respect to the electronic coordinates
can be represented exactly in the basis set.

The plot compares the exact gradient (solid) with its projection on the basis set (dashed).
"""

import matplotlib.pyplot as plt
import numpy
import pyscf.gto

from msdft.GradientAugmentedBasis import orbital_gradient_projection


mol = pyscf.gto.Mole()
mol.build(
    atom = 'Li 0 0 -0.55; Li 0 0 0.55',  # in Angstrom
    basis = '6-31g',
    symmetry = False,
    cart = True,
)

# Plot gradients of orbitals along the z-axis.
Ncoord = 2000
r = numpy.linspace(-5.0, 5.0, Ncoord)
coords = numpy.zeros((Ncoord, 3))
coords[:,2] = r

grad_ao, projected_grad_ao = orbital_gradient_projection(mol, coords, level=8)

# compare ∇a and ∇a_projected(r) = ∑ₘ∑ₙ <r|m> S⁻¹ₘₙ <n|∇a>
# Only the z-component of the gradient is shown.
xyz = 2
ao_labels = mol.ao_labels(fmt=False)

number_of_orbitals = grad_ao.shape[2]
# Loop over atomic orbitals in the 6-31g basis set.
for orb in range(0, number_of_orbitals):
    atom_id, symbol, nl_str, str_of_AO_notation = ao_labels[orb]
    label = "%3s(%d) %s%-4s" % (symbol, atom_id+1, nl_str, str_of_AO_notation)
    # Only plot the p-orbitals on the first atom.
    if 'p' in label:
        # Plot the exact gradients
        line, = plt.plot(r, grad_ao[xyz,:,orb], lw=3, alpha=0.5, label=label)
        # Plot the projected gradients.
        if orb == number_of_orbitals-1:
            plt.plot(
                r, projected_grad_ao[xyz,:,orb],
                color=line.get_color(), ls="--", label='projected')
        else:
            plt.plot(
                r, projected_grad_ao[xyz,:,orb],
                color=line.get_color(), ls="--")

plt.xlabel('r / $a_0$')
plt.ylabel(r'Orbital gradient $\frac{\partial}{\partial_z} AO(x,y,z)$')

plt.legend(title='AOs')

#plt.savefig("projected_orbital_gradients.svg")
#plt.savefig("projected_orbital_gradients.png", dpi=300)

plt.show()
