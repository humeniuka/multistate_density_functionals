#!/usr/bin/env python
# coding: utf-8
"""
Compare the exact electron repulsion energy matrix between electronic states

  Cᵢⱼ = ∫dx1 ∫dx2...∫dxn Ψ*ᵢ(x1,x2,...,xn) ∑ᵦ<ᵧ 1/|rᵦ-rᵧ| Ψⱼ(x1,x2,...,xn)

with the multi-state local-density approximation

  Cᵢⱼ ≈ Jᵢⱼ[D] - Kᵢⱼ[D]
      = 1/2 ∑ₖ ∫∫' Dᵢₖ(r) Dₖⱼ(r')/|r-r'| - 2¹ᐟ³ Cₓ ∫ [Dᵅ(r)⁴ᐟ³]ᵢⱼ + [Dᵝ(r)⁴ᐟ³]ᵢⱼ dr

for the water molecule at its equilibrium geometry.
The exact matrix density Dᵢⱼ(r) is calculated using full configuration interaction.
"""
import numpy

import pyscf.fci
import pyscf.scf

from msdft.ElectronRepulsionOperators import HartreeLikeOperatorFunctional
from msdft.ElectronRepulsionOperators import LSDAExchangeLikeFunctional

from msdft.MultistateMatrixDensity import MultistateMatrixDensityFCI


def compare_electron_repulsion_energies(mol, nstate=2):
    """
    The electron repulsion energy matrix is computed exactly
    and approximately using a local functional consisting of
    a Hartree-like and an LSDA-exchange-like term.

    :param nstate: Number of electronic states in the subspace.
       The full CI problem is solved for the lowest nstate states.
    :type nstate: int > 0
    """
    # compute D(r) from full CI
    msmd = MultistateMatrixDensityFCI.create_matrix_density(mol, nstate=nstate)
    # Functionals for parts of electron repulsion.
    # J[D]
    hartree_functional = HartreeLikeOperatorFunctional(mol)
    # K[D]
    exchange_functional = LSDAExchangeLikeFunctional(mol)

    # exact electron repulsion
    # Cᵢⱼ = ∫dx1 ∫dx2...∫dxn Ψ*ᵢ(x1,x2,...,xn) ∑ᵦ<ᵧ 1/|rᵦ-rᵧ| Ψⱼ(x1,x2,...,xn)
    C_exact = msmd.exact_electron_repulsion()
    # approximate multi-state LSDA electron repulsion
    # Cᵢⱼ ≈ Jᵢⱼ[D] - Kᵢⱼ[D]
    J_Hartree = hartree_functional(msmd)
    K_LSDA = exchange_functional(msmd)
    C_approximate = J_Hartree - K_LSDA
    #
    print("=== Electron Repulsion Matrices ===")
    print("C_exact")
    print(C_exact)
    print("C_approximate")
    print(C_approximate)
    print("J Hartree")
    print(J_Hartree)
    print("-K_LSDA")
    print(-K_LSDA)

    # relative errors
    relative_errors = abs(C_approximate - C_exact)/(abs(C_exact) + 1.0e-8)
    print("=== Relative Errors ===")
    print("|C(approximate)-C(exact)|/|C(exact)|")
    print(relative_errors)


if __name__ == "__main__":
    # water
    mol = pyscf.gto.M(
        atom = 'O  0 0 0; H 0.75 0.00 0.50; H 0.75 0.00 -0.50',
        basis = 'sto-3g',
        charge = 0,
        # singlet
        spin = 0)

    compare_electron_repulsion_energies(mol, nstate=4)
