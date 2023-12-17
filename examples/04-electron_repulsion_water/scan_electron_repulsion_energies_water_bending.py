#!/usr/bin/env python
# coding: utf-8
"""
compare the exact electron repulsion energy matrix between electronic states

  Cᵢⱼ = ∫dx1 ∫dx2...∫dxn Ψ*ᵢ(x1,x2,...,xn) ∑ᵦ<ᵧ 1/|rᵦ-rᵧ| Ψⱼ(x1,x2,...,xn)

with the multi-state local-density approximation

  Cᵢⱼ ≈ Jᵢⱼ[D] - Kᵢⱼ[D]
      = 1/2 ∑ₖ ∫∫' Dᵢₖ(r) Dₖⱼ(r')/|r-r'| - 2¹ᐟ³ Cₓ ∫ [Dᵅ(r)⁴ᐟ³]ᵢⱼ + [Dᵝ(r)⁴ᐟ³]ᵢⱼ dr

for a range of the water geometries with different HOH angles.
The exact matrix density Dᵢⱼ(r) is calculated using full configuration interaction.
"""
import json
import numpy

import pyscf.fci
import pyscf.scf

from msdft.ElectronRepulsionOperators import HartreeLikeOperatorFunctional
from msdft.ElectronRepulsionOperators import LSDAExchangeLikeFunctional

from msdft.MultistateMatrixDensity import MultistateMatrixDensityFCI

if __name__ == "__main__":
    print("This calculation can take a while. Please be patient ...")
    # Save exact and approximate electron repulsion energies and other
    # data along the scan.
    scan_data = {
        # HOH angle, the scan variable
        'angle': [],
        # Total energy (electronic plus nuclear repulsion).
        'eigenenergies': [],
        # electron repulsion matrix
        'C_exact': [],
        # approximate electron repulsion matrix and its constituents.
        'C_approximate': [],
        'J_Hartree': [],
        'K_LSDA': []
    }

    # water
    # Compute lowest 4 singlet states.
    nstate = 4
    # To fix the signs of the off-diagonal elements of the matrix densities,
    # so that we can plot smooth, continuous curves, the global phases of
    # the wavefunctions have to be aligned with the phases at the previous
    # scan point (the reference).
    msmd_ref = None
    # experimental OH bond length of water (NIST) in Å
    rOH = 0.958
    # The HOH angle is scanned from 90° to 180°
    angles = numpy.linspace(numpy.pi/2.0, numpy.pi, 20)
    for angle in angles:
        print(rf"* HOH angle: {angle * 180.0/numpy.pi:4f}°")

        xH2 = rOH * numpy.cos(numpy.pi - angle)
        yH2 = rOH * numpy.sin(numpy.pi - angle)
        mol = pyscf.gto.M(
            atom = f'O  0 0 0; H {-rOH} 0 0; H {xH2} {yH2} 0',
            # minimal basis set
            basis = 'sto-3g',
            charge = 0,
            # singlet
            spin = 0)

        # Compute D(r) from full CI.
        msmd = MultistateMatrixDensityFCI.create_matrix_density(mol, nstate=nstate)
        # Align the signs of the wavefunctions with those at the privious scan point.
        # At the first scan point there is no reference.
        if msmd_ref is not None:
            msmd.align_phases(msmd_ref)
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

        # Compare approximate and exact electron repulsion.
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

        # Save data for later plotting
        scan_data['angle'].append(angle)
        scan_data['eigenenergies'].append(msmd.eigenenergies.tolist())
        scan_data['C_exact'].append(C_exact.tolist())
        scan_data['C_approximate'].append(C_approximate.tolist())
        scan_data['J_Hartree'].append(J_Hartree.tolist())
        scan_data['K_LSDA'].append(K_LSDA.tolist())

        # This scan point becomes the reference for the next scan point.
        msmd_ref = msmd

        # Save intermediate results, in case the calculation is stopped early.
        with open('electron_repulsion_energies_water_bending.json', 'w') as filehandle:
            json.dump(scan_data, filehandle)
