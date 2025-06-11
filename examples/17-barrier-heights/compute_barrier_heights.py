#!/usr/bin/env python
# coding: utf-8
"""
Barrier heights and reaction energies with MSDFT

The electronic matrix density is computed with CASSCF using a small active space,
while the Hamiltonian matrix in the subspace of the electronic states is computed
as the sum of
    - the kinetic energy of the CASSCF wavefunctions Tᵢⱼ
    - the Hartree matrix Jᵢⱼ[D(r)]
    - the nuclear attraction Vᵢⱼ[D(r)]
    - and the MSDFT exchange-correlation functional evaluated non-self-consistently
      on the CASSCF matrix density, -Xᵢⱼ[D(r)] + Cᵢⱼ[D(r)]
The final energies are obtained as the eigenvalues of the Hamiltonian.

The molecular geometries for the educts, products and transition states as well as the
reference energies for the barrier heights and reaction energies are taken from the BH9
set [1].

References:
-----------
    [1] Prasad, Viki Kumar, et al.
    "Correction to 'BH9, a New Comprehensive Benchmark Data Set for Barrier Heights and Reaction Energies:
    Assessment of Density Functional Approximations and Basis Set Incompleteness Potentials'."
    J. Chem. Theory Comput. 18.6 (2022): 4041-4044.
"""
import numpy
import scipy.linalg

import pyscf.gto

# Chachiyo correlation
from msdft.ElectronRepulsionOperators import LDACorrelationLikeFunctional
# Becke 1988 exchange
from msdft.ElectronRepulsionOperators import GGABecke88ExchangeLikeFunctional
from msdft.MultistateMatrixDensity import MultistateMatrixDensityCASSCF

def casscf_matrix_density(xyz_file: str) -> MultistateMatrixDensityCASSCF:
    """
    Compute matrix density for the molecule in the xyz_file with CASSCF
    """
    # Read charge and spin multiplicity from the comment line
    with open(xyz_file, 'r') as fxyz:
        # skip 1st line
        fxyz.readline()
        # read comment lint
        words = fxyz.readline().split()
    charge, multiplicity = int(words[0]), int(words[1])
    # multiplicity = 2*S+1
    S = (multiplicity-1)/2
    spin = int(2*S)

    # create molecule
    mol = pyscf.gto.M(atom=xyz_file, basis='cc-pvdz', charge=charge, spin=spin)
    # compute matrix density with CASSCF(2e,2o) for singlet states
    # or CASSCF(1e,2o) for doublet states
    if multiplicity == 2:
        nelecas = 1
    else:
        nelecas = 2
    msmd = MultistateMatrixDensityCASSCF.create_matrix_density(
        mol, nstate=2, ncas=2, nelecas=nelecas
    )

    return msmd

def msdft_hamiltonian(msmd: MultistateMatrixDensityCASSCF):
    """
    Evaluate MSDFT Hamiltonian matrix for matrix density obtained from
    a CASSCF calculation,

    Hᵢⱼ = Tᵢⱼ[{Ψ}] + Jᵢⱼ[D(r)] + Vᵢⱼ[D(r)] + XCᵢⱼ[D(r)] + N δᵢⱼ

    The kinetic energy is computed from the wavefunctions instead of the densities.
    """
    # GGA functional for exchange
    exchange_gga = GGABecke88ExchangeLikeFunctional(msmd.mol, level=4)
    # LDA functionals for correlation
    correlation_lda = LDACorrelationLikeFunctional(msmd.mol, level=4)

    # The kinetic energy is evaluated exactly from the CASSCF wavefunctions
    # since we lack an accurate orbital-free kinetic energy functional.
    T = msmd.exact_kinetic_energy()
    # Hartree-term Jᵢⱼ[D(r)]
    J = msmd.hartree_matrix_product()
    # nuclear attraction Vᵢⱼ[D(r)]
    V = msmd.nuclear_attraction_energy()
    # exchange-correlation energy
    XC = -exchange_gga(msmd) + correlation_lda(msmd)
    # repulsion between nuclei
    N = msmd.mol.energy_nuc() * numpy.eye(msmd.number_of_states)

    # Total electronic Hamiltonian in the subspace
    H = T + J + V + XC + N

    ### DEBUG
    print("XC")
    print(XC)
    print("H")
    print(H)
    ###

    return H

def msdft_energies(msmd):
    H = msdft_hamiltonian(msmd)
    # Diagonalize Hamiltonian
    eigvals, _ = scipy.linalg.eigh(H)
    return eigvals


if __name__ == "__main__":
    # for unit conversion
    hartree_to_kcalmol = 627.509469

    print("Reference (SI of Prasad et al. BH9 dataset)")
    barrier_forward_ref = 11.74
    barrier_reverse_ref = 9.01
    reaction_energy_ref = 2.74
    print(f"forward barrier : {barrier_forward_ref} kcal/mol")
    print(f"reverse barrier : {barrier_reverse_ref} kcal/mol")
    print(f"reaction energy : {reaction_energy_ref} kcal/mol")

    # Matrix densities
    print("CASSCF calculation ...")
    print(" ... on reactant")
    reactant_msmd = casscf_matrix_density('01_1R.xyz')
    print(" ... on product")
    product_msmd = casscf_matrix_density('01_1P.xyz')
    print(" ... on transition state")
    ts_msmd = casscf_matrix_density('01_1TS.xyz')

    print("CASSCF")
    barrier_forward_casscf = ts_msmd.eigenenergies[0] - reactant_msmd.eigenenergies[0]
    barrier_reverse_casscf = ts_msmd.eigenenergies[0] - product_msmd.eigenenergies[0]
    reaction_energy_casscf = product_msmd.eigenenergies[0] - reactant_msmd.eigenenergies[0]

    print(f"forward barrier : {barrier_forward_casscf*hartree_to_kcalmol} kcal/mol")
    print(f"reverse barrier : {barrier_reverse_casscf*hartree_to_kcalmol} kcal/mol")
    print(f"reaction energy : {reaction_energy_casscf*hartree_to_kcalmol} kcal/mol")

    print("MSDFT")
    print("MSDFT energies are evaluated on CASSCF matrix densities ...")
    print(" ... of reactant")
    reactant_energies = msdft_energies(reactant_msmd)
    print(" ... of product")
    product_energies = msdft_energies(product_msmd)
    print(" ... of transition state")
    ts_energies = msdft_energies(ts_msmd)
 
    barrier_forward_msdft = ts_energies[0] - reactant_energies[0]
    barrier_reverse_msdft = ts_energies[0] - product_energies[0]
    reaction_energy_msdft = product_energies[0] - reactant_energies[0]

    print(f"forward barrier : {barrier_forward_msdft*hartree_to_kcalmol} kcal/mol")
    print(f"reverse barrier : {barrier_reverse_msdft*hartree_to_kcalmol} kcal/mol")
    print(f"reaction energy : {reaction_energy_msdft*hartree_to_kcalmol} kcal/mol")
