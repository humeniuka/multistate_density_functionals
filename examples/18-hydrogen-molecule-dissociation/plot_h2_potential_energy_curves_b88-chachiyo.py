#!/usr/bin/env python
# coding: utf-8
"""
Compute the potential energy curves of H2 as a function of the bond length using MSDFT.

The electronic matrix density is computed with full configuration interation,
while the Hamiltonian matrix in the subspace of the electronic states is computed
as the sum of
    - the kinetic energy of the CASSCF wavefunctions Tᵢⱼ
    - the Hartree matrix Jᵢⱼ[D(r)]
    - the nuclear attraction Vᵢⱼ[D(r)]
    - and the MSDFT exchange-correlation functional evaluated non-self-consistently
      on the CASSCF matrix density, -Xᵢⱼ[D(r)] + Cᵢⱼ[D(r)]
The final energies are obtained as the eigenvalues of the Hamiltonian.

"""
import matplotlib.pyplot as plt
import numpy
import scipy.linalg
from tqdm import tqdm

import pyscf.gto

# Chachiyo correlation
from msdft.ElectronRepulsionOperators import LDACorrelationLikeFunctional
# Becke 1988 exchange
from msdft.ElectronRepulsionOperators import GGABecke88ExchangeFunctional
from msdft.MultistateMatrixDensity import MultistateMatrixDensityFCI


def msdft_hamiltonian(msmd: MultistateMatrixDensityFCI):
    """
    Evaluate MSDFT Hamiltonian matrix for matrix density obtained from
    a CASSCF calculation,

    Hᵢⱼ = Tᵢⱼ[{Ψ}] + Jᵢⱼ[D(r)] + Vᵢⱼ[D(r)] + XCᵢⱼ[D(r)] + N δᵢⱼ

    The kinetic energy is computed from the wavefunctions instead of the densities.
    """
    # GGA functional for exchange
    exchange_gga = GGABecke88ExchangeFunctional(msmd.mol, level=4)
    # LDA functionals for correlation
    correlation_lda = LDACorrelationLikeFunctional(msmd.mol, level=4)

    # The kinetic energy is evaluated exactly from the CASSCF wavefunctions
    # since we lack an accurate orbital-free kinetic energy functional.
    T = msmd.exact_kinetic_energy()
    # Hartree-term Jᵢⱼ[D(r)]
    J = msmd.hartree_matrix_product()
    # nuclear attraction Vᵢⱼ[D(r)]
    V = msmd.nuclear_attraction_energy()
    # repulsion between nuclei
    N = msmd.mol.energy_nuc() * numpy.eye(msmd.number_of_states)
    # correlation energy
    C = correlation_lda(msmd)
    # exchange energy
    X = -exchange_gga(msmd)
    # exchange-correlation energy
    XC = X + C

    # Total electronic Hamiltonian in the subspace
    H = T + J + V + XC + N

    return H

def msdft_energies(msmd):
    H = msdft_hamiltonian(msmd)
    # Diagonalize Hamiltonian
    eigvals, _ = scipy.linalg.eigh(H)
    return eigvals


if __name__ == "__main__":
    # number of electronic states (of any spin)
    nstate = 10
    # number of bond lengths
    npts = 30
    # H-H bond lengths in Ang
    bond_lengths = numpy.linspace(0.2, 3.0, npts)

    # adiabatic eigenergies
    energies_fci = numpy.zeros((npts, nstate))
    energies_msdft = numpy.zeros((npts, nstate))

    for i, rHH in enumerate(tqdm(bond_lengths)):
        # hydrogen molecule
        mol = pyscf.gto.M(
            atom = f'H 0 0 {-rHH/2.0}; H 0 0 {rHH/2.0}',
            basis = 'cc-pvdz'
        )

        # compute exact matrix density with full CI
        msmd = MultistateMatrixDensityFCI.create_matrix_density(
            mol, nstate=nstate,
            # Spin symmetry is turned off, since we just want the lowest
            # electronic states no matter what spin state.
            spin_symmetry=False)
        energies_fci[i,:] = msmd.eigenenergies
        # evaluate MSDFT Hamiltonian on FCI matrix density and diagonalize
        energies_msdft[i,:] = msdft_energies(msmd)

    # plot potential energy curves
    plt.xlabel(r"r(H-H) / $\AA$")
    plt.ylabel("Potential Energy $E_h$")

    # shift energies to 0 at r=oo
    energies_fci -= energies_fci[-1,0]
    energies_msdft -= energies_msdft[-1,0]

    for s in range(0, nstate):
        line, = plt.plot(
            bond_lengths, energies_fci[:,s],
            ls="-", lw=2, alpha=0.5, label="$S_{%d}$ (FCI)" % s)
        plt.plot(
            bond_lengths, energies_msdft[:,s],
            ls="--", color=line.get_color(), label="$S_{%d}$ (MSDFT@FCI)" % s)

    plt.legend(ncols=2)
    plt.show()
