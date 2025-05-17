#!/usr/bin/env python
"""
The exact kinetic energy functional for a Fermionic gas in a harmonic confinement potential.
For this system the local density method is exact.

References
----------
[Lawes1979] 
    G.P. Lawes, N.H. March, "Exact local density method for linear harmonic oscillator"
    J. Chem. Phys. 71, 1007-1009 (1979)
    https://doi.org/10.1063/1.438398
[Brack2001]
    M. Brack, B.P. van Zyl,
    "Simple Analytical Particle and Kinetic Energy Densities for a Dilute Fermionic Gas
     in a d-Dimensional Harmonic Trap"
    Phys. Rev. Lett. 86, 1574-1577 (2001)
    https://10.1103/PhysRevLett.86.1574
"""
import numpy
from numpy.polynomial.hermite import Hermite
import scipy.special
    
            
class HarmonicOscillator1D:
    def __init__(self, mass=1.0, omega=1.0, hbar=1.0):
        self.mass = mass
        self.omega = omega
        self.hbar = hbar
    def wavefunction(self, x, n=0):
        """
        Evaluate the n-th eigenfunction of the 1D HO wavefunction on a grid x.
        """
        # The harmonic oscillator wavefunctions only depend on the ratio (m*w/hbar)
        zeta = numpy.sqrt(self.mass*self.omega/self.hbar)
        # Hermite polynomials
        def H(n,y):
            coef = numpy.zeros(n+1)
            coef[n] = 1.0
            Hn = Hermite(coef)
            return Hn(y)

        y = zeta * x
        # see https://en.wikipedia.org/wiki/Quantum_harmonic_oscillator
        prefactor = (1.0/numpy.sqrt(pow(2,n)*scipy.special.factorial(n))
                     * pow(zeta/numpy.pi, 1.0/4.0))
        # ψ
        wfn = (prefactor
               * numpy.exp(-0.5 * pow(y,2))
               * H(n,y)
        )
        # ψ'
        wfn_deriv1 = (
            zeta * prefactor
            * numpy.exp(-0.5 * pow(y,2))
            * (y * H(n,y) - H(n+1,y))
        )
        # ψ''
        wfn_deriv2 = (
            zeta**2 * prefactor
            * numpy.exp(-0.5 * pow(y,2))
            * ((y**2+1) * H(n,y) - 2*y*H(n+1,y) + H(n+2,y))
        )

        return wfn, wfn_deriv1, wfn_deriv2

    def density(self, x, n=0):
        wfn, wfn_deriv1, wfn_deriv2 = self.wavefunction(x, n=n)
        rho = wfn**2
        rho_deriv1 = 2*wfn*wfn_deriv1
        rho_deriv2 = 2*(wfn_deriv1**2 + wfn*wfn_deriv2)
        return rho, rho_deriv1, rho_deriv2

    def kinetic_energy_density(self, x, n=0):
        """
        t(x) = -1/2 ψ ψ'' = 1/2 (ψ')² - 1/4 d²/dx² (ψ²)
        """
        # The harmonic oscillator wavefunctions only depend on the ratio (m*w/hbar)
        zeta = numpy.sqrt(self.mass*self.omega/self.hbar)
        y = zeta * x
        # Hermite polynomials
        def H(n,y):
            coef = numpy.zeros(n+1)
            coef[n] = 1.0
            Hn = Hermite(coef)
            return Hn(y)

        prefactor = (1.0/numpy.sqrt(pow(2,n)*scipy.special.factorial(n))
                     * pow(zeta/numpy.pi, 1.0/4.0))
        tn = (
            -pow(self.hbar,2)/(2.0*self.mass) *
            pow(zeta,2) *
            pow(prefactor, 2) *
            numpy.exp(-pow(y,2)) *
            H(n,y) *
            ((pow(y,2)+1) * H(n,y) - 2*y*H(n+1,y) + H(n+2,y))
            )

        return tn

    def kinetic_energy_density_positive(self, x, n=0):
        """
        positive definite kinetic energy density

        tp(x) = 1/2 ψ' ψ'

        It differs from -1/2 ψ ψ''by the term (ψψ)'' which integrates to 0.
        """
        # The harmonic oscillator wavefunctions only depend on the ratio (m*w/hbar)
        zeta = numpy.sqrt(self.mass*self.omega/self.hbar)
        y = zeta * x
        # Hermite polynomials
        def H(n,y):
            coef = numpy.zeros(n+1)
            coef[n] = 1.0
            Hn = Hermite(coef)
            return Hn(y)

        prefactor = (1.0/numpy.sqrt(pow(2,n)*scipy.special.factorial(n))
                     * pow(zeta/numpy.pi, 1.0/4.0))
        tn = (
            pow(self.hbar,2)/(2.0*self.mass) *
            pow(zeta,2) *
            pow(prefactor, 2) *
            numpy.exp(-pow(y,2)) *
            pow(y*H(n,y) - H(n+1,y), 2)
            )

        return tn

    
    
class FermiGasHarmonicPotential:
    def __init__(self,
                 mass=1.0,
                 omega=1.0,
                 hbar=1.0,
                 number_of_electrons=10):
        # Harmonic oscillator
        self.mass = mass
        self.omega = omega
        self.hbar = hbar
        # number of Fermions
        self.number_of_electrons = number_of_electrons
        # The N electrons occupied the lowest N eigenstates of
        # the harmonic oscillator.
        self.HO = HarmonicOscillator1D(mass, omega, hbar)
        
    def electron_density(self, x):
        rho = 0.0*x
        rho_deriv1 = 0.0*x
        rho_deriv2 = 0.0*x
        for n in range(0, self.number_of_electrons):
            rho_n, rho_n_deriv1, rho_n_deriv2 = self.HO.density(x, n)
            rho += rho_n
            rho_deriv1 += rho_n_deriv1
            rho_deriv2 += rho_n_deriv2
        return rho, rho_deriv1, rho_deriv2

    def kinetic_energy_density(self, x):
        ked = 0.0*x
        for n in range(0, self.number_of_electrons):
            ked += self.HO.kinetic_energy_density(x, n)
        return ked
    
    def kinetic_energy_density_positive(self, x):
        ked = 0.0*x
        for n in range(0, self.number_of_electrons):
            ked += self.HO.kinetic_energy_density_positive(x, n)
        return ked

    def kinetic_energy_density_average(self, x):
        """
        Kinetic energy density of Eqn. (7) in [Brack2001]
        """
        ked = 0.0*x
        for n in range(0, self.number_of_electrons):
            ked += 0.5 * (self.HO.kinetic_energy_density_positive(x, n) +
                          self.HO.kinetic_energy_density(x, n))
        return ked
    
    def total_energy_exact(self):
        energy = 0.0
        for n in range(0, self.number_of_electrons):
            energy += self.hbar * self.omega * (n + 1.0/2.0)
        return energy

    def integrated_density(self, x):
        # differential
        dx = numpy.ediff1d(x, to_end=x[-1]-x[-2])
        rho, _, _ = self.electron_density(x)
        electrons = numpy.sum(rho*dx)
        return electrons
    
    def integrated_total_energy(self, x):
        rho, _, _ = self.electron_density(x)
        # potential energy
        v = 0.5 * self.mass * pow(self.omega * x, 2)
        # kinetic energy density
        t = self.kinetic_energy_density(x)

        # differential
        dx = numpy.ediff1d(x, to_end=x[-1]-x[-2])

        energy = numpy.sum( (t + v*rho)*dx )

        return energy

    def kinetic_energy_density_vW(self, x):
        """
        t(x) = 1/8 (ρ' ρ')/ρ - 1/4 ρ''
        """
        rho, rho_deriv1, rho_deriv2 = self.electron_density(x)
        t_vW = 1.0/8.0 * rho_deriv1**2 / rho - 1.0/4.0 * rho_deriv2

        return t_vW

    def kinetic_energy_density_TF(self, x):
        """
        see Eqn. 33 in [Brack2001]
        """
        rho, _, _ = self.electron_density(x)
        t_TF = (pow(self.hbar,2)/(2.0*self.mass)
                * pow(numpy.pi,2)/12.0
                * pow(rho,3)
        )

        return t_TF

    def kinetic_energy_density_LDA(self, x):
        """
        Eqn. (A4) from [Lawes1979] is evaluated for a range of dt/dρ values.
        Then the curve ρ(dt/dρ) is inverted to give dt/dρ(ρ) and t(ρ) is obtained
        by integrating

            t(ρ) = ∫ dt/dρ(ρ') dρ'

        from ρ'=0 to ρ'=ρ
        """
        # Hermite polynomials
        def H(n,y):
            coef = numpy.zeros(n+2)
            coef[n] = 1.0
            Hn = Hermite(coef)
            return Hn(y)

        N = self.number_of_electrons
        normalization = (
            1.0/pow(numpy.pi,1.0/4.0) *
            numpy.sqrt(1.0/(pow(2,N-1)*scipy.special.factorial(N-1)))
            )
        
        def density_vs_dtdrho(dt_drho):
            y = numpy.sqrt(2*N - 2*dt_drho)
            rho = pow(normalization,2) * numpy.exp(-y**2) * (
                N * pow(H(N-1,y),2) +
                2*pow(N-1,2) * pow(H(N-2,y),2) -
                2*(N-1) * y * H(N-1,y) * H(N-2,y))
            return rho

        dt = numpy.linspace(-10.0, 10.0, 5000)
        rho = density_vs_dtdrho(dt)

        plt.plot(rho, dt)
        plt.show()
                   
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    for N in [10]: #[1,2,3]: #,5,10,20]:
        fermi_gas = FermiGasHarmonicPotential(number_of_electrons=N, omega=1.0)
        x = numpy.linspace(-20.0, 20.0, 4000)

        ### DEBUG
        #fermi_gas.kinetic_energy_density_LDA(x)
        ###
    
        rho, rho_deriv1, rho_deriv2 = fermi_gas.electron_density(x)
        ked = fermi_gas.kinetic_energy_density(x)
        ked_vW = fermi_gas.kinetic_energy_density_vW(x)
        ked_TF = fermi_gas.kinetic_energy_density_TF(x)

        print("Total energy (exact) = ", fermi_gas.total_energy_exact())
        print("Total energy         = ", fermi_gas.integrated_total_energy(x))
        print("Numbef of electrons (exact) = ", fermi_gas.number_of_electrons)
        print("Integral of density         = ", fermi_gas.integrated_density(x))
        """
        plt.plot(x, rho, label=r"$\rho(x)$")
        plt.plot(x, ked, label=r"$t(x)$")
        """
    
        l, = plt.plot(rho, ked,
                      label=r"$t(\rho)$ N=%d" % N)
        plt.plot(rho, ked_vW,
                 ls="-.",
                 #color=l.get_color(), lw=2, alpha=0.2,
                 label=r"$t_{vW}(\rho)$ N=%d" % N)
        plt.plot(rho, ked_TF,
                 ls="--",
                 #color=l.get_color(), lw=2, alpha=0.2,
                 label=r"$t_{TF}(\rho)$ N=%d" % N)
        """
        plt.plot(rho, 2.35*rho**3, #1.65*rho**3,
                 ls="--",
                 #color=l.get_color(), lw=2, alpha=0.2,
                 label=r"$c \rho^3$ N=%d" % N)
        """
        """
        plt.plot(rho, ked/ked_vW,
                 ls="--",
                 label=r"$t/t_{vW}$ N=%d" % N)
        """
        
    plt.xlabel(r"$\rho$")
    plt.ylabel(r"$t$")

    plt.legend()
    plt.show()
