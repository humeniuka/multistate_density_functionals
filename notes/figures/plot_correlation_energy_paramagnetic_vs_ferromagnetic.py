#!/usr/bin/env python
"""
Compare the paramagnetic and ferromagnetic parts of the correlation
energy of the uniform electron gas
"""
import matplotlib.pyplot as plt
import numpy

plt.style.use('./latex.mplstyle')

# Parameters of Chachiyo's functional from Eqn.(3) of [Chachiyo]
a = (numpy.log(2.0)-1.0)/(2*numpy.pi**2)
# b from Eqn.(3) for the paramagnetic part εᶜ_0
b_paramagnetic = 20.4562557
# b from Eqn.(12) for the ferromagnetic part εᶜ_1
b_ferromagnetic = 27.4203609

def correlation_chachiyo(density, b):
    # The correlation energy of a uniform electron gas in Chachiyo's parameterization.
    # In terms of the Wigner-Seitz radius
    #  rₛ = (4π/3 ρ)⁻¹ᐟ³
    # the correlation energy per electron is expressed as
    #  εᶜ(rₛ) = a log(1+b/rₛ+b/rₛ²)
    b1 = pow(4.0/3.0*numpy.pi, 1.0/3.0) * b
    b2 = pow(4.0/3.0*numpy.pi, 2.0/3.0) * b
    # In terms of the density the correlation energy becomes
    #  εᶜ(ρ) = a log( 1 + b1 ρ¹ᐟ³ + b2 ρ²ᐟ³ )
    epsilon_c = a * numpy.log(1.0 + b1 * pow(density, 1.0/3.0) + b2 * pow(density, 2.0/3.0))
    # Multiply the correlation energy per particle by the particel density.
    return epsilon_c

# Number of points for plotting.
npts = 100

# Wigner-Seitz radius
rs = numpy.linspace(0.001, 10.0, npts)
# electron density
rho = 3.0/(4.0*numpy.pi) * pow(rs,-3)

plt.plot(rs, correlation_chachiyo(rho, b_paramagnetic),
    label=r"$\epsilon_{\text{corr}}^0(r_s)$ paramagnetic")
plt.plot(rs, correlation_chachiyo(rho, b_ferromagnetic),
    label=r"$\epsilon_{\text{corr}}^1(r_s)$ ferromagnetic",
    ls="-.")
# Ratio of ferromagnetic to paramagnetic correlation energy
#plt.plot(rs, correlation_chachiyo(rho, b_ferromagnetic) / correlation_chachiyo(rho, b_paramagnetic), ls="--")

plt.xlabel(r"Wigner-Seitz radius $r_s$ / $a_0$")
plt.ylabel(r"Correlation Energy $\epsilon_{\text{corr}}$ / $E_h$")
plt.legend()

plt.tight_layout()

#plt.savefig("correlation_energy_paramagnetic_fs_ferromagnetic.png", dpi=300)
#plt.savefig("correlation_energy_paramagnetic_fs_ferromagnetic.svg")

plt.show()
