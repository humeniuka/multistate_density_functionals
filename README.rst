
Multistate Density Functional Theory for Excited States
-------------------------------------------------------
This python package implements a multistate kinetic energy density matrix functional.
For a single electronic state it reduces to the von Weizsäcker kinetic energy functional.

Requirements
------------

Required python packages:

 * numpy, matplotlib
 * pyscf

Installation
------------
The package is installed with

.. code-block:: bash

   $ pip install -e .

in the top directory. To verify the proper functioning of the code
a set of tests should be run with

.. code-block:: bash

   $ cd tests
   $ python -m unittest

Getting Started
---------------


----------
References
----------
.. [1] Yangyi Lu, Jiali Gao, "Multistate Density Functional Theory for Excited States",
    J. Phys. Chem. Lett. 2022, 13, 7762-7769,
    https://doi.org/10.1021/acs.jpclett.2c02088
