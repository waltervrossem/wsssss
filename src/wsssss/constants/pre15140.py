#!/usr/bin/env python3
# -*- coding: utf-8 -*-

version = '11701'  # MESA version used to generate constants.py.

pi = 3.1415926535897932384626433832795028841971693993751E0
pi4 = 4*pi
eulercon = 0.577215664901532861E0
ln2 = 6.9314718055994529E-01  # = log_cr(2d0)
ln3 = 1.0986122886681096E+00  # = log_cr(3d0)
ln10 = 2.3025850929940455E+00  # = log_cr(10d0)
a2rad = pi/180.0E0 # angle to radians
rad2a = 180.0E0/pi # radians to angle
one_third = 1E0/3E0
two_thirds = 2E0/3E0
ln4pi3 = 1.4324119583011810E0  # = log_cr(4*pi/3)
two_13 = 1.2599210498948730E0  # = pow_cr(2d0,1d0/3d0)
four_13 = 1.5874010519681994E0  # = pow_cr(4d0,1d0/3d0)
sqrt2 = 1.414213562373095E0  # = sqrt(2)
standard_cgrav  = 6.67428E-8  # gravitational constant (g^-1 cm^3 s^-2)
planck_h  = 6.62606896E-27  # Planck's constant (erg s)
hbar  = planck_h / (2*pi)
qe  = 4.80320440E-10  # electron charge (esu # == (g cm^3 s^-2)^(1/2))
avo  = 6.02214179E23  # Avogadro's constant (mole^-1)
clight  = 2.99792458E10  # speed of light in vacuum (cm s^1)
kerg  = 1.3806504E-16  # Boltzmann's constant (erg K^-1)
boltzm  = kerg
cgas  = boltzm*avo # R_gas# ideal gas constant# erg/K/mole
kev  = 8.617385E-5  # converts temp (kelvin) to ev (ev K^-1)
amu  = 1.660538782E-24  # atomic mass unit (g)
mn  = 1.6749286E-24 # neutron mass (g)
mp  = 1.6726231E-24 # proton mass (g)
me  = 9.1093826E-28 # (was 9.1093897d-28) electron mass (g)
rbohr  = hbar*hbar/(me * qe * qe) # Bohr radius (cm)
fine  = qe*qe/(hbar*clight) # fine structure constant
hion  = 13.605698140E0 # hydrogen ionization energy (eV)
ev2erg  = 1.602176487E-12 # electron volt (erg)
mev_to_ergs  = 1E6*ev2erg
mev_amu  = mev_to_ergs/amu
qconv  = ev2erg*1.0E6*avo# convert Q rates to erg/gm/sec
boltz_sigma  = 5.670400E-5 # boltzmann's sigma # = crad*clight/4 (erg cm^-2 K^-4 s^-1)
crad  = boltz_sigma*4/clight, 7.5657673816464059E-15 # radiation density constant, a (erg cm^-3 K^-4)# Prad  # = crad * T^4 / 3
sige = 6.6524587158E-025 # Thomson scattering electron cross section
ssol  = boltz_sigma
asol  = crad
weinlam  = planck_h*clight/(kerg * 4.965114232E0)
weinfre  = 2.821439372E0*kerg/planck_h
rhonuc  = 2.342E14 # density of nucleus (g cm^3)
msol  = 1.9892E33  # solar mass (g)
rsol  = 6.9598E10 # solar radius (cm)
lsol  = 3.8418E33  # solar luminosity (erg s^-1)
agesol  = 4.57E9  # solar age (years)
msun  = msol
rsun  = rsol
lsun  = lsol
msun33  = msol*1E-33
rsun11  = rsol*1E-11
lsun33  = lsol*1E-33
teffsol  = 5777E0 # temperature (k)
loggsol  = 4.4378893534131256E0 # log surface gravity # log(g/(cm s^-2))
teffsun  = teffsol
loggsun  = loggsol
mbolsun  = 4.74 # Bolometric magnitude of the Sun
mbolsol  = mbolsun
ly  = 9.460528E17 # light year (cm)
pc  = 3.261633E0 * ly # parsec (cm)
m_earth  = 5.9764E27 # earth mass (g)
r_earth  = 6.37E8 # earth radius (cm)
au  = 1.495978921E13 # astronomical unit (cm)
m_jupiter  = 1.8986E30 # jupiter mass (g)
r_jupiter  = 6.9911E9 # jupiter mean radius (cm)
semimajor_axis_jupiter  = 7.7857E13 # jupiter semimajor axis (cm)
arg_not_provided = -9E99
missing_value = arg_not_provided
no_mixing = 0
convective_mixing = 1
softened_convective_mixing = 2   # for modified D_mix near convective boundary
overshoot_mixing = 3
semiconvective_mixing = 4
thermohaline_mixing = 5
rotation_mixing = 6
rayleigh_taylor_mixing = 7
minimum_mixing = 8
anonymous_mixing = 9