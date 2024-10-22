#!/usr/bin/env python3
# -*- coding: utf-8 -*-

version = 'r24.03.1'  # MESA version used to generate constants.py.

max_extra_inlists = 5  # number of inlists an inlist can depend on
pi = 3.1415926535897932384626433832795028841971693993751E0
pi2 = pi * pi
pi4 = 4*pi
eulercon = 0.577215664901532861E0
eulernum = 2.71828182845904523536028747135266249E0
ln2 = 6.9314718055994529E-01  # = log(2d0)
ln3 = 1.0986122886681096E+00  # = log(3d0)
lnpi = 1.14472988584940017414343  # = log(pi)
ln10 = 2.3025850929940455  # = log(10d0)
iln10 = 0.43429448190325187  # = 1d0/log(10d0)
a2rad = pi/180.0E0 # angle to radians
rad2a = 180.0E0/pi # radians to angle
one_third = 1E0/3E0
two_thirds = 2E0/3E0
four_thirds = 4E0/3E0
five_thirds = 5E0/3E0
one_sixth = 1E0/6E0
four_thirds_pi = four_thirds*pi
ln4pi3 = 1.4324119583011810E0  # = log(4*pi/3)
two_13 = 1.2599210498948730E0  # = pow(2d0,1d0/3d0)
four_13 = 1.5874010519681994E0  # = pow(4d0,1d0/3d0)
sqrt2 = 1.414213562373095E0  # = sqrt(2)
sqrt_2_div_3 = 0.816496580927726E0  # = sqrt(2/3)
avo = 6.02214076E23 # Avogadro constant (mole^-1)
amu = 1E0 / avo # atomic mass unit (g)
clight = 2.99792458E10 # speed of light in vacuum (cm s^-1)
qe = (clight/10E0) * 1.602176634E-19 # elementary charge (esu # == (g cm^3 s^-2)^(1/2))
kerg = 1.380649E-16
boltzm = kerg # Boltzmann constant (erg K^-1)
planck_h = 6.62607015E-27 # Planck constant (erg s)
hbar = planck_h / (2*pi)
cgas = boltzm*avo # ideal gas constant (erg K^-1)
ev2erg = 1.602176634E-12 # electron volt (erg)
mev_to_ergs = 1E6*ev2erg
mev_amu = mev_to_ergs/amu
mev2gr = 1E6*ev2erg/(clight*clight) # MeV to grams
qconv = mev_to_ergs*avo
kev = kerg / ev2erg # converts temp to ev (ev K^-1)
boltz_sigma = (pi*pi * boltzm*boltzm*boltzm*boltzm) / (60 * hbar*hbar*hbar * clight*clight) # Stefan-Boltzmann constant (erg cm^-2 K^-4 s^-1)
crad = boltz_sigma*4/clight # radiation density constant, AKA "a" (erg cm^-3 K^-4)# Prad # = crad * T^4 / 3
au = 1.49597870700E13 # (cm) - exact value defined by IAU 2009, 2012
pc = (3.600E3 * rad2a) * au # (cm) parsec, by definition
dayyer = 365.25E0 # days per (Julian) year
secday = 24*60*60  # seconds in a day
secyer = secday*dayyer # seconds per year
ly = clight*secyer # light year (cm)
mn = 1.67492749804E-24 # neutron mass (g)
mp = 1.67262192369E-24 # proton mass (g)
me = 9.1093837015E-28  # electron mass (g)
rbohr = 5.29177210903E-9 # Bohr radius (cm)
fine = 7.2973525693E-3   # fine-structure constant
hion = 13.605693122994E0 # Rydberg constant (eV)
sige = 6.6524587321E-25 # Thomson cross section (cm^2)
weinberg_theta  = 0.22290E0 # sin**2(theta_weinberg)
num_neu_fam = 3.0E0 # number of neutrino flavors # = 3.02 plus/minus 0.005 (1998)
standard_cgrav = 6.67430E-8 # gravitational constant (g^-1 cm^3 s^-2)
mu_sun = 1.3271244E26
mu_earth = 3.986004E20
mu_jupiter = 1.2668653E23
agesun = 4.57E9  # solar age (years) from Bahcall et al, ApJ 618 (2005) 1049-1056.
msun = mu_sun / standard_cgrav # solar mass (g)# gravitational mass, not baryonic
rsun = 6.957E10 # solar radius (cm), IAU 2015 Resolution B3
lsun = 3.828E33 # solar luminosity (erg s^-1), IAU 2015 Resolution B3
teffsun = 5772.0E0# solar effective temperature (K), IAU 2015 Resolution B3
loggsun = 4.4380676273031332 # log10(mu_sun/(Rsun*Rsun)), can't call log10 because we don't have math_lib at this point
mbolsun = 4.74E0  # Bolometric magnitude of the Sun, IAU 2015 Resolution B2
m_earth = mu_earth/standard_cgrav# earth mass (g)
r_earth = 6.3781E8 # earth equatorial radius (cm)
r_earth_polar = 6.3568E8 # earth polar radius (cm)
m_jupiter = mu_jupiter/standard_cgrav # jupiter mass (g)
r_jupiter = 7.1492E9 # jupiter equatorial radius (cm)
r_jupiter_polar = 6.6854E9 # jupiter polar radius (cm)
semimajor_axis_jupiter = 7.7857E13 # jupiter semimajor axis (cm)
arg_not_provided = -9E99
missing_value = arg_not_provided
crystallized = -1
no_mixing = 0
convective_mixing = 1
overshoot_mixing = 2
semiconvective_mixing = 3
thermohaline_mixing = 4
rotation_mixing = 5
rayleigh_taylor_mixing = 6
minimum_mixing = 7
anonymous_mixing = 8  # AKA "WTF_mixing"
leftover_convective_mixing = 9  # for regions with non-zero conv_vel that are not unstable to convection
phase_separation_mixing = 10
number_of_mixing_types =  phase_separation_mixing+1
