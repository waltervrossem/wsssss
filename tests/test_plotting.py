#!/usr/bin/env python

import unittest
import os
from .common import have_mesa_data

from wsssss import load_data as ld
from wsssss.plotting import plotting as pl
plt = pl.plt

have_mesa_data()
test_data = os.path.join(os.path.dirname(__file__), 'data', 'mesa')

class TestFunctions(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.hist = ld.History(f'{test_data}/0000/LOGS/history.data')
        cls.profs = ld.load_profs(cls.hist)
        cls.gss = ld.load_gss(cls.hist)
        

    def test_make_vhrd(self):
        hist = self.hist
        profs = self.profs
        gss = self.gss
        f, ax = pl.make_vhrd(hist)
        

    def test_make_hrd(self):
        hist = self.hist
        profs = self.profs
        gss = self.gss
        f, ax = pl.make_hrd(hist)
        

    def test_make_propagation(self):
        hist = self.hist
        profs = self.profs
        gss = self.gss
        f, ax = pl.make_propagation(profs[-1], hist)
        

    def test_make_propagation2(self):
        hist = self.hist
        profs = self.profs
        gss = self.gss
        f, ax = pl.make_propagation2(profs[-1], hist)
        

    def test_make_echelle(self):
        hist = self.hist
        profs = self.profs
        gss = self.gss
        f, ax = pl.make_echelle(gss[-1], hist)
        

    def test_make_inertia(self):
        hist = self.hist
        profs = self.profs
        gss = self.gss
        f, ax = pl.make_inertia(gss[-1])
        

    def test_make_resolutions(self):
        hist = self.hist
        profs = self.profs
        gss = self.gss
        f, ax, (counts, edges_x, edges_y) = pl.make_resolutions(profs)
        

    def test_make_hrd_profiles(self):
        hist = self.hist
        profs = self.profs
        gss = self.gss
        f, ax = pl.make_hrd_profiles(hist)
        

    def test_make_hrd_models(self):
        hist = self.hist
        profs = self.profs
        gss = self.gss
        f, ax = pl.make_hrd_models(hist)
        

    def test_make_age_nu(self):
        hist = self.hist
        profs = self.profs
        gss = self.gss
        f, ax = pl.make_age_nu(hist, gss)
        

    def test_make_mesa_gyre_delta_nu(self):
        hist = self.hist
        profs = self.profs
        gss = self.gss
        f, ax = pl.make_mesa_gyre_delta_nu(hist, gss)
        

    def test_make_composition(self):
        hist = self.hist
        profs = self.profs
        gss = self.gss
        f, ax = pl.make_composition(profs[-1])
        

    def test_make_gradients(self):
        hist = self.hist
        profs = self.profs
        gss = self.gss
        f, ax = pl.make_gradients(profs[-1])
        

    def test_make_period_spacing(self):
        hist = self.hist
        profs = self.profs
        gss = self.gss
        f, ax = pl.make_period_spacing(gss[-1], hist)
        

    def test_make_structural(self):
        hist = self.hist
        profs = self.profs
        gss = self.gss
        f, ax = pl.make_structural(profs[-1])
        

    # def test_make_eos(self):
    #     hist = self.hist
    #     profs = self.profs
    #     gss = self.gss
    #     f, ax = pl.make_eos(profs[-1])
        

    def test_make_eigenfunc_compare(self):
        hist = self.hist
        profs = self.profs
        gss = self.gss
        gef = ld.load_modes_from_profile(profs[-1])
        f, ax = pl.make_eigenfunc_compare(gss[-1], gef, profs[-1], hist)
        

    def test_make_kipp(self):
        hist = self.hist
        profs = self.profs
        gss = self.gss
        f, ax = pl.make_kipp(hist)
        f, ax = pl.make_kipp(hist, profs, yaxis='mass')
