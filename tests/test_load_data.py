#/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import profile
import unittest
import numpy as np

from wsssss import load_data as ld
from .common import have_mesa_data

have_mesa_data()
test_data = os.path.join(os.path.dirname(__file__), 'data', 'mesa')
curdir = os.path.dirname(__file__)
class TestLoadData(unittest.TestCase):

    def test_History(self):
        hist = ld.History(os.path.join(test_data, '0000', 'LOGS', 'history.data'))
        hist.dump()

        hist_dill = ld.History(os.path.join(test_data, '0000', 'LOGS', 'history.data.dill'))
        np.testing.assert_array_equal(hist.data, hist_dill.data)
        self.assertDictEqual(hist.header, hist_dill.header)
        del hist_dill

        hist_reload = ld.History(os.path.join(test_data, '0000', 'LOGS', 'history.data'), save_dill=True, reload=True)
        np.testing.assert_array_equal(hist.data, hist_reload.data)
        self.assertDictEqual(hist.header, hist_reload.header)
        del hist_reload

        np.testing.assert_array_equal(np.arange(1, 1001), hist.get('model_number'))
        np.testing.assert_array_equal(np.arange(1, 1001), hist.data.model_number)
        np.testing.assert_array_equal(np.ones(1000), hist.data.star_mass)
        np.testing.assert_array_equal(np.arange(1, 11), hist[:10].data.model_number)

        np.testing.assert_array_equal(hist.index[:, 0]-1, hist.get_profile_index(hist.index[:, 2]))
        np.testing.assert_array_equal((2, 100, 99), hist.get_profile_num(150))
        np.testing.assert_array_equal((3, 200, 199), hist.get_profile_num(150, earlier=False))

        hist_cols = ld.History(os.path.join(test_data, '0000', 'LOGS', 'history.data'), keep_columns=['model_number', 'center_he4'])
        self.assertListEqual(['model_number', 'center_he4'], hist_cols.columns)
        self.assertListEqual(hist_cols.columns, list(hist_cols.data.dtype.names))
        np.testing.assert_array_equal(hist.data[hist_cols.columns], hist_cols.data[hist_cols.columns])

    def test_Profile(self):
        prof = ld.Profile(os.path.join(test_data, '0000', 'LOGS', 'profile1.data'))
        hist = ld.History(os.path.join(test_data, '0000', 'LOGS', 'history.data'))

        np.testing.assert_array_equal(np.zeros(1), hist.get_profile_index(prof))
        np.testing.assert_array_equal(np.zeros(1), hist.get_profile_index([prof]))
        np.testing.assert_array_equal(np.zeros(1), hist.get_profile_index(prof.profile_num))
        np.testing.assert_array_equal(np.zeros(1), hist.get_profile_index([prof.profile_num]))

        prof = ld.Profile(os.path.join(test_data, '0000', 'LOGS', 'profile1.data'), load_GyreProfile=True)

        self.assertEqual(prof.get_hist_index(hist), 0)

    def test_GyreSummary(self):
        gsum = ld.GyreSummary(os.path.join(test_data, '0000', 'gyre_out', 'profile10.data.GYRE.sgyre_l'))
        self.assertEqual(7, len(gsum.data[gsum.data['l'] == 0]))
        self.assertEqual(236, len(gsum.data[gsum.data['l'] == 1]))

    def test_GyreProfile(self):
        prof = ld.Profile(os.path.join(test_data, '0000', 'LOGS', 'profile1.data'))
        gprof = ld.GyreProfile(os.path.join(test_data, '0000', 'LOGS', 'profile1.data.GYRE'))

        np.testing.assert_allclose(prof.data.mass/prof.data.mass[0], np.interp(prof.data.radius/prof.data.radius[0], gprof.data.radius/gprof.header['star_radius'], gprof.data.mass/gprof.header['star_mass']), rtol=1e-11)

    def test_GyreMode(self):
        gsum = ld.GyreSummary(os.path.join(test_data, '0000', 'gyre_out', 'profile10.data.GYRE.sgyre_l'))
        gmode = ld.GyreSummary(os.path.join(test_data, '0000', 'gyre_out', 'profile10.data.GYRE_l0_00005_np+9_ng+0.mgyre'))
        np = 9
        ng = 0
        mask = (gsum.data.n_p == np) & (gsum.data.n_g == ng)
        self.assertEqual(1, sum(mask))
        self.assertEqual(gmode.header['Re(freq)'], gsum.data['Re(freq)'][mask])

    def test_load_profs(self):
        hist = ld.History(os.path.join(test_data, '0000', 'LOGS', 'history.data'))
        profs = ld.load_profs(hist)
        self.assertEqual(11, len(profs))
        self.assertListEqual(list(np.arange(1, 12)), [prof.profile_num for prof in profs])

    def test_load_gss(self):
        hist = hist = ld.History(os.path.join(test_data, '0000', 'LOGS', 'history.data'))
        gss = ld.load_gss(hist)
        self.assertEqual(11, len(gss))

    def load_modes_from_profile(self):
        gsum = ld.GyreSummary(os.path.join(test_data, '0000', 'gyre_out', 'profile10.data.GYRE.sgyre_l'))
        prof = ld.Profile(os.path.join(test_data, '0000', 'LOGS', 'profile10.data'))
        modes = ld.load_modes_from_profile(prof)
        self.assertEqual(7, len(gsum.data[gsum.data.l==0]))

    def load_gs_from_profile(self):
        prof = ld.Profile(os.path.join(test_data, '0000', 'LOGS', 'profile10.data'))
        gs = ld.load_gs_from_profile(prof)
        self.assertEqual(243, len(gs.data))

