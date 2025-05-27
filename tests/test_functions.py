#/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import profile
import unittest
import numpy as np

from wsssss import load_data as ld
from wsssss import functions as uf

from .common import have_mesa_data

have_mesa_data()

must_have_environ = ['MESA_DIR']
for env in must_have_environ:
    if env not in os.environ:
        raise EnvironmentError(f'{env} not set.')

test_data = os.path.join(os.path.dirname(__file__), 'data', 'mesa')

class TestFunctions(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.hist = ld.History(f'{test_data}/0000/LOGS/history.data')

    def test_get_mesa_version(self):
        mesa_version = uf.get_mesa_version(os.environ['MESA_DIR'])

    def test_compare_mesa_version(self):
        self.assertTrue(uf.compare_version('11701', 'r24.03.1', '<'))
        self.assertTrue(uf.compare_version('11701', 'r24.03.1', '<='))
        self.assertFalse(uf.compare_version('11701', 'r24.03.1', '>'))
        self.assertFalse(uf.compare_version('11701', 'r24.03.1', '>='))
        self.assertFalse(uf.compare_version('11701', 'r24.03.1', '=='))
        self.assertTrue(uf.compare_version('11701', 'r24.03.1', '!='))
        self.assertTrue(uf.compare_version('8888', '11701', '<'))

    def test_get_constants(self):
        const = uf.get_constants(self.hist)

        hist_post15140 = uf.compare_version(self.hist.header['version_number'], '15140', '>=')
        const_post15140 = uf.compare_version(const.version, '15140', '>=')

        self.assertTrue(hist_post15140 == const_post15140)  # Both True or both False is correct

    def test_convert_mixing_type(self):
        types = np.arange(10)
        types_pre = uf.convert_mixing_type(types, '11701', -1)
        types_post = uf.convert_mixing_type(types, '15140', -1)

        np.testing.assert_array_equal(np.array([100, 101,  -1, 103, 104, 105, 106,  -1, 107, 109]), types_pre)
        np.testing.assert_array_equal(np.array([100, 101, 103, 104, 105, 106,  -1, 107, 109,  -1]), types_post)

    def test_masks(self):
        for mask_func in uf.mask_functions:
            mask = mask_func(self.hist)
        mask = uf.get_ms_mask(self.hist)
        self.assertEqual(len(self.hist), len(mask))
        self.assertEqual(len(self.hist[mask]), sum(mask))

    def test_calc_deltanu(self):
        gss = ld.load_gss(self.hist)
        gs = gss[0]
        uf.calc_deltanu(gs, self.hist)

    def test_calc_deltaPg(self):
        gss = ld.load_gss(self.hist)
        gs = gss[0]
        uf.calc_deltaPg(gs, self.hist, 1)

    def test_correct_seismo(self):
        uf.correct_seismo(self.hist, ld.load_gss(self.hist, return_pnums=True), uf.get_rgb_mask, 'star_age')

