#/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import unittest

import numpy as np
from wsssss import load_data as ld
from wsssss._bin.gyre_driver import gyre_driver

test_data = os.path.join(os.path.dirname(__file__), '..', 'data', 'gyre')

class TestGyreDriver(unittest.TestCase):
    def setUp(self):
        gs = ld.GyreSummary(os.path.join(test_data, 'profile10.data.GYRE.sgyre_l'))
        self.gs = gs

    def test_gyre_driver(self):
        os.chdir(test_data)
        sys.argv = ['gyre-driver', '0 MESA LOGS/profile10.data.GYRE --gyre G7']
        ierr = gyre_driver.run()
        self.assertEqual(0, ierr)
        gs_path = os.path.join(test_data, 'gyre_out', 'profile10.data.GYRE.sgyre_l')
        gs = ld.GyreSummary(gs_path)
        np.testing.assert_array_equal(self.gs.data[self.gs.data.l == 0], gs.data)
        os.remove(gs_path)

    def test_lenient(self):
        sys.argv = ['gyre-driver', '0 MESA LOGS/profile10.data.GYRE --gyre G6 --lenient']
        ierr = gyre_driver.run()
        self.assertEqual(0, ierr)
        gs_path = os.path.join(test_data, 'gyre_out', 'profile10.data.GYRE.sgyre_l')
        gs = ld.GyreSummary(gs_path)
        np.testing.assert_array_equal(self.gs.data[self.gs.data.l == 0], gs.data)
        os.remove(gs_path)
