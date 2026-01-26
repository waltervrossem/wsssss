#/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import unittest

import numpy as np
from wsssss import load_data as ld
from wsssss._bin.gyre_driver import gyre_driver

test_data = os.path.join(os.path.dirname(__file__), '..', 'data', 'gyre')

must_have_environ = ['GYRE_DIR']
for env in must_have_environ:
    if env not in os.environ:
        raise EnvironmentError(f'{env} not set.')

@unittest.skipIf(os.name == 'nt', 'Skipping on Windows')
class TestGyreDriver(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        gs = ld.GyreSummary(os.path.join(test_data, 'profile10.data.GYRE.sgyre_l'))
        cls.ref_gs = gs

    def test_gyre_driver(self):
        os.chdir(test_data)
        sys.argv = ['gyre-driver', '0', 'MESA', 'LOGS/profile10.data.GYRE', '--gyre', 'G7']
        ierr = gyre_driver.run()
        self.assertEqual(0, ierr)
        gs_path = os.path.join(test_data, 'gyre_out', 'profile10.data.GYRE.sgyre_l')
        gs = ld.GyreSummary(gs_path)
        np.testing.assert_array_equal(self.ref_gs.data[self.ref_gs.data.l == 0], gs.data)
        os.remove(gs_path)

    def test_gyre_min_numax(self):
        os.chdir(test_data)
        sys.argv = ['gyre-driver', '01', 'MESA', 'LOGS/profile10.data.GYRE', '--min-numax', '45', '--gyre', 'G7']
        ierr = gyre_driver.run()
        self.assertEqual(0, ierr)
        gs_path = os.path.join(test_data, 'gyre_out', 'profile10.data.GYRE.sgyre_l')
        gs = ld.GyreSummary(gs_path)
        # Should only have l=0 modes as the Model's numax is 44.6
        np.testing.assert_array_equal(self.ref_gs.data[self.ref_gs.data.l == 0], gs.data)
        os.remove(gs_path)

    def test_lenient(self):
        sys.argv = ['gyre-driver', '0', 'MESA', 'LOGS/profile10.data.GYRE', '--gyre', 'G4', '--lenient']
        ierr = gyre_driver.run()
        self.assertEqual(0, ierr)
        gs_path = os.path.join(test_data, 'gyre_out', 'profile10.data.GYRE.sgyre_l')
        gs = ld.GyreSummary(gs_path)
        np.testing.assert_array_equal(self.ref_gs.data[self.ref_gs.data.l == 0], gs.data)
        os.remove(gs_path)

    def test_gyre_versions(self):
        possible_versions = os.listdir(f'{os.environ["GYRE_DIR"]}/..')
        tested_versions = []
        for gyredir in possible_versions:
            path = os.path.abspath(f'{os.environ["GYRE_DIR"]}/../{gyredir}')
            version_file = f'{path}/src/common/gyre_version.fpp'
            if not os.path.exists(version_file):  # Different file name for 7.2 and after
                version_file = f'{path}/src/common/version_m.fypp'

            if not os.path.exists(version_file):
                continue
            with open(version_file, 'r') as handle:
                lines = handle.readlines()

            version_str = ''
            for line in lines:
                if 'VERSION = ' in line.upper():
                    line = line.upper()
                    version_str = line.split('VERSION =')[1].strip().replace("'", '').replace('(', '').replace(')', '')
                    break
            major_version = version_str.split('.')[0]
            os.environ["GYRE_DIR"] = path
            sys.argv = ['gyre-driver', '0', 'MESA', 'LOGS/profile10.data.GYRE', '--gyre', f'G{major_version}']

            tested_versions.append(version_str)
        np.testing.assert_array_equal(np.array([6, 7, 8]), np.unique(np.array([int(v.split('.')[0]) for v in tested_versions])))
