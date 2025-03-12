# /usr/bin/env python3
# -*- coding: utf-8 -*-

import io
import os
import sys
import time
import unittest
import shutil

from wsssss.inlists import create_grid as cg
from wsssss.inlists import inlists as inl


must_have_environ = ['MESA_DIR']
for env in must_have_environ:
    if env not in os.environ:
        raise EnvironmentError(f'{env} not set.')

class TestCreateGrid(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.grid_dir = os.path.join(os.path.dirname(__file__), '../data/grid')

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.grid_dir)

    def setUp(self):
        grid = cg.MesaGrid()

        grid.inlist['controls']['read_extra_controls_inlist(1)'] = True
        grid.inlist['controls']['extra_controls_inlist_name(1)'] = 'copy_this_namelist'

        grid.star_job['show_net_species_info'] = [True]

        grid.controls['initial_mass'] = [1, 2]
        grid.controls['max_model_number'] = 1000

        grid.controls[f'{cg.non_mesa_key_start}group_unpack'].append([{'overshoot_scheme(1)': 'exponential',
                                                                       'overshoot_zone_type(1)': 'nonburn',
                                                                       'overshoot_zone_loc(1)': 'shell',
                                                                       'overshoot_bdy_loc(1)': 'bottom',
                                                                       'overshoot_f(1)': 0.021,
                                                                       'overshoot_f0(1)': 0.001},
                                                                      {'overshoot_scheme(1)': 'exponential',
                                                                       'overshoot_zone_type(1)': 'nonburn',
                                                                       'overshoot_zone_loc(1)': 'shell',
                                                                       'overshoot_bdy_loc(1)': 'bottom',
                                                                       'overshoot_f(1)': 0.03,
                                                                       'overshoot_f0(1)': 0.01}])

        self.grid = grid

    def test_unpack(self):
        self.grid.unpack_inlists()
        self.assertEqual(4, len(self.grid.unpacked))
        self.assertEqual(1, self.grid.unpacked[0]['controls']['initial_mass'])
        self.assertEqual(2, self.grid.unpacked[1]['controls']['initial_mass'])
        self.assertEqual(0.021, self.grid.unpacked[0]['controls']['overshoot_f(1)'])
        self.assertEqual(0.03, self.grid.unpacked[3]['controls']['overshoot_f(1)'])

        self.assertEqual(True, self.grid.unpacked[0]['star_job']['show_net_species_info'])

    def test_validate_inlist(self):
        self.assertEqual(None, self.grid.validate_inlists())
        self.assertEqual(None, self.grid.validate_inlists(self.grid.mesa_dir))

        # Raise KeyError for option which does not exist in MESA
        self.grid.star_job['this_option_does_not_exits'] = True
        self.assertRaises(KeyError, self.grid.validate_inlists)
        self.grid.star_job.pop('this_option_does_not_exits')

        # Raise ValueError if extra_controls_inlist_name is the same as master inlist name.
        self.grid.inlist['controls']['read_extra_controls_inlist(1)'] = True
        self.grid.inlist['controls']['extra_controls_inlist_name(1)'] = 'inlist_project'
        self.grid.add_file('/home/walter/Github/MESA_templates/24.03.1/template_24031/inlist_project')
        self.assertRaises(ValueError, self.grid.validate_inlists)

    def test_MesaGrid(self):

        grid_dir = self.grid_dir
        time_now = str(int(time.time()))
        if os.path.exists(os.path.join(grid_dir, 'test_time')):
            with open(os.path.join(grid_dir, 'test_time'), 'r') as handle:
                prev_time = handle.read()
        else:
            prev_time = -1

        def finalize_function(gridobj, i):
            paths = os.listdir('.')
            paths.sort()
            print(paths)

        self.grid.set_griddir_finalize_function(finalize_function)

        capturedOutput = io.StringIO()
        sys.stdout = capturedOutput

        self.grid.create_grid(grid_dir)
        sys.stdout = sys.__stdout__
        expected = ("['copy_this_namelist', 'inlist', 'inlist_project']\n"
                    "['copy_this_namelist', 'inlist', 'inlist_project']\n"
                    "['copy_this_namelist', 'inlist', 'inlist_project']\n"
                    "['copy_this_namelist', 'inlist', 'inlist_project']\n")
        self.assertEqual(expected, capturedOutput.getvalue())

        self.assertFalse(os.path.exists(os.path.join(grid_dir, 'test_time')),
                         msg="This file should have been deleted when create_grid was called.")

        with open(os.path.join(grid_dir, 'test_time'), 'w') as handle:
            handle.write(time_now)

        inlist_from_file = inl.evaluate_inlist(os.path.join(grid_dir, '0000', 'inlist'))
        inlist_from_grid = self.grid.unpacked[0]

        for namelist in self.grid.namelists:
            mesa_only = {k: v for k, v in inlist_from_grid[namelist].items() if not k.startswith(cg.non_mesa_key_start)}
            diff = inl.inlist_diff(mesa_only, inlist_from_file[namelist])
            if namelist == 'controls':  # in copy_this_namelist
                self.assertDictEqual({'max_age': (None, 1000000000000.0)}, diff['changed'], msg=str(diff['changed']))
            else:
                self.assertDictEqual({}, diff['changed'], msg=str(diff['changed']))
