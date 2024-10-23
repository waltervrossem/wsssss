#/usr/bin/env python3
# -*- coding: utf-8 -*-
import unittest
import os
import sys
import io
import numpy as np

from wsssss.inlists import inlists as inl

test_data = os.path.join(os.path.dirname(__file__), '..', 'data', 'inlists')
mesa_dir = os.environ['MESA_DIR']

class TestInlists(unittest.TestCase):
    def setUp(self):
        inlist = inl.evaluate_inlist(os.path.join(test_data, 'inlist'))
        in1 = inl.evaluate_inlist(os.path.join(test_data, 'inlist_1'))
        in2 = inl.evaluate_inlist(os.path.join(test_data, 'inlist_2'))
        in3 = inl.evaluate_inlist(os.path.join(test_data, 'inlist_3'))

        self.inlist = inlist
        self.sub_inlists = [in1, in2, in3]

        correct = {}
        correct['star_job'] = {}
        correct['eos'] = {}
        correct['kap'] = {}
        correct['controls'] = dict([('initial_mass', 3.0), ('initial_z', 0.01), ('initial_y', 0.25)])
        correct['pgstar'] = {}
        self.correct = correct

        self.mesa_version = inl.get_mesa_version(mesa_dir)


    def test_evaluate_inlist(self):
        for key in self.correct.keys():
            self.assertDictEqual(dict(self.correct[key]), dict(self.inlist[key]))

    def test_evaluate_inlist_str(self):
        with open(os.path.join(test_data, 'inlist'), 'r') as handle:
            inlist_str = handle.read()
        inlist = inl.evaluate_inlist_str(inlist_str, test_data)
        for key in self.correct.keys():
            self.assertDictEqual(dict(self.correct[key]), dict(inlist[key]))


    def test_inlist_diff(self):
        self.assertRaises(ValueError, inl.inlist_diff, self.inlist, self.inlist)
        self.assertDictEqual({'same':dict(self.inlist['controls']), 'changed':{}},
                             inl.inlist_diff(self.inlist['controls'], self.inlist['controls']),)

        diff_in1_in3 = {'same': {},
                        'changed': {'initial_mass': (1.0, 3.0), 'initial_z': (0.01, None)}}
        self.assertDictEqual(diff_in1_in3, inl.inlist_diff(self.sub_inlists[0]['controls'], self.sub_inlists[2]['controls']))


    def test_compare_inlist(self):
        capturedOutput = io.StringIO()  # Create StringIO.
        sys.stdout = capturedOutput  # Redirect stdout.
        inl.compare_inlist(os.path.join(test_data, 'inlist_3'), os.path.join(test_data, 'inlist_1'))
        sys.stdout = sys.__stdout__ # Reset redirect.
        expected = ("left : /home/walter/Github/wsssss/tests/test_inlists/../data/inlists/inlist_3\n"
                    "right: /home/walter/Github/wsssss/tests/test_inlists/../data/inlists/inlist_1\n"
                    "\n"
                    "####### star_job differences #######\n"
                    "chem_isotopes_filename: ('isotopes.data', None)\n"
                    "\n"
                    "####### controls differences #######\n"
                    "initial_mass: (3.0, 1.0)\n"
                    "initial_z: (None, 0.01)\n"
                    )
        self.assertEqual(expected, capturedOutput.getvalue())


    def test_variable_to_string(self):
        self.assertEqual('1d10', inl.variable_to_string(1e10))
        self.assertEqual('1d0', inl.variable_to_string(1.0))
        self.assertEqual('1', inl.variable_to_string(1))
        self.assertEqual('0', inl.variable_to_string(np.arange(1)[0]))
        self.assertEqual('.true.', inl.variable_to_string(True))
        self.assertEqual('.false.', inl.variable_to_string(False))
        self.assertEqual("'abc'", inl.variable_to_string('abc'))

    def test_round_trip(self):
        path = os.path.join(test_data, 'write_inlist')
        inl.write_inlist(self.inlist, path)
        read_inlist = inl.evaluate_inlist(path)

        for key in self.correct.keys():
            self.assertDictEqual(self.inlist[key], read_inlist[key])
        os.remove(path)

    def test_check_inlist(self):
        checked = inl.check_inlist(os.path.join(test_data, 'inlist'), mesa_dir)
        num_incorrect = 0
        for nml_type in checked.keys():
            num_incorrect += len(checked[nml_type])
        if self.mesa_version >= '15140':
            self.assertEqual(0, num_incorrect)
        else:
            self.assertEqual(2, num_incorrect)
