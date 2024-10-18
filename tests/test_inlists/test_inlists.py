#/usr/bin/env python3
# -*- coding: utf-8 -*-
import unittest
import os
import sys
import io

from f90nml import Namelist
from wssss.inlists import inlists as inl

test_data = '../data/inlists'
test_data = os.path.join(os.path.dirname(__file__), '..', 'data', 'inlists')

class TestInlists(unittest.TestCase):
    def setUp(self):
        inlist = inl.evaluate_inlist(os.path.join(test_data, 'inlist'))
        in1 = inl.evaluate_inlist(os.path.join(test_data, 'inlist_1'))
        in2 = inl.evaluate_inlist(os.path.join(test_data, 'inlist_2'))
        in3 = inl.evaluate_inlist(os.path.join(test_data, 'inlist_3'))

        self.inlist = inlist
        self.sub_inlists = [in1, in2, in3]

        correct = Namelist()
        correct['star_job'] = Namelist()
        correct['eos'] = Namelist()
        correct['kap'] = Namelist()
        correct['controls'] = Namelist([('initial_mass', 2.0), ('initial_z', 0.01), ('initial_y', 0.25)])
        correct['pgstar'] = Namelist()
        self.correct = correct


    def test_evaluate_inlist(self):
        self.assertDictEqual(self.correct, self.inlist)


    def test_evaluate_inlist_str(self):
        with open(os.path.join(test_data, 'inlist'), 'r') as handle:
            inlist_str = handle.read()
        inlist = inl.evaluate_inlist_str(inlist_str, test_data)
        self.assertDictEqual(self.correct, inlist)


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
        expected = ("left : /home/walter/Github/wssss/tests/test_inlists/../data/inlists/inlist_3\n"
                    "right: /home/walter/Github/wssss/tests/test_inlists/../data/inlists/inlist_1\n"
                    "\n"
                    "####### star_job differences #######\n"
                    "chem_isotopes_filename: ('isotopes.data', None)\n"
                    "\n"
                    "####### controls differences #######\n"
                    "initial_mass: (3.0, 1.0)\n"
                    "initial_z: (None, 0.01)\n"
                    )
        self.assertEqual(expected, capturedOutput.getvalue())

    def test_round_trip(self):
        self.inlist.write(os.path.join(test_data, 'write_inlist'), force=True)
        read_inlist = inl.evaluate_inlist(os.path.join(test_data, 'write_inlist'))
        self.assertDictEqual(self.inlist, read_inlist)
        os.remove(os.path.join(test_data, 'write_inlist'))
