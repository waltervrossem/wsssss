#/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import io
import subprocess
import unittest

test_data = os.path.join(os.path.dirname(__file__), '..', 'data', 'mesa')

class TestCheckGrid(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        available_data = os.listdir(test_data)
        for req_data in ['0000', '0001', 'out_0000', 'out_0001']:
            if not req_data in available_data:
                raise FileNotFoundError('Run make_data/sh in tests/make_data before running this test.')

    def test_check_grid(self):
        os.chdir(test_data)
        output = subprocess.run(['check-grid', '--no-slurm',  '--out-file', '../out_{}'], stdout=subprocess.PIPE)

        expected = ("--------------------------------------------\n"
                    "  termination_code                   count\n"
                    "--------------------------------------------\n"
                    "max_model_number                           2\n"
                    "--------------------------------------------\n"
                    "\n")

        self.assertEqual(expected, output.stdout.decode())
