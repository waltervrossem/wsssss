#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import subprocess
import unittest

from ..common import have_mesa_data

have_mesa_data()
test_data = os.path.join(os.path.dirname(__file__), "..", "data", "mesa")


class TestCheckGrid(unittest.TestCase):
    def test_check_grid(self):
        os.chdir(test_data)
        output = subprocess.run(
            ["check-grid", "--no-slurm", "--out-file", "../out_{}", "--make-restart-file", "last"],
            stdout=subprocess.PIPE,
        )

        expected = (
            "--------------------------------------------\n"
            "  termination_code                   count\n"
            "--------------------------------------------\n"
            "max_model_number                           2\n"
            "--------------------------------------------\n"
            "\n"
        )

        self.assertEqual(expected, output.stdout.decode().replace("\r\n", "\n"))
        with open("grid_restart", "r") as handle:
            s = handle.read()
        self.assertEqual(s, "0000 x700\n0001 x700\n")
