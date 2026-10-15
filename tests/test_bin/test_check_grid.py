#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import subprocess
import unittest

from ..common import have_mesa_data

have_mesa_data()
TEST_DATA = os.path.join(os.path.dirname(__file__), "..", "data", "mesa")


class TestCheckGrid(unittest.TestCase):
    def test_check_grid(self):
        os.chdir(TEST_DATA)
        output = subprocess.run(
            [
                "check-grid",
                ".",
                "--out-file",
                "../out_{}",
                "--make-restart-file",
                "last",
                "--list-all",
                "--min-photo-delta",
                "200",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        self.assertEqual(
            output.returncode,
            0,
            msg=f"check_grid.py failed:\n{output.stderr}",
        )

        expected = (
            "--------------------------------------------\n"
            "  termination_code                   count\n"
            "--------------------------------------------\n"
            "max_model_number                           2\n"
            " └ pre-CHeX                             2\n"
            "    └ post-CHX                       2\n"
            "--------------------------------------------\n"
            "0000               pre-CHeX    1000 max_model_number\n"
            "0001               pre-CHeX    1000 max_model_number\n"
            "--------------------------------------------\n"
            "\n"
        )

        self.assertEqual(expected, output.stdout.replace("\r\n", "\n"))
        with open("grid_restart", "r") as handle:
            restart_text = handle.read()
        self.assertEqual("0000 x800\n0001 x800\n", restart_text)
