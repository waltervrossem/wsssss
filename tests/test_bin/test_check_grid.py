#!/usr/bin/env python
# -*- coding: utf-8 -*-
import contextlib
import io
import os
import subprocess
import sys
import tempfile
import unittest
from collections import Counter

import wsssss._bin.check_grid as cg
from ..common import have_mesa_data

have_mesa_data()
TEST_DATA = os.path.join(os.path.dirname(__file__), "..", "data", "mesa")


def make_args(argv):
    """
    Build an argparse Namespace using check_grid.get_parser().
    """
    parser = cg.get_parser()
    return parser.parse_args(argv)


def write_mesa_history(path, truncate=0):
    """
    Write a minimal history file that read_hist_first_last_row() can parse.
    """
    with open(path, "w") as handle:
        handle.write("""
                                       1                                        2                                        3                                        4                                        5                                        6                                        7                                        8                                        9                                       10                                       11 
                          version_number                                 compiler                                    build                         MESA_SDK_version                             math_backend                                     date                                burn_min1                                burn_min2                                     msun                                     rsun                                     lsun 
                              "r24.08.1"                               "gfortran"                                 "13.3.0"                    "x86_64-linux-24.7.1"                                 "CRMATH"                               "20250519"                  5.0000000000000000E+001                  1.0000000000000000E+003                  1.9884098706980504E+033                  6.9570000000000000E+010                  3.8280000000000003E+033 
                                       1                                        2                                        3                                        4
                            model_number                                center_h1                               center_he4                           mass_conv_core
""")
        lines = [
            "                                       1                   1.0000000000000000E+00                   0.0000000000000000E+00                   0.0000000000000000E+00\n",
            "                                     500                   0.5000000000000000E+00                   0.5000000000000000E+00                   0.0000000000000000E+00\n",
            "                                    1000                   0.0000000000000000E+00                   1.0000000000000000E+00                   0.0000000000000000E+00\n",
            "                                    1500                   0.0000000000000000E+00                   0.5000000000000000E+00                   1.0000000000000000E+00\n",
            "                                    2600                   0.0000000000000000E+00                   0.0000000000000000E+00                   1.0000000000000000E+00\n",
        ]
        if truncate > 0:
            lines = lines[:-truncate]
        handle.writelines(lines)


def make_grid_dir(root, subdir="0000", termcode="max_model_number", truncate=0):
    """
    Create a minimal grid directory containing:
      - subdir/LOGS/history.data
      - subdir/photos/x750
      - out_<subdir>
    """
    run_dir = os.path.join(root, subdir)
    logs_dir = os.path.join(run_dir, "LOGS")
    photos_dir = os.path.join(run_dir, "photos")

    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(photos_dir, exist_ok=True)

    write_mesa_history(os.path.join(logs_dir, "history.data"), truncate=truncate)

    for name in ["x000", "x500", "1000", "2000"]:
        with open(os.path.join(photos_dir, name), "w") as handle:
            handle.write("")

    out_path = os.path.join(root, f"out_{subdir}")
    with open(out_path, "w") as handle:
        handle.write("MESA output\n")
        handle.write(f"termination code: {termcode}\n")

    return root


class TestCheckGridIntegration(unittest.TestCase):
    def setUp(self):
        self.old_cwd = os.getcwd()
        os.chdir(TEST_DATA)

    def tearDown(self):
        restart_path = os.path.join(TEST_DATA, "grid_restart")
        if os.path.isfile(restart_path):
            os.remove(restart_path)

        os.chdir(self.old_cwd)

    def test_check_grid(self):
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

    def test_check_grid_pre_CHeX(self):
        output = subprocess.run(
            [
                "check-grid",
                ".",
                "--out-file",
                "../out_{}",
                "--make-restart-file",
                "pre-CHeX",
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

    def test_check_grid_redoRC(self):
        output = subprocess.run(
            [
                "check-grid",
                ".",
                "--out-file",
                "../out_{}",
                "--make-restart-file",
                "redoRC",
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

        self.assertEqual("", restart_text)


class TestCheckGridUnit(unittest.TestCase):

    def run_check_grid(self, tmpdir, argv):
        old_argv = sys.argv
        sys.argv = argv
        stdout = io.StringIO()
        try:
            with contextlib.redirect_stdout(stdout):
                rc = cg.run()
        finally:
            sys.argv = old_argv

        self.assertEqual(rc, 0)
        output = stdout.getvalue().replace("\r\n", "\n")
        if "--make-restart-file" in argv:
            restart_path = os.path.join(tmpdir, "grid_restart")
            with open(restart_path, "r") as handle:
                restart_text = handle.read()
        else:
            restart_text = None

        return output, restart_text

    def test_get_termination_code_running(self):
        lines = [
            "         10   5.477527   5057.577 -20.882435 -20.882435   1.000000   1.000000   0.700000   0.001010   0.280000  -8.024213    559      1\n",
        ]
        self.assertEqual(cg.get_termination_code(lines), "Running")

    def test_get_termination_code_normal(self):
        lines = [
            "         10   5.477527   5057.577 -20.882435 -20.882435   1.000000   1.000000   0.700000   0.001010   0.280000  -8.024213    559      1\n",
            "termination code: max_model_number\n",
        ]
        self.assertEqual(cg.get_termination_code(lines), "max_model_number")

    def test_get_termination_code_stop(self):
        lines = [
            "         10   5.477527   5057.577 -20.882435 -20.882435   1.000000   1.000000   0.700000   0.001010   0.280000  -8.024213    559      1\n",
            "STOP logQ_min_limit\n",
        ]
        self.assertEqual(cg.get_termination_code(lines), "logQ_min_limit")

    def test_get_termination_code_backtrace(self):
        lines = [
            "         10   5.477527   5057.577 -20.882435 -20.882435   1.000000   1.000000   0.700000   0.001010   0.280000  -8.024213    559      1\n",
            "Backtrace\n",
        ]
        self.assertEqual(cg.get_termination_code(lines), "Error")

    def test_get_termination_code_mesh_failed(self):
        lines = [
            "         10   5.477527   5057.577 -20.882435 -20.882435   1.000000   1.000000   0.700000   0.001010   0.280000  -8.024213    559      1\n",
            "adjust_mesh_failed\n",
        ]
        self.assertEqual(cg.get_termination_code(lines), "mesh_failed")

    def test_get_termination_code_mesh_plan_problem(self):
        lines = [
            "         10   5.477527   5057.577 -20.882435 -20.882435   1.000000   1.000000   0.700000   0.001010   0.280000  -8.024213    559      1\n",
            "mesh_plan problem\n",
        ]
        self.assertEqual(cg.get_termination_code(lines), "mesh_failed")

    def test_get_termination_code_min_timestep_limit(self):
        lines = [
            "         10   5.477527   5057.577 -20.882435 -20.882435   1.000000   1.000000   0.700000   0.001010   0.280000  -8.024213    559      1\n",
            "stopping because of problems dt < min_timestep_limit\n",
        ]
        self.assertEqual(cg.get_termination_code(lines), "min_timestep_limit")

    def test_get_termination_code_eos_limits(self):
        lines = [
            "         10   5.477527   5057.577 -20.882435 -20.882435   1.000000   1.000000   0.700000   0.001010   0.280000  -8.024213    559      1\n",
            "stop because hit EOS limits\n",
        ]
        self.assertEqual(cg.get_termination_code(lines), "EOS_limits")

    def test_get_termination_code_stopping_because_of_problems(self):
        lines = [
            "         10   5.477527   5057.577 -20.882435 -20.882435   1.000000   1.000000   0.700000   0.001010   0.280000  -8.024213    559      1\n",
            "stopping because of problems -- nonzero_ierr\n",
        ]
        self.assertEqual(cg.get_termination_code(lines), "nonzero_ierr")

    def test_get_termination_code_nonzero_ierr(self):
        lines = [
            "         10   5.477527   5057.577 -20.882435 -20.882435   1.000000   1.000000   0.700000   0.001010   0.280000  -8.024213    559      1\n",
            "terminated evolution: nonzero_ierr\n",
        ]
        self.assertEqual(cg.get_termination_code(lines), "nonzero_ierr")

    def test_count_in_dict(self):
        info = {
            "0000": {"mesa_termcode": "max_model_number"},
            "0001": {"mesa_termcode": "max_model_number"},
            "0002": {"mesa_termcode": "Running"},
        }

        cnt, dct = cg.count_in_dict(info, key="mesa_termcode")

        self.assertEqual(cnt, Counter({"max_model_number": 2, "Running": 1}))
        self.assertEqual(dct["max_model_number"], ["0000", "0001"])
        self.assertEqual(dct["Running"], ["0002"])

    def test_count_in_dict_with_filter_mainkeys(self):
        info = {
            "0000": {"mesa_termcode": "max_model_number", "CHeX": "pre-CHeX"},
            "0001": {"mesa_termcode": "max_model_number", "CHeX": "post-CHeX"},
            "0002": {"mesa_termcode": "Running", "CHeX": "pre-CHeX"},
        }

        cnt, dct = cg.count_in_dict(
            info,
            key="CHeX",
            filter_mainkeys=["0000", "0001"],
        )

        self.assertEqual(cnt, Counter({"pre-CHeX": 1, "post-CHeX": 1}))
        self.assertEqual(dct["pre-CHeX"], ["0000"])
        self.assertEqual(dct["post-CHeX"], ["0001"])

    def test_read_hist_first_last_row(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "history.data")
            write_mesa_history(path)

            data = cg.read_hist_first_last_row(path)

            self.assertEqual(tuple(data.dtype.names), ("model_number", "center_h1", "center_he4", "mass_conv_core"))
            self.assertEqual(1, data.model_number[0])
            self.assertEqual(2600, data.model_number[-1])
            self.assertAlmostEqual(1.0, data.center_h1[0])
            self.assertAlmostEqual(0.0, data.center_h1[-1])
            self.assertAlmostEqual(0.0, data.center_he4[0])
            self.assertAlmostEqual(0.0, data.center_he4[-1])

    def test_read_hist_first_last_row_incomplete_last_row(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "history.data")

            write_mesa_history(path)
            with open(path, "a") as handle:
                # Incomplete final line.
                handle.write("                                    3002                   0.000000000000E")

            data = cg.read_hist_first_last_row(path)

            self.assertEqual(
                1,
                data.model_number[0],
            )
            self.assertEqual(2600, data.model_number[-1])

    def test_get_history_path(self):
        args = make_args(["grid_dir"])
        path = cg.get_history_path("0000", args)
        self.assertEqual(path, os.path.join("grid_dir", "0000", "LOGS/history.data"))

    def test_get_history_path_with_format(self):
        args = make_args(["grid_dir", "--history-file", "history_{}.data"])
        path = cg.get_history_path("0000", args)
        self.assertEqual(path, os.path.join("grid_dir", "0000", "LOGS/history_0000.data"))

    def test_get_history_path_glob_single_match(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logs_dir = os.path.join(tmpdir, "0000", "LOGS")
            os.makedirs(logs_dir, exist_ok=True)

            history_path = os.path.join(logs_dir, "history_0000.data")
            with open(history_path, "w") as handle:
                handle.write("")

            args = make_args([tmpdir, "--history-file", "*/history_*.data"])
            path = cg.get_history_path("0000", args)

            self.assertEqual(os.path.abspath(path), os.path.abspath(history_path))

    def test_get_history_path_glob_no_match(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logs_dir = os.path.join(tmpdir, "0000", "LOGS")
            os.makedirs(logs_dir, exist_ok=True)

            args = make_args([tmpdir, "--history-file", "*/history_*.data"])
            path = cg.get_history_path("0000", args)

            self.assertEqual(path, "")

    def test_get_history_path_glob_multiple_matches(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logs_dir = os.path.join(tmpdir, "0000", "LOGS")
            os.makedirs(logs_dir, exist_ok=True)

            with open(os.path.join(logs_dir, "history_0000.data"), "w") as handle:
                handle.write("")
            with open(os.path.join(logs_dir, "history_0001.data"), "w") as handle:
                handle.write("")

            args = make_args([tmpdir, "--history-file", "*/history_*.data"])

            with self.assertRaises(ValueError):
                cg.get_history_path("0000", args)

    def test_get_mesa_termcode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            make_grid_dir(tmpdir, subdir="0000", termcode="max_model_number")

            args = make_args([tmpdir])
            subdir, termcode = cg.get_mesa_termcode("0000", args)

            self.assertEqual(subdir, "0000")
            self.assertEqual(termcode, "max_model_number")

    def test_get_mesa_termcode_not_run(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(os.path.join(tmpdir, "0000"), exist_ok=True)

            args = make_args([tmpdir])
            subdir, termcode = cg.get_mesa_termcode("0000", args)

            self.assertEqual(subdir, "0000")
            self.assertEqual(termcode, "NotRun")

    def test_get_mesa_termcode_full_termcode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = os.path.join(tmpdir, "0000")
            os.makedirs(run_dir, exist_ok=True)

            out_path = os.path.join(tmpdir, "out_0000")
            with open(out_path, "w") as handle:
                handle.write("early line\n")
                handle.write("termination code: early_code\n")
                handle.write("x\n" * 20000)
                handle.write("\n")

            args = make_args([tmpdir, "--full-termcode"])
            subdir, termcode = cg.get_mesa_termcode("0000", args)

            self.assertEqual(subdir, "0000")
            self.assertEqual(termcode, "early_code")

    def test_get_mesa_termcode_not_full_termcode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = os.path.join(tmpdir, "0000")
            os.makedirs(run_dir, exist_ok=True)

            out_path = os.path.join(tmpdir, "out_0000")
            with open(out_path, "w") as handle:
                handle.write("early line\n")
                handle.write("termination code: early_code\n")
                handle.write("x\n" * 20000)
                handle.write("\n")

            args = make_args([tmpdir])
            subdir, termcode = cg.get_mesa_termcode("0000", args)

            self.assertEqual(subdir, "0000")
            self.assertEqual(termcode, "Running")

    def test_get_restart_photo_no_photos(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = os.path.join(tmpdir, "0000")
            os.makedirs(run_dir, exist_ok=True)

            args = make_args([tmpdir])
            photo = cg.get_restart_photo("0000", "last", 1000, args)

            self.assertEqual(photo, "full_restart")

    def test_get_restart_photo_last(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            photos_dir = os.path.join(tmpdir, "0000", "photos")
            os.makedirs(photos_dir, exist_ok=True)

            for name in ("100", "500", "900"):
                with open(os.path.join(photos_dir, name), "w") as handle:
                    handle.write("")

            args = make_args([tmpdir, "--min-photo-delta", "200"])
            photo = cg.get_restart_photo("0000", "last", 1000, args)

            self.assertEqual(photo, "500")

    def test_get_restart_photo_last_no_photo_meets_delta(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            photos_dir = os.path.join(tmpdir, "0000", "photos")
            os.makedirs(photos_dir, exist_ok=True)

            for name in ("900", "950", "990"):
                with open(os.path.join(photos_dir, name), "w") as handle:
                    handle.write("")

            args = make_args([tmpdir, "--min-photo-delta", "200"])
            photo = cg.get_restart_photo("0000", "last", 1000, args)

            self.assertEqual(photo, "full_restart")

    def test_get_restart_photo_last_modulo_photo(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            photos_dir = os.path.join(tmpdir, "0000", "photos")
            os.makedirs(photos_dir, exist_ok=True)

            for name in ("x10", "x50", "x90", "100", "200"):
                with open(os.path.join(photos_dir, name), "w") as handle:
                    handle.write("")

            args = make_args([tmpdir, "--min-photo-delta", "100"])
            photo = cg.get_restart_photo("0000", "last", 249, args)

            self.assertEqual(photo, "100")

    def test_get_restart_photo_last_modulo_photo_no_delta(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            photos_dir = os.path.join(tmpdir, "0000", "photos")
            os.makedirs(photos_dir, exist_ok=True)

            with open(os.path.join(photos_dir, "x50"), "w") as handle:
                handle.write("")

            args = make_args([tmpdir, "--min-photo-delta", "51"])
            photo = cg.get_restart_photo("0000", "last", 1000, args)

            self.assertEqual(photo, "full_restart")

            args = make_args([tmpdir, "--min-photo-delta", "49"])
            photo = cg.get_restart_photo("0000", "last", 1000, args)

            self.assertEqual(photo, "x50")

    def test_get_restart_photo_full_restart_kind(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = os.path.join(tmpdir, "0000")
            os.makedirs(run_dir, exist_ok=True)

            args = make_args([tmpdir])
            photo = cg.get_restart_photo("0000", "full_restart", 1000, args)

            self.assertEqual(photo, "full_restart")

    def test_get_restart_photo_pre_rc_no_history(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            photos_dir = os.path.join(tmpdir, "0000", "photos")
            os.makedirs(photos_dir, exist_ok=True)

            with open(os.path.join(photos_dir, "500"), "w") as handle:
                handle.write("")

            args = make_args([tmpdir, "--min-photo-delta", "200"])
            photo = cg.get_restart_photo("0000", "preRC", 1000, args)

            self.assertEqual(photo, "full_restart")

    def test_get_restart_photo_pre_rc_invalid_history(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = os.path.join(tmpdir, "0000")
            logs_dir = os.path.join(run_dir, "LOGS")
            photos_dir = os.path.join(run_dir, "photos")

            os.makedirs(logs_dir, exist_ok=True)
            os.makedirs(photos_dir, exist_ok=True)

            with open(os.path.join(logs_dir, "history.data"), "w") as handle:
                handle.write("not a valid MESA history file\n")

            with open(os.path.join(photos_dir, "500"), "w") as handle:
                handle.write("")

            args = make_args([tmpdir, "--min-photo-delta", "200"])
            photo = cg.get_restart_photo("0000", "preRC", 1000, args)

            self.assertEqual(photo, "full_restart")

    def test_run_list_all(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            make_grid_dir(tmpdir, subdir="0000", termcode="max_model_number")

            argv = [
                "check-grid",
                tmpdir,
                "--out-file",
                "../out_{}",
                "--list-all",
            ]

            output, restart_text = self.run_check_grid(tmpdir, argv)

            self.assertIn("termination_code", output)
            self.assertIn("max_model_number", output)
            self.assertIn("post-CHeX", output)
            self.assertIn("0000", output)
            self.assertIn("2600", output)

    def test_run_make_restart_file_last(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            make_grid_dir(tmpdir, subdir="0000", termcode="max_model_number")

            argv = ["check-grid", tmpdir, "--out-file", "../out_{}", "--make-restart-file", "last"]

            output, restart_text = self.run_check_grid(tmpdir, argv)

            self.assertEqual(restart_text, "0000 x500\n")

    def test_run_make_restart_file_pre_CHeX(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            make_grid_dir(tmpdir, subdir="0000", termcode="max_model_number", truncate=1)

            argv = [
                "check-grid",
                tmpdir,
                "--out-file",
                "../out_{}",
                "--make-restart-file",
                "pre-CHeX",
            ]

            output, restart_text = self.run_check_grid(tmpdir, argv)
            self.assertEqual(restart_text, "0000 x500\n")

    def test_run_make_restart_file_pre_CHeX_skips_CHeX(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            make_grid_dir(tmpdir, subdir="0000", termcode="max_model_number")

            argv = [
                "check-grid",
                tmpdir,
                "--out-file",
                "../out_{}",
                "--make-restart-file",
                "pre-CHeX",
            ]

            output, restart_text = self.run_check_grid(tmpdir, argv)
            self.assertEqual(restart_text, "")

    def test_run_make_restart_file_redoRC_skips_pre_CHeX(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            make_grid_dir(tmpdir, subdir="0000", termcode="max_model_number", truncate=1)

            argv = [
                "check-grid",
                tmpdir,
                "--out-file",
                "../out_{}",
                "--make-restart-file",
                "redoRC",
            ]

            output, restart_text = self.run_check_grid(tmpdir, argv)

            self.assertEqual(restart_text, "")
