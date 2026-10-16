#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import pathlib
import shutil
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
from wsssss import load_data as ld
from wsssss.bin.gyre_driver import gyre_driver
from ..common import check_required_environment, check_if_running_in_CI

must_have_environ = ["GYRE_DIR"]
missing_environ = check_required_environment(must_have_environ)

test_data = pathlib.Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "gyre")))
test_gyre_version = pathlib.Path(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "dummy_software"))
)


def assert_allclose_recarray(arr1, arr2):
    for colname in arr1.dtype.names:
        np.testing.assert_allclose(arr1[colname], arr2[colname])


def make_args(argv):
    """
    Build an argparse Namespace using gyre_driver.get_parser().
    """
    parser = gyre_driver.get_parser()
    args = parser.parse_args(argv)
    return args


class TestGyreDriverUnit(unittest.TestCase):
    """
    Unit tests that do not require a real GYRE installation.
    """

    def test_make_args_defaults(self):
        args = make_args(["0", "MESA", "profile1.data.GYRE"])

        self.assertEqual(args.ll, "0")
        self.assertEqual(args.filetype, "MESA")
        self.assertEqual(args.files, ["profile1.data.GYRE"])
        self.assertEqual(args.gyre, "G9")
        self.assertFalse(args.pmode)
        self.assertFalse(args.save_modes)
        self.assertFalse(args.verbose)
        self.assertEqual(args.source, "")
        self.assertFalse(args.lenient)
        self.assertEqual(args.out_dir, "gyre_out")
        self.assertEqual(args.in_dir, "gyre_in")
        self.assertEqual(args.base_in, "")
        self.assertFalse(args.nosort)
        self.assertFalse(args.no_merge)
        self.assertFalse(args.skip_calc)
        self.assertFalse(args.skip_existing)
        self.assertEqual(args.min_numax, 0)
        self.assertEqual(args.summary_item_list, "l,n_pg,n_p,n_g,freq,E_norm,M_star,R_star,L_star,E")
        self.assertEqual(args.n_sig_lo, 2)
        self.assertEqual(args.n_sig_hi, 2)
        self.assertEqual(args.f_nfreq, 1)
        self.assertEqual(args.summary_suffix, ".sgyre_l")
        self.assertFalse(args.no_output)
        self.assertFalse(args.rotation)

    def test_make_args_basic(self):
        args = make_args(["mode012", "LOSC", "file1", "file2", "--gyre", "G7", "--pmode", "--save-modes", "--verbose"])

        self.assertEqual(args.ll, "mode012")
        self.assertEqual(args.filetype, "LOSC")
        self.assertEqual(args.files, ["file1", "file2"])
        self.assertEqual(args.gyre, "G7")
        self.assertTrue(args.pmode)
        self.assertTrue(args.save_modes)
        self.assertTrue(args.verbose)

    def test_make_args_flags(self):
        args = make_args(
            [
                "01",
                "MESA",
                "profile",
                "--source",
                "env.sh",
                "--lenient",
                "--out-dir",
                "out",
                "--in-dir",
                "in",
                "--base-in",
                "base.in",
                "--nosort",
                "--no-merge",
                "--skip-calc",
                "--skip-existing",
                "--min-numax",
                "45",
                "--summary-item-list",
                "l,freq",
                "--n-sig-lo",
                "1.5",
                "--n-sig-hi",
                "2.5",
                "--f-nfreq",
                "2",
                "--summary-suffix",
                ".test",
                "--no-output",
                "--rotation",
            ]
        )

        self.assertEqual(args.source, "env.sh")
        self.assertTrue(args.lenient)
        self.assertEqual(args.out_dir, "out")
        self.assertEqual(args.in_dir, "in")
        self.assertEqual(args.base_in, "base.in")
        self.assertTrue(args.nosort)
        self.assertTrue(args.no_merge)
        self.assertTrue(args.skip_calc)
        self.assertTrue(args.skip_existing)
        self.assertEqual(args.min_numax, 45)
        self.assertEqual(args.summary_item_list, "l,freq")
        self.assertEqual(args.n_sig_lo, 1.5)
        self.assertEqual(args.n_sig_hi, 2.5)
        self.assertEqual(args.f_nfreq, 2)
        self.assertEqual(args.summary_suffix, ".test")
        self.assertTrue(args.no_output)
        self.assertTrue(args.rotation)

    def test_sort_files(self):
        files = ["2.data", "1.data", "12.data", "111.data"]
        sorted_files = gyre_driver.sort_files(files)

        self.assertEqual(sorted_files, ["1.data", "2.data", "12.data", "111.data"])

    def test_sort_files_paths(self):
        files = ["2.data", "1.data", "12.data", "111.data"]
        files = [pathlib.Path(file) for file in files]
        sorted_files = gyre_driver.sort_files(files)

        self.assertEqual([str(_) for _ in sorted_files], ["1.data", "2.data", "12.data", "111.data"])

    def test_summary_path_out_dir_nonempty(self):
        args = make_args(["0", "MESA", "profile1.data.GYRE", "--out-dir", "gyre_out"])

        model_name = pathlib.Path("dir/profile.data")
        summary_file = gyre_driver.summary_path(model_name, 1, ".sgyre_l", args)

        self.assertEqual(summary_file, pathlib.Path("gyre_out/profile.data.sgyre_l1"))

    def test_get_gyre_environment_specific_version(self):
        gyre_dir = test_gyre_version / "gyre-9.1.1"
        with mock.patch.dict(os.environ, {f"GYRE_DIR_G9": str(gyre_dir)}, clear=False):
            args = make_args(["0", "MESA"])
            path, version = gyre_driver.get_gyre(args, True)

        self.assertEqual(version, "G9")
        self.assertEqual(path, gyre_dir / "bin" / "gyre")

    def test_get_gyre_checks_version(self):
        gyre_dir = test_gyre_version / "gyre-9.1.1"

        with mock.patch.dict(os.environ, {"GYRE_DIR": str(gyre_dir)}, clear=False):
            args = make_args(["0", "MESA", "--gyre", "G9"])
            path, version = gyre_driver.get_gyre(args, True)

        self.assertEqual(version, "G9")
        self.assertEqual(path, gyre_dir / "bin" / "gyre")

    def test_get_gyre_version_mismatch_raises(self):
        gyre_dir = test_gyre_version / "gyre-7.2.1"

        with mock.patch.dict(os.environ, {"GYRE_DIR": str(gyre_dir)}, clear=False):
            args = make_args(["0", "MESA", "--gyre", "G9"])
            with self.assertRaises(ValueError):
                gyre_driver.get_gyre(args, True)

    def test_get_gyre_lenient(self):
        gyre_dir = test_gyre_version / "gyre-7.2.1"

        with mock.patch.dict(os.environ, {"GYRE_DIR": str(gyre_dir)}, clear=False):
            args = make_args(["0", "MESA", "--gyre", "G9", "--lenient"])
            path, version = gyre_driver.get_gyre(args, True, True)

        self.assertEqual(version, "G7")
        self.assertEqual(path, gyre_dir / "bin" / "gyre")

    def test_get_gyre_missing_version_string(self):
        tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(tmp_dir.cleanup)

        gyre_dir = pathlib.Path(tmp_dir.name) / "GYRE"
        src_common = gyre_dir / "src" / "common"
        bin_dir = gyre_dir / "bin"
        src_common.mkdir(parents=True, exist_ok=True)
        bin_dir.mkdir(parents=True, exist_ok=True)

        version_file = src_common / "gyre_version.fpp"
        with open(version_file, "w") as handle:
            handle.write("  no version here\n")

        with mock.patch.dict(os.environ, {"GYRE_DIR": str(gyre_dir)}, clear=False):
            args = make_args(["0", "MESA"])
            with self.assertRaises(ValueError):
                gyre_driver.get_gyre(args, True)

    def test_get_gyre_no_environment(self):
        args = make_args(["0", "MESA"])
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(KeyError):
                gyre_driver.get_gyre(args, True)

    def test_check_args_basic(self):
        tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(tmp_dir.cleanup)

        gyre_dir = test_gyre_version / "gyre-9.1.1"

        old_cwd = os.getcwd()
        os.chdir(tmp_dir.name)
        self.addCleanup(os.chdir, old_cwd)

        args = make_args(
            [
                "012",
                "MESA",
                "profile1.data.GYRE",
                "--out-dir",
                "out",
                "--in-dir",
                "in",
                "--base-in",
                "base.in",
            ]
        )

        with mock.patch.dict(os.environ, {"GYRE_DIR": str(gyre_dir)}, clear=False):
            args = gyre_driver.check_args(args)

        self.assertEqual(args.ll, [0, 1, 2])
        self.assertEqual(args.files, [pathlib.Path("profile1.data.GYRE")])
        self.assertEqual(args.out_dir, pathlib.Path("out"))
        self.assertEqual(args.in_dir, pathlib.Path("in"))
        self.assertEqual(args.base_in, pathlib.Path("base.in"))
        self.assertEqual(args.gyre_adin_template, "in/gyre_ad.in_")
        self.assertEqual(args.original_gyre, "G9")
        self.assertTrue(args.out_dir.exists())
        self.assertTrue(args.in_dir.exists())

    def test_check_args_mode_overrides_save_modes(self):
        gyre_dir = test_gyre_version / "gyre-9.1.1"

        args = make_args(["mode01", "MESA", "profile1.data.GYRE"])

        with mock.patch.dict(os.environ, {"GYRE_DIR": str(gyre_dir)}, clear=False):
            args = gyre_driver.check_args(args)

        self.assertTrue(args.save_modes)
        self.assertEqual(args.ll, [0, 1])

    def test_check_args_adds_l0(self):
        gyre_dir = test_gyre_version / "gyre-9.1.1"

        args = make_args(["123", "MESA", "profile1.data.GYRE"])

        with mock.patch.dict(os.environ, {"GYRE_DIR": str(gyre_dir)}, clear=False):
            args = gyre_driver.check_args(args)

        self.assertEqual(args.ll, [0, 1, 2, 3])

    def test_check_args_sorts_files(self):
        gyre_dir = test_gyre_version / "gyre-9.1.1"

        args = make_args(["0", "MESA", "b.data", "a10.data", "a2.data"])

        with mock.patch.dict(os.environ, {"GYRE_DIR": str(gyre_dir)}, clear=False):
            args = gyre_driver.check_args(args)

        self.assertEqual(args.files, [pathlib.Path("b.data"), pathlib.Path("a2.data"), pathlib.Path("a10.data")])

    def test_check_args_skip_existing(self):
        tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(tmp_dir.cleanup)
        old_cwd = os.getcwd()
        os.chdir(tmp_dir.name)
        self.addCleanup(os.chdir, old_cwd)

        gyre_dir = test_gyre_version / "gyre-9.1.1"

        args = make_args(
            [
                "0",
                "MESA",
                "model1.data",
                "model2.data",
                "--out-dir",
                "out",
                "--skip-existing",
            ]
        )

        with mock.patch.dict(os.environ, {"GYRE_DIR": str(gyre_dir)}, clear=False):
            args = gyre_driver.check_args(args)

        existing = pathlib.Path("out") / "model1.data.sgyre_l"
        existing.parent.mkdir(exist_ok=True, parents=True)
        existing.write_text("dummy\n")

        args = make_args(
            [
                "0",
                "MESA",
                "model1.data",
                "model2.data",
                "--out-dir",
                "out",
                "--skip-existing",
            ]
        )

        with mock.patch.dict(os.environ, {"GYRE_DIR": str(gyre_dir)}, clear=False):
            args = gyre_driver.check_args(args)

        self.assertEqual(args.files, [pathlib.Path("model2.data")])

    def test_get_nu_max_dnu_dp_mesa_l0(self):
        profile = os.path.join(test_data, "LOGS", "profile10.data.GYRE")
        args = make_args(["0", "MESA", profile])

        nu_max, Dnu, DP = gyre_driver.get_nu_max_dnu_dp(args, profile, 0)

        self.assertTrue(np.isfinite(nu_max))
        self.assertTrue(np.isfinite(Dnu))
        self.assertTrue(np.isinf(DP))

        self.assertGreater(nu_max, 0)
        self.assertGreater(Dnu, 0)

    def test_get_nu_max_dnu_dp_mesa_l1(self):
        profile = os.path.join(test_data, "LOGS", "profile10.data.GYRE")
        args = make_args(["1", "MESA", profile])

        nu_max, Dnu, DP = gyre_driver.get_nu_max_dnu_dp(args, profile, 1)

        self.assertTrue(np.isfinite(nu_max))
        self.assertTrue(np.isfinite(Dnu))
        self.assertTrue(np.isfinite(DP))

        self.assertGreater(nu_max, 0)
        self.assertGreater(Dnu, 0)
        self.assertGreater(DP, 0)

    def test_get_nu_max_dnu_dp_fgong_not_implemented(self):
        profile = os.path.join(test_data, "LOGS", "profile10.data.GYRE")
        args = make_args(["0", "MESA", profile])
        args.filetype = "FGONG"

        with self.assertRaises(NotImplementedError):
            gyre_driver.get_nu_max_dnu_dp(args, profile, 0)

    def test_get_nu_max_dnu_dp_invalid_filetype(self):
        profile = os.path.join(test_data, "LOGS", "profile10.data.GYRE")
        args = make_args(["0", "MESA", profile])
        args.filetype = "BAD"

        with self.assertRaises(ValueError):
            gyre_driver.get_nu_max_dnu_dp(args, profile, 0)

    def test_calc_scan_l0(self):
        profile = os.path.join(test_data, "LOGS", "profile10.data.GYRE")
        args = make_args(["0", "MESA", profile])

        fmin, fmax, n_freq, num_scan, nu_max = gyre_driver.calc_scan(profile, 0, args)

        self.assertEqual(num_scan, 1)
        self.assertTrue(np.isfinite(nu_max))
        self.assertGreater(nu_max, 0)
        self.assertLess(fmin, fmax)
        self.assertTrue(np.isscalar(n_freq))
        self.assertGreater(int(n_freq), 0)

    def test_calc_scan_l1(self):
        profile = os.path.join(test_data, "LOGS", "profile10.data.GYRE")
        ref_summary = os.path.join(test_data, "profile10.data.GYRE.sgyre_l")

        tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(tmp_dir.cleanup)

        out_dir = pathlib.Path(tmp_dir.name) / "gyre_out"
        out_dir.mkdir(parents=True, exist_ok=True)

        l0_summary = out_dir / "profile10.data.GYRE.sgyre_l0"
        shutil.copy2(ref_summary, l0_summary)

        args = make_args(["1", "MESA", profile, "--out-dir", str(out_dir)])
        model_name = pathlib.Path(profile)

        fmin, fmax, n_freq, num_scan, nu_max = gyre_driver.calc_scan(model_name, 1, args)

        self.assertTrue(num_scan >= 1)
        self.assertEqual(np.atleast_1d(fmin).size, num_scan)
        self.assertEqual(np.atleast_1d(fmax).size, num_scan)
        self.assertEqual(np.atleast_1d(n_freq).size, num_scan)

        self.assertTrue(np.all(np.atleast_1d(fmin) < np.atleast_1d(fmax)))
        self.assertTrue(np.all(np.atleast_1d(n_freq) > 0))

    def test_write_gyre_adin_basic(self):
        tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(tmp_dir.cleanup)

        in_dir = pathlib.Path(tmp_dir.name) / "gyre_in"
        out_dir = pathlib.Path(tmp_dir.name) / "gyre_out"
        in_dir.mkdir(parents=True, exist_ok=True)
        out_dir.mkdir(parents=True, exist_ok=True)

        base_in = in_dir / "base.in"
        base_in.write_text("! base inlist\n")

        model_name = test_data / "LOGS" / "profile10.data.GYRE"

        args = make_args(["0", "MESA", str(model_name), "--out-dir", str(out_dir), "--in-dir", str(in_dir)])
        args.gyre_adin_template = str(in_dir / "gyre_ad.in_")
        args.base_in = base_in

        gyre_adin = gyre_driver.write_gyre_adin(
            model_name,
            0,
            "MESA",
            args.summary_suffix,
            False,
            "LINEAR",
            1.0,
            2.0,
            10,
            args,
        )

        self.assertTrue(gyre_adin.exists())

        text = gyre_adin.read_text()
        self.assertIn("! base inlist\n", text)
        self.assertIn("model_type = 'EVOL'", text)
        self.assertIn("file_format = 'MESA'", text)
        self.assertIn("grid_type = 'LINEAR'", text)
        self.assertIn("freq_min =   1.0", text)
        self.assertIn("freq_max =   2.0", text)
        self.assertIn("n_freq =     10", text)
        self.assertIn("&mode", text)
        self.assertIn("l = 0", text)

    def test_write_gyre_adin_save_modes(self):
        tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(tmp_dir.cleanup)

        in_dir = pathlib.Path(tmp_dir.name) / "gyre_in"
        out_dir = pathlib.Path(tmp_dir.name) / "gyre_out"
        in_dir.mkdir(parents=True, exist_ok=True)
        out_dir.mkdir(parents=True, exist_ok=True)

        model_name = test_data / "LOGS" / "profile10.data.GYRE"

        args = make_args(
            [
                "0",
                "MESA",
                str(model_name),
                "--out-dir",
                str(out_dir),
                "--in-dir",
                str(in_dir),
                "--gyre",
                "G9",
                "--save-modes",
            ]
        )
        args.gyre_adin_template = str(in_dir / "gyre_ad.in_")

        gyre_adin = gyre_driver.write_gyre_adin(
            model_name,
            0,
            "MESA",
            args.summary_suffix,
            True,
            "LINEAR",
            1.0,
            2.0,
            10,
            args,
        )

        text = gyre_adin.read_text()
        self.assertIn("detail_template", text)
        self.assertIn("detail_item_list", text)
        self.assertIn("%ID", text)

    def test_write_gyre_adin_rotation(self):
        tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(tmp_dir.cleanup)

        in_dir = pathlib.Path(tmp_dir.name) / "gyre_in"
        in_dir.mkdir(parents=True, exist_ok=True)

        model_name = pathlib.Path("profile10.data.GYRE")

        args = make_args(["1", "MESA", str(model_name), "--rotation"])
        args.gyre_adin_template = str(in_dir / "gyre_ad.in_")

        gyre_adin = gyre_driver.write_gyre_adin(
            model_name,
            1,
            "MESA",
            args.summary_suffix,
            False,
            "INVERSE",
            1.0,
            2.0,
            10,
            args,
        )

        text = gyre_adin.read_text()
        self.assertIn("grid_frame ='COROT_O'", text)
        self.assertIn("m = -1", text)
        self.assertIn("m = 0", text)
        self.assertIn("m = 1", text)

    def test_write_gyre_adin_multi_scan(self):
        tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(tmp_dir.cleanup)

        in_dir = pathlib.Path(tmp_dir.name) / "gyre_in"
        in_dir.mkdir(parents=True, exist_ok=True)

        model_name = pathlib.Path("profile10.data.GYRE")

        args = make_args(["1", "MESA", str(model_name)])
        args.gyre_adin_template = str(in_dir / "gyre_ad.in_")

        fmin = np.array([1.0, 2.0])
        fmax = np.array([2.0, 3.0])
        n_freq = np.array([10, 20])

        gyre_adin = gyre_driver.write_gyre_adin(
            model_name,
            1,
            "MESA",
            args.summary_suffix,
            False,
            "INVERSE",
            fmin,
            fmax,
            n_freq,
            args,
        )

        text = gyre_adin.read_text()
        self.assertEqual(text.count("&scan"), 2)
        self.assertIn("freq_min =   1.0", text)
        self.assertIn("freq_max =   3.0", text)
        self.assertIn("n_freq =     20", text)

    def test_write_gyre_adin_missing_base_in(self):
        tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(tmp_dir.cleanup)

        in_dir = pathlib.Path(tmp_dir.name) / "gyre_in"
        in_dir.mkdir(parents=True, exist_ok=True)

        model_name = pathlib.Path("profile10.data.GYRE")

        args = make_args(["0", "MESA", str(model_name)])
        args.gyre_adin_template = str(in_dir / "gyre_ad.in_")
        args.base_in = in_dir / "missing.in"

        with self.assertRaises(FileNotFoundError):
            gyre_driver.write_gyre_adin(
                model_name,
                0,
                "MESA",
                args.summary_suffix,
                False,
                "LINEAR",
                1.0,
                2.0,
                10,
                args,
            )

    def test_write_gyre_adin_invalid_filetype(self):
        tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(tmp_dir.cleanup)

        in_dir = pathlib.Path(tmp_dir.name) / "gyre_in"
        in_dir.mkdir(parents=True, exist_ok=True)

        model_name = pathlib.Path("profile10.data.GYRE")

        args = make_args(["0", "MESA", str(model_name), "--in-dir", str(in_dir)])
        args.gyre_adin_template = str(in_dir / "gyre_ad.in_")

        with self.assertRaises(ValueError):
            gyre_driver.write_gyre_adin(
                model_name,
                0,
                "BAD",
                args.summary_suffix,
                True,
                "LINEAR",
                1.0,
                2.0,
                10,
                args,
            )

    def test_run_gyre_skip_calc(self):
        gyre_dir = test_gyre_version / "gyre-9.1.1"
        args = make_args(["1", "MESA", "profile1.data.GYRE", "--skip-calc"])

        with mock.patch.dict(os.environ, {"GYRE_DIR": str(gyre_dir)}, clear=False):
            output = gyre_driver.run_gyre(
                pathlib.Path("profile1.data.GYRE"),
                1,
                ".sgyre_l",
                pathlib.Path("gyre_ad.in_profile.data.sgyre_l1"),
                args,
            )

        self.assertTrue(str(output).startswith("skipped"))

    def test_merge_summary_parts(self):
        tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(tmp_dir.cleanup)

        out_dir = pathlib.Path(tmp_dir.name) / "gyre_out"
        out_dir.mkdir(parents=True, exist_ok=True)

        model_name = pathlib.Path("profile1.data.GYRE")
        args = make_args(["0", "MESA", str(model_name), "--out-dir", str(out_dir)])

        part1 = gyre_driver.summary_path(model_name, 1, ".part1of2" + args.summary_suffix, args)
        part2 = gyre_driver.summary_path(model_name, 1, ".part2of2" + args.summary_suffix, args)

        header = [f"header{i}\n" for i in range(6)]
        part1.write_text("".join(header) + "data1\n")
        part2.write_text("".join(header) + "data2\n")

        gyre_driver.merge_summary_parts(model_name, 1, 2, args)

        merged = gyre_driver.summary_path(model_name, 1, args.summary_suffix, args)
        self.assertTrue(merged.exists())
        self.assertFalse(part1.exists())
        self.assertFalse(part2.exists())

        lines = merged.read_text().splitlines()
        self.assertEqual(lines[:6], [line.rstrip() for line in header])
        self.assertEqual(lines[6:], ["data1", "data2"])

    def test_merge_summary_parts_keep_files(self):
        tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(tmp_dir.cleanup)

        out_dir = pathlib.Path(tmp_dir.name) / "gyre_out"
        out_dir.mkdir(parents=True, exist_ok=True)

        model_name = pathlib.Path("profile1.data.GYRE")
        args = make_args(["0", "MESA", str(model_name), "--out-dir", str(out_dir)])

        part1 = gyre_driver.summary_path(model_name, 1, ".part1of2" + args.summary_suffix, args)
        part2 = gyre_driver.summary_path(model_name, 1, ".part2of2" + args.summary_suffix, args)

        header = [f"header{i}\n" for i in range(6)]
        part1.write_text("".join(header) + "data1\n")
        part2.write_text("".join(header) + "data2\n")

        gyre_driver.merge_summary_parts(model_name, 1, 2, args, keep_files=True)

        merged = gyre_driver.summary_path(model_name, 1, args.summary_suffix, args)
        self.assertTrue(merged.exists())
        self.assertTrue(part1.exists())
        self.assertTrue(part2.exists())

    def test_merge_summary_l(self):
        tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(tmp_dir.cleanup)

        out_dir = pathlib.Path(tmp_dir.name) / "gyre_out"
        out_dir.mkdir(parents=True, exist_ok=True)

        model_name = pathlib.Path("profile1.data.GYRE")
        args = make_args(["01", "MESA", str(model_name), "--out-dir", str(out_dir)])
        args.ll = [0, 1]

        file0 = gyre_driver.summary_path(model_name, 0, args.summary_suffix, args)
        file1 = gyre_driver.summary_path(model_name, 1, args.summary_suffix, args)

        header = [f"header{i}\n" for i in range(6)]
        file0.write_text("".join(header) + "data0\n")
        file1.write_text("".join(header) + "data1\n")

        gyre_driver.merge_summary_l(model_name, args)

        merged = gyre_driver.summary_path(model_name, "", args.summary_suffix, args)
        self.assertTrue(merged.exists())
        self.assertFalse(file0.exists())
        self.assertFalse(file1.exists())

        lines = merged.read_text().splitlines()
        self.assertEqual(lines[:6], [line.rstrip() for line in header])
        self.assertEqual(lines[6:], ["data0", "data1"])

    def test_merge_summary_l_skips_missing(self):
        tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(tmp_dir.cleanup)

        out_dir = pathlib.Path(tmp_dir.name) / "gyre_out"
        out_dir.mkdir(parents=True, exist_ok=True)

        model_name = pathlib.Path("profile1.data.GYRE")
        args = make_args(["01", "MESA", str(model_name), "--out-dir", str(out_dir)])
        args.ll = [0, 1]

        file0 = gyre_driver.summary_path(model_name, 0, args.summary_suffix, args)
        header = [f"header{i}\n" for i in range(6)]
        file0.write_text("".join(header) + "data0\n")

        gyre_driver.merge_summary_l(model_name, args)

        merged = gyre_driver.summary_path(model_name, "", args.summary_suffix, args)
        self.assertTrue(merged.exists())

        lines = merged.read_text().splitlines()
        self.assertEqual(lines[:6], [line.rstrip() for line in header])
        self.assertEqual(lines[6:], ["data0"])

    def test_run_no_files(self):
        tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(tmp_dir.cleanup)

        old_cwd = os.getcwd()
        os.chdir(tmp_dir.name)
        self.addCleanup(os.chdir, old_cwd)

        gyre_dir = test_gyre_version / "gyre-9.1.1"

        args = make_args(
            ["0", "MESA", "--out-dir", "gyre_out_should_not_exist", "--in-dir", "gyre_in_should_not_exist"]
        )

        with mock.patch.dict(os.environ, {"GYRE_DIR": str(gyre_dir)}, clear=False):
            self.assertRaises(ValueError, gyre_driver.run, args)

        self.assertFalse(pathlib.Path("gyre_out_should_not_exist").exists())
        self.assertFalse(pathlib.Path("gyre_in_should_not_exist").exists())

    def test_run_missing_gyre_dir(self):
        tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(tmp_dir.cleanup)

        old_cwd = os.getcwd()
        os.chdir(tmp_dir.name)
        self.addCleanup(os.chdir, old_cwd)

        args = make_args(["0", "MESA", "profile1.data.GYRE"])

        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(KeyError):
                gyre_driver.run(args)


@unittest.skipIf(os.name == "nt", "Skipping on Windows")
@unittest.skipIf(check_if_running_in_CI(), "Can't run GYRE in CI.")
@unittest.skipIf(len(missing_environ) > 0, "GYRE environment is not available.")
class TestGyreDriver(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        gs = ld.GyreSummary(os.path.join(test_data, "profile10.data.GYRE.sgyre_l"))
        cls.ref_gs = gs

    def setUp(self):
        tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(tmp_dir.cleanup)

        old_cwd = os.getcwd()
        os.chdir(tmp_dir.name)
        self.addCleanup(os.chdir, old_cwd)

        shutil.copytree(test_data, tmp_dir.name, dirs_exist_ok=True)
        self.tmp_dir = tmp_dir.name

    def test_gyre_driver(self):
        args = make_args(["0", "MESA", "LOGS/profile10.data.GYRE", "--gyre", "G9"])
        ierr = gyre_driver.run(args)
        self.assertEqual(0, ierr)

        gs_path = os.path.join(self.tmp_dir, "gyre_out", "profile10.data.GYRE.sgyre_l")
        gs = ld.GyreSummary(gs_path)

        assert_allclose_recarray(self.ref_gs.data[self.ref_gs.data.l == 0], gs.data)
        os.remove(gs_path)

    def test_gyre_min_numax(self):
        args = make_args(["01", "MESA", "LOGS/profile10.data.GYRE", "--min-numax", "45", "--gyre", "G9"])
        ierr = gyre_driver.run(args)
        self.assertEqual(0, ierr)

        gs_path = os.path.join(self.tmp_dir, "gyre_out", "profile10.data.GYRE.sgyre_l")
        gs = ld.GyreSummary(gs_path)

        # Should only have l=0 modes as the Model's numax is 44.6
        assert_allclose_recarray(self.ref_gs.data[self.ref_gs.data.l == 0], gs.data)
        os.remove(gs_path)

    def test_lenient(self):
        args = make_args(["0", "MESA", "LOGS/profile10.data.GYRE", "--gyre", "G6", "--lenient"])
        ierr = gyre_driver.run(args)
        self.assertEqual(0, ierr)

        gs_path = os.path.join(self.tmp_dir, "gyre_out", "profile10.data.GYRE.sgyre_l")
        gs = ld.GyreSummary(gs_path)

        assert_allclose_recarray(self.ref_gs.data[self.ref_gs.data.l == 0], gs.data)
        os.remove(gs_path)

    def test_no_merge(self):
        args = make_args(["0", "MESA", "LOGS/profile10.data.GYRE", "--gyre", "G9", "--no-merge"])
        ierr = gyre_driver.run(args)
        self.assertEqual(0, ierr)

        merged_path = pathlib.Path(self.tmp_dir) / "gyre_out" / "profile10.data.GYRE.sgyre_l"
        l0_path = pathlib.Path(self.tmp_dir) / "gyre_out" / "profile10.data.GYRE.sgyre_l0"

        self.assertFalse(merged_path.exists())
        self.assertTrue(l0_path.exists())

        gs = ld.GyreSummary(str(l0_path))
        assert_allclose_recarray(self.ref_gs.data[self.ref_gs.data.l == 0], gs.data)

        os.remove(l0_path)

    def test_save_modes(self):
        args = make_args(["mode0", "MESA", "LOGS/profile10.data.GYRE", "--gyre", "G9"])
        ierr = gyre_driver.run(args)
        self.assertEqual(0, ierr)

        gs_path = pathlib.Path(self.tmp_dir) / "gyre_out" / "profile10.data.GYRE.sgyre_l"
        gs = ld.GyreSummary(str(gs_path))
        assert_allclose_recarray(self.ref_gs.data[self.ref_gs.data.l == 0], gs.data)

        mode_files = list((pathlib.Path(self.tmp_dir) / "gyre_out").glob("*_l0_*.mgyre"))
        self.assertTrue(len(mode_files) > 0)

        os.remove(gs_path)
        for mode_file in mode_files:
            os.remove(mode_file)

    def test_skip_existing(self):
        args = make_args(["0", "MESA", "LOGS/profile10.data.GYRE", "--gyre", "G9"])
        ierr = gyre_driver.run(args)
        self.assertEqual(0, ierr)

        gs_path = pathlib.Path(self.tmp_dir) / "gyre_out" / "profile10.data.GYRE.sgyre_l"
        self.assertTrue(gs_path.exists())

        args = make_args(
            [
                "0",
                "MESA",
                "LOGS/profile10.data.GYRE",
                "--gyre",
                "G9",
                "--skip-existing",
            ]
        )
        ierr = gyre_driver.run(args)
        self.assertEqual(0, ierr)

        self.assertTrue(gs_path.exists())
        os.remove(gs_path)

    def test_gyre_versions(self):
        original_gyre_dir = os.environ["GYRE_DIR"]

        try:
            possible_versions = os.listdir(f"{os.environ['GYRE_DIR']}/..")
            tested_versions = []

            for gyredir in possible_versions:
                path = os.path.abspath(f"{os.environ['GYRE_DIR']}/../{gyredir}")
                version_file = f"{path}/src/common/gyre_version.fpp"
                if not os.path.exists(version_file):  # Different file name for 7.2 and after
                    version_file = f"{path}/src/common/version_m.fypp"

                if not os.path.exists(version_file):
                    continue

                with open(version_file, "r") as handle:
                    lines = handle.readlines()

                version_str = ""
                for line in lines:
                    if "VERSION = " in line.upper():
                        line = line.upper()
                        version_str = (
                            line.split("VERSION =")[1].strip().replace("'", "").replace("(", "").replace(")", "")
                        )
                        break

                major_version = version_str.split(".")[0]
                os.environ["GYRE_DIR"] = path

                args = make_args(["0", "MESA", "LOGS/profile10.data.GYRE", "--gyre", f"G{major_version}"])
                ierr = gyre_driver.run(args)

                if ierr == 0:
                    tested_versions.append(version_str)

                gs_path = pathlib.Path(test_data) / "gyre_out" / "profile10.data.GYRE.sgyre_l"
                if gs_path.exists():
                    os.remove(gs_path)

            np.testing.assert_array_equal(
                np.unique(np.array([int(v.split(".")[0]) for v in tested_versions])),
                np.array([6, 7, 8, 9]),
            )
        finally:
            os.environ["GYRE_DIR"] = original_gyre_dir
