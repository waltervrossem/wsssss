#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import unittest
import shutil
import tempfile
import numpy as np

from wsssss import load_data as ld
from .common import have_mesa_data

have_mesa_data()
test_data = os.path.join(os.path.dirname(__file__), "data", "mesa")
curdir = os.path.dirname(__file__)


def _copy_mesa_case_to_tmp(tmpdir):
    """
    Copy the 0000 test case into a temporary directory so tests can write
    .dill files and modify files without polluting the repository.
    """
    src = os.path.join(test_data, "0000")
    dst = os.path.join(tmpdir, "0000")
    shutil.copytree(src, dst, ignore=lambda _, names: [x for x in names if x.endswith(".dill")])
    return dst


class TestLoadData(unittest.TestCase):
    def setUp(self):
        self.hist_path = os.path.join(test_data, "0000", "LOGS", "history.data")
        self.profile1_path = os.path.join(test_data, "0000", "LOGS", "profile1.data")
        self.profile10_path = os.path.join(test_data, "0000", "LOGS", "profile10.data")
        self.gyre_profile1_path = os.path.join(test_data, "0000", "LOGS", "profile1.data.GYRE")
        self.gyre_summary10_path = os.path.join(test_data, "0000", "gyre_out", "profile10.data.GYRE.sgyre_l")

    def test_History(self):
        hist = ld.History(self.hist_path)
        hist.dump()

        hist_dill = ld.History(os.path.join(test_data, "0000", "LOGS", "history.data.dill"))
        np.testing.assert_array_equal(hist_dill.data, hist.data)
        self.assertDictEqual(hist.header, hist_dill.header)
        del hist_dill

        hist_reload = ld.History(self.hist_path, save_dill=True, reload=True)
        np.testing.assert_array_equal(hist_reload.data, hist.data)
        self.assertDictEqual(hist.header, hist_reload.header)
        del hist_reload

        np.testing.assert_array_equal(hist.get("model_number"), np.arange(1, 1001))
        np.testing.assert_array_equal(hist.data.model_number, np.arange(1, 1001))
        np.testing.assert_array_equal(hist.data.star_mass, np.ones(1000))
        np.testing.assert_array_equal(hist[:10].data.model_number, np.arange(1, 11))

        np.testing.assert_array_equal(hist.get_profile_index(hist.index[:, 2]), hist.index[:, 0] - 1)
        np.testing.assert_array_equal(hist.get_profile_num(150), (2, 100, 99))
        np.testing.assert_array_equal(hist.get_profile_num(150, method="previous"), (2, 100, 99))
        np.testing.assert_array_equal(hist.get_profile_num(150, method="next"), (3, 200, 199))
        np.testing.assert_array_equal(hist.get_profile_num(150, earlier=False), (3, 200, 199))

        hist_cols = ld.History(self.hist_path, keep_columns=["model_number", "center_he4"])
        self.assertListEqual(["model_number", "center_he4"], hist_cols.columns)
        self.assertListEqual(hist_cols.columns, list(hist_cols.data.dtype.names))
        np.testing.assert_array_equal(hist_cols.data[hist_cols.columns], hist.data[hist_cols.columns])
        del hist_cols

        self.assertRaises(
            ValueError, ld.History, self.hist_path, keep_columns=["model_number", "center_he4", "does_not_exist"]
        )

    def test_Profile(self):
        prof = ld.Profile(self.profile1_path)
        hist = ld.History(self.hist_path)

        np.testing.assert_array_equal(hist.get_profile_index(prof), np.zeros(1))
        np.testing.assert_array_equal(hist.get_profile_index([prof]), np.zeros(1))
        np.testing.assert_array_equal(hist.get_profile_index(prof.profile_num), np.zeros(1))
        np.testing.assert_array_equal(hist.get_profile_index([prof.profile_num]), np.zeros(1))

        prof = ld.Profile(self.profile1_path, load_GyreProfile=True)

        self.assertEqual(prof.get_hist_index(hist), 0)

    def test_GyreSummary(self):
        gsum = ld.GyreSummary(self.gyre_summary10_path)
        self.assertEqual(7, len(gsum.data[gsum.data["l"] == 0]))
        self.assertEqual(236, len(gsum.data[gsum.data["l"] == 1]))
        np.testing.assert_array_almost_equal_nulp(gsum.get_frequencies("Hz"), gsum.data["Re(freq)"] / 1e6)

    def test_GyreProfile(self):
        prof = ld.Profile(self.profile1_path)
        gprof = ld.GyreProfile(self.gyre_profile1_path)

        np.testing.assert_allclose(
            prof.data.mass / prof.data.mass[0],
            np.interp(
                prof.data.radius / prof.data.radius[0],
                gprof.data.radius / gprof.header["star_radius"],
                gprof.data.mass / gprof.header["star_mass"],
            ),
            rtol=1e-11,
        )

    def test_GyreMode(self):
        gsum = ld.GyreSummary(self.gyre_summary10_path)
        gmode = ld.GyreSummary(
            os.path.join(test_data, "0000", "gyre_out", "profile10.data.GYRE_l0_00005_np+9_ng+0.mgyre")
        )
        n_p = 9
        ng = 0
        mask = (gsum.data.n_p == n_p) & (gsum.data.n_g == ng)
        self.assertEqual(1, sum(mask))
        self.assertEqual(gmode.header["Re(freq)"], gsum.data["Re(freq)"][mask])
        np.testing.assert_array_almost_equal_nulp(gmode.get_frequencies("Hz"), gmode.header["Re(freq)"] / 1e6)

    def test_load_profs(self):
        hist = ld.History(self.hist_path)
        profs = ld.load_profs(hist)
        self.assertEqual(11, len(profs))
        self.assertListEqual(list(np.arange(1, 12)), [prof.profile_num for prof in profs])

    def test_load_gss(self):
        hist = ld.History(self.hist_path)
        gss = ld.load_gss(hist)
        self.assertEqual(11, len(gss))

    def load_modes_from_profile(self):
        gsum = ld.GyreSummary(self.gyre_summary10_path)
        prof = ld.Profile(self.profile10_path)
        modes = ld.load_modes_from_profile(prof)
        self.assertEqual(7, len(gsum.data[gsum.data.l == 0]))

    def load_gs_from_profile(self):
        prof = ld.Profile(self.profile10_path)
        gs = ld.load_gs_from_profile(prof)
        self.assertEqual(243, len(gs.data))

    def test_history_repr(self):
        hist = ld.History(self.hist_path, save_dill=False)
        text = repr(hist)

        self.assertIn("MESA history data file at", text)
        self.assertIn("Initial model=", text)
        self.assertIn("mass=", text)
        self.assertIn("age=", text)

        self.assertIn(str(hist._first_row["model_number"]), text)

    def test_history_get_single_and_multiple_columns(self):
        hist = ld.History(self.hist_path, save_dill=False)

        model_number = hist.get("model_number")
        self.assertIsInstance(model_number, np.ndarray)
        np.testing.assert_array_equal(model_number, np.arange(1, 1001))

        cols = hist.get("model_number", "star_mass")
        self.assertIsInstance(cols, list)
        self.assertEqual(len(cols), 2)
        np.testing.assert_array_equal(cols[0], np.arange(1, 1001))
        np.testing.assert_array_equal(cols[1], np.ones(1000))

    def test_history_get_with_mask(self):
        hist = ld.History(self.hist_path, save_dill=False)

        mask = hist.get("model_number") <= 10

        np.testing.assert_array_equal(hist.get("model_number", mask=mask), np.arange(1, 11))

        cols = hist.get("model_number", "star_mass", mask=mask)
        np.testing.assert_array_equal(cols[0], np.arange(1, 11))
        np.testing.assert_array_equal(cols[1], np.ones(10))

    def test_history_getitem_slicing_updates_index(self):
        hist = ld.History(self.hist_path, save_dill=False)

        sliced = hist[:10]

        self.assertEqual(len(sliced.data), 10)
        np.testing.assert_array_equal(sliced.data.model_number, np.arange(1, 11))

        self.assertIsNotNone(sliced.index)
        self.assertLessEqual(len(sliced.index), len(hist.index))

        mnum0, mnum1 = sliced.data.model_number[[0, -1]]
        self.assertTrue(np.all(sliced.index[:, 0] >= mnum0))
        self.assertTrue(np.all(sliced.index[:, 0] <= mnum1))

    def test_history_getitem_boolean_mask(self):
        hist = ld.History(self.hist_path, save_dill=False)

        mask = hist.get("model_number") <= 10
        sliced = hist[mask]

        self.assertEqual(len(sliced.data), 10)
        np.testing.assert_array_equal(sliced.data.model_number, np.arange(1, 11))

    def test_history_empty_on_error(self):
        bad_path = os.path.join(test_data, "does_not_exist", "history.data")

        hist = ld.History(bad_path, empty_on_error=True, save_dill=False)

        self.assertEqual(hist.header, {})
        self.assertEqual(hist.columns, [])
        self.assertEqual(len(hist.data), 0)
        self.assertFalse(hist.loaded)

    def test_history_missing_file_raises(self):
        bad_path = os.path.join(test_data, "does_not_exist", "history.data")

        with self.assertRaises(FileNotFoundError):
            ld.History(bad_path, empty_on_error=False, save_dill=False)

    def test_history_no_index(self):
        hist = ld.History(self.hist_path, index_name=None, save_dill=False)

        self.assertIsNone(hist.index)
        self.assertEqual(hist.index_path, "")

        with self.assertRaises(ValueError):
            hist.get_profile_num(150)

        with self.assertRaises(ValueError):
            hist.get_profile_index([1])

    def test_history_get_profile_num_invalid_method(self):
        hist = ld.History(self.hist_path, save_dill=False)

        with self.assertRaises(ValueError):
            hist.get_profile_num(150, method="not_a_method")

    def test_history_get_profile_num_previous_no_candidates(self):
        hist = ld.History(self.hist_path, save_dill=False)

        # No profile can be earlier than model 0.
        with self.assertRaises(ValueError):
            hist.get_profile_num(0, method="previous")

    def test_history_get_profile_num_next_no_candidates(self):
        hist = ld.History(self.hist_path, save_dill=False)

        # No profile can be later than a very large model number.
        with self.assertRaises(ValueError):
            hist.get_profile_num(10**9, method="next")

    def test_history_get_profile_num_previous_and_next(self):
        hist = ld.History(self.hist_path, save_dill=False)

        pnum_prev, pmod_prev, idx_prev = hist.get_profile_num(150, method="previous")
        pnum_next, pmod_next, idx_next = hist.get_profile_num(150, method="next")

        self.assertLessEqual(pmod_prev, 150)
        self.assertGreaterEqual(pmod_next, 150)

    def test_history_get_profile_index_int(self):
        hist = ld.History(self.hist_path, save_dill=False)

        idx = hist.get_profile_index(1)
        self.assertEqual(len(idx), 1)
        self.assertEqual(idx[0], 0)

    def test_history_get_profile_index_list_of_ints(self):
        hist = ld.History(self.hist_path, save_dill=False)

        idx = hist.get_profile_index([1, 2, 3])
        self.assertEqual(len(idx), 3)
        np.testing.assert_array_equal(idx, np.array([0, 99, 199]))

    def test_history_get_profile_index_profile_object(self):
        hist = ld.History(self.hist_path, save_dill=False)
        prof = ld.Profile(self.profile1_path, save_dill=False)

        idx = hist.get_profile_index(prof)
        self.assertEqual(len(idx), 1)
        self.assertEqual(idx[0], 0)

    def test_history_get_profile_index_list_of_profiles(self):
        hist = ld.History(self.hist_path, save_dill=False)
        profs = ld.load_profs(hist)

        idx = hist.get_profile_index(profs[:3])
        self.assertEqual(len(idx), 3)
        np.testing.assert_array_equal(idx, np.array([0, 99, 199]))

    def test_history_get_profile_index_empty_list(self):
        hist = ld.History(self.hist_path, save_dill=False)

        self.assertRaises(IndexError, hist.get_profile_index, [])

    def test_history_save_dill_creates_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            case_dir = _copy_mesa_case_to_tmp(tmpdir)
            hist_path = os.path.join(case_dir, "LOGS", "history.data")
            dill_path = hist_path + ".dill"

            self.assertFalse(os.path.exists(dill_path))

            hist = ld.History(hist_path, save_dill=True)
            _ = hist.data

            self.assertTrue(os.path.exists(dill_path))

            hist_loaded = ld.History(hist_path, save_dill=False)
            np.testing.assert_array_equal(hist_loaded.data, hist.data)
            self.assertDictEqual(hist.header, hist_loaded.header)

    def test_history_save_dill_false_does_not_create_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            case_dir = _copy_mesa_case_to_tmp(tmpdir)
            hist_path = os.path.join(case_dir, "LOGS", "history.data")
            dill_path = hist_path + ".dill"

            hist = ld.History(hist_path, save_dill=False)
            _ = hist.data

            self.assertFalse(os.path.exists(dill_path))

    def test_history_dill_only_after_deleting_original(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            case_dir = _copy_mesa_case_to_tmp(tmpdir)
            hist_path = os.path.join(case_dir, "LOGS", "history.data")
            dill_path = hist_path + ".dill"

            hist = ld.History(hist_path, save_dill=True)
            _ = hist.data

            self.assertTrue(os.path.exists(dill_path))

            os.remove(hist_path)

            hist_dill_only = ld.History(hist_path, save_dill=False)

            self.assertTrue(hist_dill_only.dill_only)
            np.testing.assert_array_equal(hist_dill_only.data, hist.data)
            self.assertDictEqual(hist.header, hist_dill_only.header)

    def test_history_dill_older_than_source_is_reloaded(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            case_dir = _copy_mesa_case_to_tmp(tmpdir)
            hist_path = os.path.join(case_dir, "LOGS", "history.data")
            dill_path = hist_path + ".dill"

            hist = ld.History(hist_path, save_dill=True)
            _ = hist.data

            self.assertTrue(os.path.exists(dill_path))

            # Make the original file newer than the dill file.
            os.utime(hist_path, None)
            os.utime(dill_path, (0, 0))

            hist2 = ld.History(hist_path, save_dill=False)
            _ = hist2.data

            np.testing.assert_array_equal(hist2.data, hist.data)
            self.assertDictEqual(hist.header, hist2.header)

            # Because save_dill was implicitly set to True when the dill was stale,
            # the dill file should have been rewritten and now be newer.
            self.assertTrue(os.path.exists(dill_path))
            self.assertGreaterEqual(os.path.getmtime(dill_path), os.path.getmtime(hist_path))

    def test_history_keep_columns_all(self):
        hist = ld.History(self.hist_path, keep_columns="all", save_dill=False)

        self.assertListEqual(hist.columns, list(hist.data.dtype.names))

    def test_history_keep_columns_preserves_requested_columns(self):
        keep = ["model_number", "center_he4"]

        hist = ld.History(self.hist_path, keep_columns=keep, save_dill=False)

        self.assertListEqual(hist.columns, keep)
        self.assertListEqual(list(hist.data.dtype.names), keep)

        np.testing.assert_array_equal(hist.data[keep], ld.History(self.hist_path, save_dill=False).data[keep])

    def test_history_keep_columns_missing_raises(self):
        with self.assertRaises(ValueError):
            ld.History(self.hist_path, keep_columns=["model_number", "center_he4", "does_not_exist"], save_dill=False)

    def test_profile_repr(self):
        prof = ld.Profile(self.profile1_path, save_dill=False)
        text = repr(prof)

        self.assertIn("MESA profile data file at", text)
        self.assertIn(str(prof.header["model_number"]), text)

    def test_profile_get_hist_index(self):
        hist = ld.History(self.hist_path, save_dill=False)
        prof = ld.Profile(self.profile1_path, save_dill=False)

        self.assertEqual(prof.get_hist_index(hist), 0)

    def test_profile_load_gyre_profile(self):
        prof = ld.Profile(self.profile1_path, load_GyreProfile=True, save_dill=False)

        self.assertIsInstance(prof.GyreProfile, ld.GyreProfile)
        self.assertEqual(prof.GyreProfile.path, self.gyre_profile1_path)

    def test_profile_missing_file_raises(self):
        bad_path = os.path.join(test_data, "does_not_exist", "profile1.data")

        with self.assertRaises(FileNotFoundError):
            ld.Profile(bad_path, save_dill=False)

    def test_profile_keep_columns(self):
        keep = ["mass", "radius"]

        prof = ld.Profile(self.profile1_path, keep_columns=keep, save_dill=False)

        self.assertListEqual(prof.columns, keep)
        self.assertListEqual(list(prof.data.dtype.names), keep)

    def test_profile_keep_columns_missing_raises(self):
        with self.assertRaises(ValueError):
            ld.Profile(self.profile1_path, keep_columns=["mass", "does_not_exist"], save_dill=False)

    def test_gyre_profile_header_and_data(self):
        gprof = ld.GyreProfile(self.gyre_profile1_path)

        self.assertIn("num_zones", gprof.header)
        self.assertIn("star_mass", gprof.header)
        self.assertIn("star_radius", gprof.header)
        self.assertIn("star_luminosity", gprof.header)
        self.assertIn("version", gprof.header)

        self.assertIn(gprof.version, (100, 101, 120))

        self.assertEqual(len(gprof.data), gprof.header["num_zones"])
        self.assertListEqual(list(gprof.data.dtype.names), gprof.columns)

    def test_gyre_profile_get(self):
        gprof = ld.GyreProfile(self.gyre_profile1_path)

        radius = gprof.get("radius")
        self.assertIsInstance(radius, np.ndarray)
        np.testing.assert_array_equal(radius, gprof.data.radius)

        cols = gprof.get("radius", "mass")
        self.assertIsInstance(cols, list)
        self.assertEqual(len(cols), 2)
        np.testing.assert_array_equal(cols[0], gprof.data.radius)
        np.testing.assert_array_equal(cols[1], gprof.data.mass)

    def test_gyre_profile_missing_file_raises(self):
        bad_path = os.path.join(test_data, "does_not_exist", "profile1.data.GYRE")

        with self.assertRaises(FileNotFoundError):
            ld.GyreProfile(bad_path)

    def test_gyre_profile_repr(self):
        gprof = ld.GyreProfile(self.gyre_profile1_path)
        text = repr(gprof)

        self.assertIn("GyreProfile data file at", text)
        self.assertIn("num_zones", text)

    def test_gyre_summary_repr(self):
        gsum = ld.GyreSummary(self.gyre_summary10_path)
        text = repr(gsum)

        self.assertIn("GyreSummary at", text)

    def test_gyre_summary_get_frequencies_units(self):
        gsum = ld.GyreSummary(self.gyre_summary10_path)

        freq_hz = gsum.get_frequencies("Hz")
        freq_mhz = gsum.get_frequencies("mHz")
        freq_uhz = gsum.get_frequencies("uHz")

        self.assertEqual(len(freq_hz), len(gsum.data))
        self.assertEqual(len(freq_mhz), len(gsum.data))
        self.assertEqual(len(freq_uhz), len(gsum.data))

        np.testing.assert_allclose(freq_mhz, freq_hz * 1e3, rtol=1e-12)
        np.testing.assert_allclose(freq_uhz, freq_hz * 1e6, rtol=1e-12)

    def test_gyre_summary_get_frequencies_bad_units(self):
        gsum = ld.GyreSummary(self.gyre_summary10_path)

        with self.assertRaises(KeyError):
            gsum.get_frequencies("not-a-unit")

    def test_gyre_mode_repr(self):
        gmode = ld.GyreMode(os.path.join(test_data, "0000", "gyre_out", "profile10.data.GYRE_l0_00005_np+9_ng+0.mgyre"))
        text = repr(gmode)

        self.assertIn("GyreMode at", text)

    def test_gyre_mode_get_frequencies(self):
        gmode = ld.GyreMode(os.path.join(test_data, "0000", "gyre_out", "profile10.data.GYRE_l0_00005_np+9_ng+0.mgyre"))

        freq_hz = gmode.get_frequencies("Hz")

        np.testing.assert_allclose(freq_hz, gmode.header["Re(freq)"] / 1e6, rtol=1e-12)

    def test_load_profs_all(self):
        hist = ld.History(self.hist_path, save_dill=False)
        profs = ld.load_profs(hist)

        self.assertEqual(len(profs), 11)
        self.assertTrue(all(isinstance(p, ld.Profile) for p in profs))
        self.assertListEqual([p.profile_num for p in profs], list(np.arange(1, 12)))

    def test_load_profs_no_index(self):
        hist = ld.History(self.hist_path, index_name=None, save_dill=False)
        profs = ld.load_profs(hist)

        self.assertEqual(profs, [])

    def test_load_profs_with_mask(self):
        hist = ld.History(self.hist_path, save_dill=False)

        mask = np.zeros(len(hist), dtype=bool)
        mask[:100] = True

        profs = ld.load_profs(hist, mask=mask)

        self.assertEqual(len(profs), 2)
        self.assertListEqual([p.profile_num for p in profs], [1, 2])

    def test_load_profs_with_mask_function(self):
        hist = ld.History(self.hist_path, save_dill=False)

        def mask_func(h):
            return h.get("model_number") <= 100

        profs = ld.load_profs(hist, mask=mask_func)

        self.assertEqual(len(profs), 2)
        self.assertListEqual([p.profile_num for p in profs], [1, 2])

    def test_load_profs_gyre_profile_suffix(self):
        hist = ld.History(self.hist_path, save_dill=False)

        # Mask so that only profile1 is requested. profile1.data.GYRE exists.
        mask = np.zeros(len(hist), dtype=bool)
        mask[0] = True

        profs = ld.load_profs(hist, suffix=".data.GYRE", mask=mask)

        self.assertEqual(len(profs), 1)
        self.assertIsInstance(profs[0], ld.GyreProfile)

    def test_load_gss_all(self):
        hist = ld.History(self.hist_path, save_dill=False)
        gss = ld.load_gss(hist)

        self.assertEqual(len(gss), 11)
        self.assertTrue(all(isinstance(g, ld.GyreSummary) for g in gss))

    def test_load_gss_return_pnums(self):
        hist = ld.History(self.hist_path, save_dill=False)
        gss_pnums = ld.load_gss(hist, return_pnums=True)

        self.assertEqual(len(gss_pnums), 11)

        gss, pnums = zip(*gss_pnums)

        self.assertTrue(all(isinstance(g, ld.GyreSummary) for g in gss))
        np.testing.assert_array_equal(np.asarray(pnums), np.arange(1, 12))

    def test_load_gss_missing_directory(self):
        hist = ld.History(self.hist_path, save_dill=False)

        with self.assertRaises(FileNotFoundError):
            ld.load_gss(hist, gyre_data_dir="does_not_exist")

    def test_load_gss_with_boolean_mask_all(self):
        hist = ld.History(self.hist_path, save_dill=False)

        mask = np.ones(len(hist), dtype=bool)
        gss = ld.load_gss(hist, use_mask=mask)

        self.assertEqual(len(gss), 11)

    def test_load_gss_with_boolean_mask_first_profile_only(self):
        hist = ld.History(self.hist_path, save_dill=False)

        mask = np.zeros(len(hist), dtype=bool)
        mask[0] = True

        gss = ld.load_gss(hist, use_mask=mask)

        self.assertEqual(len(gss), 1)

    def test_load_gss_with_mask_function(self):
        hist = ld.History(self.hist_path, save_dill=False)

        def mask_func(h):
            return h.get("model_number") <= 10

        gss = ld.load_gss(hist, use_mask=mask_func)

        self.assertEqual(len(gss), 1)

    def test_load_gss_to_hist(self):
        hist = ld.History(self.hist_path, save_dill=False)

        hist = ld.load_gss_to_hist(hist)

        self.assertTrue(hasattr(hist, "gsspnum"))
        self.assertEqual(len(hist.gsspnum), 11)

        gss, pnums = zip(*hist.gsspnum)
        self.assertTrue(all(isinstance(g, ld.GyreSummary) for g in gss))
        np.testing.assert_array_equal(np.asarray(pnums), np.arange(1, 12))

    def test_load_gs_from_profile(self):
        prof = ld.Profile(self.profile10_path, save_dill=False)
        gs = ld.load_gs_from_profile(prof)

        self.assertIsInstance(gs, ld.GyreSummary)
        self.assertEqual(len(gs.data), 243)

    def test_load_gs_from_profile_missing_file(self):
        prof = ld.Profile(self.profile10_path, save_dill=False)

        # Force a prefix that does not match any GyreSummary file.
        prof.fname = "nonexistent_profile.data"

        with self.assertRaises(FileNotFoundError):
            ld.load_gs_from_profile(prof)

    def test_load_modes_from_profile(self):
        prof = ld.Profile(self.profile10_path, save_dill=False)
        gs = ld.load_gs_from_profile(prof)
        modes = ld.load_modes_from_profile(prof)

        self.assertTrue(all(isinstance(m, ld.GyreMode) for m in modes))
        self.assertEqual(len(modes), len(gs.data[gs.data.l == 0]))

    def test_load_modes_from_profile_missing_directory(self):
        prof = ld.Profile(self.profile10_path, save_dill=False)

        with self.assertRaises(FileNotFoundError):
            ld.load_modes_from_profile(prof, gyre_data_dir="does_not_exist")

    def test_naive_merge_hists_empty_raises(self):
        hist = ld.History(self.hist_path, save_dill=False)

        with self.assertRaises(ValueError):
            ld.naive_merge_hists(hist, [])

    def test_naive_merge_hists_slices(self):
        hist = ld.History(self.hist_path, save_dill=False)

        h1 = hist[:10]
        h2 = hist[10:20]

        merged = ld.naive_merge_hists(hist, [h1, h2])

        self.assertEqual(len(merged.data), 20)
        np.testing.assert_array_equal(merged.data.model_number, np.arange(1, 21))

    def test_naive_merge_hists_same_hist(self):
        hist = ld.History(self.hist_path, save_dill=False)

        merged = ld.naive_merge_hists(hist, [hist])

        self.assertEqual(len(merged.data), len(hist.data))
        np.testing.assert_array_equal(merged.data.model_number, hist.data.model_number)

    def test_history_dill_round_trip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            src = os.path.join(test_data, "0000")
            dst = os.path.join(tmpdir, "0000")
            shutil.copytree(src, dst)

            hist_path = os.path.join(dst, "LOGS", "history.data")

            hist = ld.History(hist_path, save_dill=True)
            dill_path = hist.dill_path

            self.assertTrue(os.path.exists(dill_path))

            hist_from_dill = ld.History(dill_path, save_dill=False)

            np.testing.assert_array_equal(hist_from_dill.data, hist.data)
            self.assertDictEqual(hist_from_dill.header, hist.header)
            self.assertListEqual(hist_from_dill.columns, hist.columns)

    def test_profile_dill_round_trip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            src = os.path.join(test_data, "0000")
            dst = os.path.join(tmpdir, "0000")
            shutil.copytree(src, dst)

            prof_path = os.path.join(dst, "LOGS", "profile1.data")

            prof = ld.Profile(prof_path, save_dill=True)
            _ = prof.data  # Force load lazy data
            dill_path = prof.dill_path

            self.assertTrue(os.path.exists(dill_path))

            prof_from_dill = ld.Profile(dill_path, save_dill=False)

            np.testing.assert_array_equal(prof_from_dill.data, prof.data)
            self.assertDictEqual(prof_from_dill.header, prof.header)
            self.assertListEqual(prof_from_dill.columns, prof.columns)

    def test_gyre_summary_dill_round_trip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            src = os.path.join(test_data, "0000")
            dst = os.path.join(tmpdir, "0000")
            shutil.copytree(src, dst)

            gsum = ld.GyreSummary(self.gyre_summary10_path, save_dill=True)
            dill_path = gsum.dill_path
            _ = gsum.data  # Force load lazy data

            self.assertTrue(os.path.exists(dill_path))

            gsum_from_dill = ld.GyreSummary(dill_path, save_dill=False)

            np.testing.assert_array_equal(gsum_from_dill.data, gsum.data)
            self.assertDictEqual(gsum_from_dill.header, gsum.header)
            self.assertListEqual(gsum_from_dill.columns, gsum.columns)

    def test_gyre_mode_dill_round_trip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            src = os.path.join(test_data, "0000")
            dst = os.path.join(tmpdir, "0000")
            shutil.copytree(src, dst)

            gmode_path = os.path.join(dst, "gyre_out", "profile10.data.GYRE_l0_00005_np+9_ng+0.mgyre")

            gmode = ld.GyreMode(gmode_path, save_dill=True)
            dill_path = gmode.dill_path
            _ = gmode.data  # Force load lazy data

            self.assertTrue(os.path.exists(dill_path))

            gmode_from_dill = ld.GyreMode(dill_path, save_dill=False)

            np.testing.assert_array_equal(gmode_from_dill.data, gmode.data)
            self.assertDictEqual(gmode_from_dill.header, gmode.header)
            self.assertListEqual(gmode_from_dill.columns, gmode.columns)
