#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import multiprocessing as mp
import os
import shutil
import sys
import tempfile
import unittest
import subprocess
from unittest import mock

from wsssss._bin import mesa_go
from wsssss.inlists import create_grid as cg

must_have_environ = ["MESA_DIR", "MESASDK_ROOT"]
missing_environ = []
for env in must_have_environ:
    if env not in os.environ:
        missing_environ.append(env)

MESASDK_initialized = False
if "MESASDK_VERSION" in os.environ:
    MESASDK_initialized = True

mesa_dir = os.environ.get("MESA_DIR", "")


@unittest.skipIf(os.name == "nt", "Skipping on Windows")
@unittest.skipIf(mesa_dir == "", "Environment variable MESA_DIR not set.")
class TestMesaGO(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        if len(missing_environ) > 0:
            raise EnvironmentError(f'{",".join(missing_environ)} not set.')
        if not MESASDK_initialized:
            raise EnvironmentError("The MESASDK has not been initialized.")

        self.init_dir = os.path.abspath(".")
        self.base_grid_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/mesago"))
        grid = cg.MesaGrid()

        grid.star_job["history_columns_file"] = "history_test.list"
        grid.star_job["pgstar_flag"] = False
        grid.star_job["create_pre_main_sequence_model"] = False

        grid.kap["use_Type2_opacities"] = True
        grid.kap["Zbase"] = 0.02

        grid.controls["initial_mass"] = [1, 2]
        grid.controls["initial_z"] = 0.02
        grid.controls["max_model_number"] = 10
        grid.controls["mesh_delta_coeff"] = 5

        grid.controls["history_interval"] = 1
        grid.controls["photo_interval"] = 8

        grid.controls["write_profiles_flag"] = False

        grid.add_file(os.path.join(os.path.dirname(__file__), "..", "make_data", grid.star_job["history_columns_file"]))

        for fname in ["clean", "rn", "mk", "re"]:
            grid.add_file(os.path.join(mesa_dir, "star/work", fname))
        for dirname in ["make", "src"]:
            grid.add_dir(os.path.join(mesa_dir, "star/work", dirname))

        self.grid = grid

        # Compile star so we can skip the compilation step for each test
        cwd = os.path.abspath(".")
        self.grid.create_grid(f"{self.base_grid_dir}/tmp")
        os.chdir(f"{self.base_grid_dir}/tmp/0000")
        subprocess.run("./mk")
        shutil.copy2("star", f"{self.base_grid_dir}/star")
        shutil.rmtree(f"{self.base_grid_dir}/tmp")
        os.chdir(cwd)

    def setUp(self):
        testname = self.id().split(".")[-1].replace("test", "")
        self.grid_dir = os.path.join(self.base_grid_dir, testname)
        self.grid.create_grid(self.grid_dir)

        for dirname in self.grid.dirnames:
            shutil.copy2(f"{self.base_grid_dir}/star", f"{self.base_grid_dir}/{testname}/{dirname}/")

        os.chdir(self.grid_dir)
        sys.argv = ["mesa-go", ""]

    @classmethod
    def tearDownClass(self):
        if os.path.isdir(self.base_grid_dir):
            shutil.rmtree(f"{self.base_grid_dir}")
        os.chdir(self.init_dir)

    def check_output(self):
        os.chdir(self.grid_dir)
        output = subprocess.run(["check-grid", "--no-slurm", "--out-file", "../out_{}"], stdout=subprocess.PIPE)

        expected = (
            "--------------------------------------------\n"
            "  termination_code                   count\n"
            "--------------------------------------------\n"
            "max_model_number                           2\n"
            "--------------------------------------------\n"
            "\n"
        )
        out_str = output.stdout.decode()
        self.assertEqual(expected, out_str)

    def test_mesago(self):
        sys.argv = ["mesa-go", "--verbose", "--cmd-pre", "touch pre", "--cmd-post", "touch post"]
        os.remove(f"{self.grid_dir}/0000/star")  # Check that auto-compile works
        ierr = mesa_go.run()
        if ierr != 0:
            raise SystemError(ierr)
        self.check_output()
        self.assertTrue(os.path.isfile(f"{self.grid_dir}/pre"))
        self.assertTrue(os.path.isfile(f"{self.grid_dir}/post"))
        self.assertTrue(os.path.isfile(f"{self.grid_dir}/0000/star"))
        os.remove(f"{self.grid_dir}/pre")
        os.remove(f"{self.grid_dir}/post")
        for dirname in self.grid.dirnames:
            with open(f"{self.grid_dir}/out_{dirname}", "r") as handle:
                lines = handle.readlines()
            self.assertEqual(131, len(lines))

    def test_mesago_each(self):
        sys.argv = ["mesa-go", "--verbose", "--cmd-pre-each", "touch preeach", "--cmd-post-each", "touch posteach"]
        ierr = mesa_go.run()
        if ierr != 0:
            raise SystemError(ierr)
        self.check_output()
        for dirname in self.grid.dirnames:
            self.assertTrue(os.path.isfile(f"{self.grid_dir}/{dirname}/preeach"))
            self.assertTrue(os.path.isfile(f"{self.grid_dir}/{dirname}/posteach"))
            os.remove(f"{self.grid_dir}/{dirname}/preeach")
            os.remove(f"{self.grid_dir}/{dirname}/posteach")

        for dirname in self.grid.dirnames:
            with open(f"{self.grid_dir}/out_{dirname}", "r") as handle:
                lines = handle.readlines()
            self.assertEqual(131, len(lines))

    def test_mesago_restart(self):
        sys.argv = ["mesa-go", "--verbose", "--restart"]
        for dirname in self.grid.dirnames:
            os.makedirs(f"{self.grid_dir}/{dirname}/photos/")
            shutil.copy2(
                os.path.join(self.base_grid_dir, "_mesago", dirname, "photos/x008"),
                f"{self.grid_dir}/{dirname}/photos/",
            )
            shutil.copy2(os.path.join(self.base_grid_dir, "_mesago", dirname, "star"), f"{self.grid_dir}/{dirname}/")
        ierr = mesa_go.run()
        if ierr != 0:
            raise SystemError(ierr)
        self.check_output()
        for dirname in self.grid.dirnames:
            with open(f"{self.grid_dir}/out_{dirname}", "r") as handle:
                lines = handle.readlines()
            self.assertEqual(60, len(lines))

    def test_mesago_restartfile(self):
        sys.argv = ["mesa-go", "--verbose", "--restart", "grid_restart"]
        with open(f"{self.grid_dir}/grid_restart", "w") as handle:
            handle.write("0000 x008\n0001 full_restart\n")

        for dirname in self.grid.dirnames:
            os.makedirs(f"{self.grid_dir}/{dirname}/photos/")
            shutil.copy2(
                os.path.join(self.base_grid_dir, "_mesago", dirname, "photos/x008"),
                f"{self.grid_dir}/{dirname}/photos/",
            )
            shutil.copy2(os.path.join(self.base_grid_dir, "_mesago", dirname, "star"), f"{self.grid_dir}/{dirname}/")

        ierr = mesa_go.run()
        if ierr != 0:
            raise SystemError(ierr)
        self.check_output()
        for dirname in self.grid.dirnames:
            with open(f"{self.grid_dir}/out_{dirname}", "r") as handle:
                lines = handle.readlines()
            self.assertEqual({"0000": 60, "0001": 131}[dirname], len(lines))


class TestMesaGoUnit(unittest.TestCase):
    """
    Lightweight tests for mesa_go helper functions.

    These tests do not require a compiled MESA star executable.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.grid_dir = os.path.join(self.tmp.name, "grid")
        os.makedirs(self.grid_dir)

        self.logger = logging.getLogger("mesa_go_unit")
        self.logger.setLevel(logging.CRITICAL)

        self._old_cwd = os.getcwd()

    def tearDown(self):
        os.chdir(self._old_cwd)
        self.tmp.cleanup()

    def make_args(self, argv):
        parser = mesa_go.get_parser()
        return parser.parse_args(argv)

    def test_expand_path_empty(self):
        self.assertEqual(mesa_go.expand_path(""), "")

    def test_expand_path_relative(self):
        expected = os.path.abspath(os.path.join(os.getcwd(), "subdir"))
        self.assertEqual(mesa_go.expand_path("subdir"), expected)

    def test_expand_path_user(self):
        home = os.path.join(self.tmp.name, "home")
        with mock.patch.dict(os.environ, {"HOME": home, "USERPROFILE": home}):
            self.assertEqual(mesa_go.expand_path("~"), os.path.abspath(home))

    def test_expand_path_environment_variable(self):
        with mock.patch.dict(os.environ, {"MESAGO_TEST_PATH": "/tmp/mesa_go_test"}):
            path = mesa_go.expand_path("$MESAGO_TEST_PATH/sub")
            self.assertEqual(path, os.path.abspath("/tmp/mesa_go_test/sub"))

    def test_resolve_path_template_empty(self):
        path = mesa_go.resolve_path_template(
            "",
            grid_dir="/grid",
            run_name="0000",
        )
        self.assertEqual(path, "")

    def test_resolve_path_template_run_name(self):
        path = mesa_go.resolve_path_template(
            "out_RUN_NAME.log",
            grid_dir="/grid",
            run_name="0000",
        )
        self.assertEqual(path, os.path.abspath("/grid/out_0000.log"))

    def test_resolve_path_template_work_dir(self):
        path = mesa_go.resolve_path_template(
            "WORK_DIR/out.log",
            grid_dir="/grid",
            run_name="0000",
        )
        self.assertEqual(path, os.path.abspath("/grid/0000/out.log"))

    def test_resolve_path_template_absolute(self):
        path = mesa_go.resolve_path_template(
            "/tmp/mesa_go_logs/RUN_NAME.log",
            grid_dir="/grid",
            run_name="0000",
        )
        self.assertEqual(path, os.path.abspath("/tmp/mesa_go_logs/0000.log"))

    def test_resolve_path_template_plain_relative(self):
        path = mesa_go.resolve_path_template(
            "out.log",
            grid_dir="/grid",
            run_name="0000",
        )
        self.assertEqual(path, os.path.abspath("/grid/out.log"))

    def test_check_cores_ok(self):
        args = self.make_args([self.grid_dir, "--num-mesa", "2", "--OMP_NUM_THREADS", "2"])

        with mock.patch.object(mesa_go.os, "cpu_count", return_value=4):
            req_cores, nproc = mesa_go.check_cores(args)

        self.assertEqual(req_cores, 4)
        self.assertEqual(nproc, 4)

    def test_check_cores_too_many_cores(self):
        args = self.make_args([self.grid_dir, "--num-mesa", "2", "--OMP_NUM_THREADS", "3"])

        with mock.patch.object(mesa_go.os, "cpu_count", return_value=4):
            with self.assertRaises(ValueError):
                mesa_go.check_cores(args)

    def test_check_cores_num_mesa_zero(self):
        args = self.make_args([self.grid_dir, "--num-mesa", "0"])

        with mock.patch.object(mesa_go.os, "cpu_count", return_value=4):
            with self.assertRaises(ValueError):
                mesa_go.check_cores(args)

    def test_check_cores_omp_zero(self):
        args = self.make_args([self.grid_dir, "--OMP_NUM_THREADS", "0"])

        with mock.patch.object(mesa_go.os, "cpu_count", return_value=4):
            with self.assertRaises(ValueError):
                mesa_go.check_cores(args)

    def test_check_cores_cpu_count_none(self):
        args = self.make_args([self.grid_dir])

        with mock.patch.object(mesa_go.os, "cpu_count", return_value=None):
            with self.assertRaises(ValueError):
                mesa_go.check_cores(args)

    def test_process_args_defaults(self):
        args = self.make_args([self.grid_dir])
        processed = mesa_go.process_args(args)

        self.assertEqual(processed.grid_dir, os.path.abspath(self.grid_dir))
        self.assertTrue(os.path.isabs(processed.base_work_dir))
        self.assertFalse(processed.restart)
        self.assertIsNone(processed.restart_settings)

    def test_process_args_missing_grid_dir(self):
        missing = os.path.join(self.grid_dir, "does_not_exist")
        args = self.make_args([missing])

        with self.assertRaises(FileNotFoundError):
            mesa_go.process_args(args)

    def test_process_args_source_existing(self):
        source = os.path.join(self.grid_dir, "source.sh")
        with open(source, "w") as handle:
            handle.write("echo source\n")

        args = self.make_args([self.grid_dir, "--source", source])
        processed = mesa_go.process_args(args)

        self.assertEqual(processed.source, os.path.abspath(source))

    def test_process_args_source_missing(self):
        source = os.path.join(self.grid_dir, "missing_source.sh")
        args = self.make_args([self.grid_dir, "--source", source])

        with self.assertRaises(FileNotFoundError):
            mesa_go.process_args(args)

    def test_process_args_log_path_requires_template(self):
        args = self.make_args([self.grid_dir, "--log-path", "out.log"])

        with self.assertRaises(ValueError):
            mesa_go.process_args(args)

    def test_process_args_log_path_ok(self):
        args = self.make_args([self.grid_dir, "--log-path", "out_RUN_NAME.log"])
        processed = mesa_go.process_args(args)

        self.assertEqual(processed.log_path, "out_RUN_NAME.log")

    def test_process_args_skip_if_file_exists_requires_template(self):
        args = self.make_args([self.grid_dir, "--skip-if-file-exists", "skip.log"])

        with self.assertRaises(ValueError):
            mesa_go.process_args(args)

    def test_process_args_skip_if_file_exists_mesago(self):
        args = self.make_args([self.grid_dir, "--skip-if-file-exists", "MESAGO_LOG_FILE"])
        processed = mesa_go.process_args(args)

        self.assertEqual(processed.skip_if_file_exists, "MESAGO_LOG_FILE")

    def test_process_args_restart_bool(self):
        args = self.make_args([self.grid_dir, "--restart"])
        processed = mesa_go.process_args(args)

        self.assertTrue(processed.restart)
        self.assertIsNone(processed.restart_settings)

    def test_process_args_restart_file_missing(self):
        restart_file = os.path.join(self.grid_dir, "missing_restart")
        args = self.make_args([self.grid_dir, "--restart", restart_file])

        with self.assertRaises(FileNotFoundError):
            mesa_go.process_args(args)

    def test_process_args_restart_file(self):
        restart_file = os.path.join(self.grid_dir, "grid_restart")
        with open(restart_file, "w") as handle:
            handle.write("0000 x008\n# comment\n\n0001 full_restart #inline comment # with multiple #\n")

        args = self.make_args([self.grid_dir, "--restart", restart_file])
        processed = mesa_go.process_args(args)

        self.assertTrue(processed.restart)
        self.assertEqual(processed.restart_settings, {"0000": "x008", "0001": "full_restart"})

    def test_process_args_restart_file_bad_line(self):
        restart_file = os.path.join(self.grid_dir, "bad_restart")
        with open(restart_file, "w") as handle:
            handle.write("0000\n")

        args = self.make_args([self.grid_dir, "--restart", restart_file])

        with self.assertRaises(ValueError):
            mesa_go.process_args(args)

    def test_get_subdirs_auto_numeric(self):
        os.makedirs(os.path.join(self.grid_dir, "0001"))
        os.makedirs(os.path.join(self.grid_dir, "0000"))
        os.makedirs(os.path.join(self.grid_dir, "not_a_number"))

        args = self.make_args([self.grid_dir])
        sub_dirs = mesa_go.get_subdirs(args)

        self.assertEqual(sub_dirs, ["0000", "0001"])

    def test_get_subdirs_empty_list_auto_detects(self):
        os.makedirs(os.path.join(self.grid_dir, "0000"))
        os.makedirs(os.path.join(self.grid_dir, "0001"))

        args = self.make_args([self.grid_dir, "--sub-dirs"])
        sub_dirs = mesa_go.get_subdirs(args)

        self.assertEqual(sub_dirs, ["0000", "0001"])

    def test_get_subdirs_custom(self):
        os.makedirs(os.path.join(self.grid_dir, "alpha"))
        os.makedirs(os.path.join(self.grid_dir, "beta"))

        args = self.make_args([self.grid_dir, "--sub-dirs", "beta", "alpha"])
        sub_dirs = mesa_go.get_subdirs(args)

        self.assertEqual(sub_dirs, ["beta", "alpha"])

    def test_get_subdirs_custom_missing(self):
        args = self.make_args([self.grid_dir, "--sub-dirs", "missing"])

        with self.assertRaises(FileNotFoundError):
            mesa_go.get_subdirs(args)

    def test_get_subdirs_ignores_files_and_non_numeric(self):
        os.makedirs(os.path.join(self.grid_dir, "0000"))
        os.makedirs(os.path.join(self.grid_dir, "abc"))
        with open(os.path.join(self.grid_dir, "0001"), "w") as handle:
            handle.write("")

        args = self.make_args([self.grid_dir])
        sub_dirs = mesa_go.get_subdirs(args)

        self.assertEqual(sub_dirs, ["0000"])

    def test_run_cmd_file_mode_invalid(self):
        with self.assertRaises(ValueError):
            mesa_go.run_cmd("echo hello", self.logger, file_mode="x")

    def test_run_cmd_list_capture_output(self):
        out = mesa_go.run_cmd(
            [sys.executable, "-c", "print('hello')"],
            self.logger,
            capture_output=True,
        )
        self.assertEqual(out.stdout.decode().strip(), "hello")

    def test_run_cmd_to_file(self):
        out_file = os.path.join(self.grid_dir, "out.txt")
        mesa_go.run_cmd(
            [sys.executable, "-c", "print('hello')"],
            self.logger,
            to_file=out_file,
        )

        with open(out_file, "r") as handle:
            self.assertEqual(handle.read().strip(), "hello")

    def test_run_cmd_to_file_append(self):
        out_file = os.path.join(self.grid_dir, "out_append.txt")
        with open(out_file, "w") as handle:
            handle.write("initial\n")

        mesa_go.run_cmd(
            [sys.executable, "-c", "print('world')"],
            self.logger,
            to_file=out_file,
            file_mode="a",
        )

        with open(out_file, "r") as handle:
            self.assertEqual(handle.read(), "initial\nworld\n")

    def test_copy_base_work_dir_copies_missing_files(self):
        base_dir = os.path.join(self.tmp.name, "base_work")
        run_dir = os.path.join(self.grid_dir, "0000")

        os.makedirs(os.path.join(base_dir, "subdir"))
        os.makedirs(run_dir)

        with open(os.path.join(base_dir, "file.txt"), "w") as handle:
            handle.write("base")

        with open(os.path.join(base_dir, "subdir", "nested.txt"), "w") as handle:
            handle.write("nested")

        with open(os.path.join(run_dir, "file.txt"), "w") as handle:
            handle.write("existing")

        args = self.make_args([self.grid_dir])
        args.base_work_dir = base_dir

        mesa_go.copy_base_work_dir(args, run_dir, self.logger)

        self.assertEqual(os.path.isdir(os.path.join(run_dir, "subdir")), True)

        with open(os.path.join(run_dir, "file.txt"), "r") as handle:
            self.assertEqual(handle.read(), "existing")

        with open(os.path.join(run_dir, "subdir", "nested.txt"), "r") as handle:
            self.assertEqual(handle.read(), "nested")

    def test_copy_base_work_dir_missing_base_dir(self):
        base_dir = os.path.join(self.tmp.name, "missing_base")
        run_dir = os.path.join(self.grid_dir, "0000")
        os.makedirs(run_dir)

        args = self.make_args([self.grid_dir])
        args.base_work_dir = base_dir

        mesa_go.copy_base_work_dir(args, run_dir, self.logger)

        self.assertEqual(os.listdir(run_dir), [])

    def test_copy_base_work_dir_same_directory(self):
        base_dir = os.path.join(self.grid_dir, "0000")
        os.makedirs(base_dir)

        with open(os.path.join(base_dir, "file.txt"), "w") as handle:
            handle.write("base")

        args = self.make_args([self.grid_dir])
        args.base_work_dir = base_dir

        mesa_go.copy_base_work_dir(args, base_dir, self.logger)

        with open(os.path.join(base_dir, "file.txt"), "r") as handle:
            self.assertEqual(handle.read(), "base")

    def test_choose_restart_photo_no_restart(self):
        args = self.make_args([self.grid_dir])
        args.restart = False
        args.restart_settings = None

        photo, run_new = mesa_go.choose_restart_photo(
            args,
            ["x008"],
            "0000",
            self.logger,
            123,
        )

        self.assertIsNone(photo)
        self.assertTrue(run_new)

    def test_choose_restart_photo_no_photos(self):
        args = self.make_args([self.grid_dir])
        args.restart = True
        args.restart_settings = None

        photo, run_new = mesa_go.choose_restart_photo(
            args,
            [],
            "0000",
            self.logger,
            123,
        )

        self.assertIsNone(photo)
        self.assertTrue(run_new)

    def test_choose_restart_photo_restart_settings_present(self):
        args = self.make_args([self.grid_dir])
        args.restart = True
        args.restart_settings = {"0000": "x008"}

        photo, run_new = mesa_go.choose_restart_photo(
            args,
            ["x008"],
            "0000",
            self.logger,
            123,
        )

        self.assertEqual(photo, "x008")
        self.assertFalse(run_new)

    def test_choose_restart_photo_restart_settings_missing(self):
        args = self.make_args([self.grid_dir])
        args.restart = True
        args.restart_settings = {"0001": "x008"}

        photo, run_new = mesa_go.choose_restart_photo(
            args,
            ["x008"],
            "0000",
            self.logger,
            123,
        )

        self.assertIsNone(photo)
        self.assertTrue(run_new)

    def test_choose_restart_photo_auto_numeric(self):
        args = self.make_args([self.grid_dir])
        args.restart = True
        args.restart_settings = None

        photo, run_new = mesa_go.choose_restart_photo(
            args,
            ["x001", "x010", "x100", "not_numeric"],
            "0000",
            self.logger,
            123,
        )

        self.assertEqual(photo, "x100")
        self.assertFalse(run_new)

    def test_choose_restart_photo_non_numeric_only(self):
        args = self.make_args([self.grid_dir])
        args.restart = True
        args.restart_settings = None

        photo, run_new = mesa_go.choose_restart_photo(
            args,
            ["not_numeric"],
            "0000",
            self.logger,
            123,
        )

        self.assertIsNone(photo)
        self.assertTrue(run_new)

    def test_photo_exists_photos_dir(self):
        os.makedirs(os.path.join(self.grid_dir, "photos"))
        photo_path = os.path.join(self.grid_dir, "photos", "x008")
        with open(photo_path, "w") as handle:
            handle.write("")

        os.chdir(self.grid_dir)
        self.assertTrue(mesa_go.photo_exists("x008"))

    def test_photo_exists_bare_photo(self):
        photo_path = os.path.join(self.grid_dir, "x009")
        with open(photo_path, "w") as handle:
            handle.write("")

        os.chdir(self.grid_dir)
        self.assertTrue(mesa_go.photo_exists("x009"))

    def test_photo_exists_missing(self):
        os.chdir(self.grid_dir)
        self.assertFalse(mesa_go.photo_exists("x999"))

    def test_start_mesa_missing_run_dir(self):
        args = self.make_args([self.grid_dir])
        args.grid_dir = self.grid_dir

        with self.assertRaises(FileNotFoundError):
            mesa_go.start_mesa(args, "0000", self.logger)

    def test_start_mesa_skip_if_log_file_exists(self):
        run_dir = os.path.join(self.grid_dir, "0000")
        os.makedirs(run_dir)

        log_file = os.path.join(self.grid_dir, "out_0000")
        with open(log_file, "w") as handle:
            handle.write("existing")

        args = self.make_args([self.grid_dir, "--verbose"])
        args.grid_dir = self.grid_dir
        args.skip_if_file_exists = "MESAGO_LOG_FILE"

        with mock.patch.object(mesa_go, "run_cmd") as mock_run_cmd:
            run_name, message = mesa_go.start_mesa(args, "0000", self.logger)

        self.assertEqual(run_name, "0000")
        self.assertEqual(message, "Skip file found.")
        mock_run_cmd.assert_not_called()

    def test_start_mesa_new_run(self):
        run_dir = os.path.join(self.grid_dir, "0000")
        os.makedirs(run_dir)

        star_path = os.path.join(run_dir, "star")
        with open(star_path, "w") as handle:
            handle.write("")

        args = self.make_args([self.grid_dir, "--verbose"])
        args.grid_dir = self.grid_dir
        args.log_path = "out_RUN_NAME"
        args.skip_if_file_exists = ""
        args.source = ""
        args.cmd_pre_each = ""
        args.cmd_post_each = ""
        args.cmd_main = "./rn"
        args.restart = False
        args.restart_settings = None

        sentinel = mock.MagicMock()

        with mock.patch.object(mesa_go, "run_cmd", return_value=sentinel) as mock_run_cmd:
            with mock.patch.object(mesa_go, "copy_base_work_dir") as mock_copy:
                with mock.patch.object(mesa_go.os, "access", return_value=True):
                    run_name, out = mesa_go.start_mesa(args, "0000", self.logger)

        self.assertEqual(run_name, "0000")
        self.assertIs(out, sentinel)
        self.assertEqual(mock_run_cmd.call_count, 1)
        mock_copy.assert_called_once()

        call_args = mock_run_cmd.call_args
        self.assertIn("out_0000", call_args.kwargs["to_file"])
        self.assertEqual(call_args.kwargs["file_mode"], "w")

    def test_start_mesa_restart_photo(self):
        run_dir = os.path.join(self.grid_dir, "0000")
        os.makedirs(run_dir)

        star_path = os.path.join(run_dir, "star")
        with open(star_path, "w") as handle:
            handle.write("")

        photos_dir = os.path.join(run_dir, "photos")
        os.makedirs(photos_dir)
        with open(os.path.join(photos_dir, "x008"), "w") as handle:
            handle.write("")

        args = self.make_args([self.grid_dir, "--verbose", "--restart"])
        args = mesa_go.process_args(args)

        sentinel = mock.MagicMock()

        with mock.patch.object(mesa_go, "run_cmd", return_value=sentinel) as mock_run_cmd:
            with mock.patch.object(mesa_go, "copy_base_work_dir"):
                with mock.patch.object(mesa_go.os, "access", return_value=True):
                    run_name, out = mesa_go.start_mesa(args, "0000", self.logger)

        self.assertEqual(run_name, "0000")
        self.assertIs(out, sentinel)

        call_args = mock_run_cmd.call_args
        self.assertIn("./re x008", call_args.args[0])
        self.assertEqual(call_args.kwargs["file_mode"], "a")

    def test_start_mesa_full_restart(self):
        run_dir = os.path.join(self.grid_dir, "0000")
        os.makedirs(run_dir)

        star_path = os.path.join(run_dir, "star")
        with open(star_path, "w") as handle:
            handle.write("")

        photos_dir = os.path.join(run_dir, "photos")
        os.makedirs(photos_dir)
        with open(os.path.join(photos_dir, "x008"), "w") as handle:
            handle.write("")

        args = self.make_args([self.grid_dir, "--verbose", "--restart"])
        args = mesa_go.process_args(args)
        args.grid_dir = self.grid_dir
        args.log_path = "out_RUN_NAME"
        args.skip_if_file_exists = ""
        args.source = ""
        args.cmd_pre_each = ""
        args.cmd_post_each = ""
        args.cmd_main = "./rn"
        args.restart_settings = {"0000": "full_restart"}

        sentinel = mock.MagicMock()

        with mock.patch.object(mesa_go, "run_cmd", return_value=sentinel) as mock_run_cmd:
            with mock.patch.object(mesa_go, "copy_base_work_dir"):
                with mock.patch.object(mesa_go.os, "access", return_value=True):
                    run_name, out = mesa_go.start_mesa(args, "0000", self.logger)

        self.assertEqual(run_name, "0000")
        self.assertIs(out, sentinel)

        call_args = mock_run_cmd.call_args
        self.assertIn("./rn 2>&1", call_args.args[0])
        self.assertEqual(call_args.kwargs["file_mode"], "w")

    def test_start_mesa_restart_photo_missing_falls_back_to_new_run(self):
        run_dir = os.path.join(self.grid_dir, "0000")
        os.makedirs(run_dir)

        star_path = os.path.join(run_dir, "star")
        with open(star_path, "w") as handle:
            handle.write("")

        photos_dir = os.path.join(run_dir, "photos")
        os.makedirs(photos_dir)

        args = self.make_args([self.grid_dir, "--verbose", "--restart"])
        args.grid_dir = self.grid_dir
        args.log_path = "out_RUN_NAME"
        args.skip_if_file_exists = ""
        args.source = ""
        args.cmd_pre_each = ""
        args.cmd_post_each = ""
        args.cmd_main = "./rn"
        args.restart_settings = {"0000": "x999"}

        sentinel = mock.MagicMock()

        with mock.patch.object(mesa_go, "run_cmd", return_value=sentinel) as mock_run_cmd:
            with mock.patch.object(mesa_go, "copy_base_work_dir"):
                with mock.patch.object(mesa_go.os, "access", return_value=True):
                    run_name, out = mesa_go.start_mesa(args, "0000", self.logger)

        self.assertEqual(run_name, "0000")
        self.assertIs(out, sentinel)

        call_args = mock_run_cmd.call_args
        self.assertIn("./rn 2>&1", call_args.args[0])
        self.assertEqual(call_args.kwargs["file_mode"], "w")

    def test_start_mesa_source_and_each_commands(self):
        run_dir = os.path.join(self.grid_dir, "0000")
        os.makedirs(run_dir)

        star_path = os.path.join(run_dir, "star")
        with open(star_path, "w") as handle:
            handle.write("")

        source_path = os.path.join(self.grid_dir, "source.sh")
        with open(source_path, "w") as handle:
            handle.write("echo source\n")

        args = self.make_args([self.grid_dir, "--verbose"])
        args.grid_dir = self.grid_dir
        args.log_path = "out_RUN_NAME"
        args.skip_if_file_exists = ""
        args.source = source_path
        args.cmd_pre_each = "touch preeach"
        args.cmd_post_each = "touch posteach"
        args.cmd_main = "./rn"
        args.restart = False
        args.restart_settings = None

        sentinel = mock.MagicMock()

        with mock.patch.object(mesa_go, "run_cmd", return_value=sentinel) as mock_run_cmd:
            with mock.patch.object(mesa_go, "copy_base_work_dir"):
                with mock.patch.object(mesa_go.os, "access", return_value=True):
                    mesa_go.start_mesa(args, "0000", self.logger)

        commands = [call.args[0] for call in mock_run_cmd.call_args_list]

        self.assertIn("touch preeach", commands)
        self.assertIn("touch posteach", commands)

        main_call = [call for call in mock_run_cmd.call_args_list if "out_0000" in call.kwargs.get("to_file", "")]
        self.assertEqual(len(main_call), 1)
        self.assertIn("source.sh", main_call[0].args[0])

    def test_queue_start_mesa_success(self):
        args = self.make_args([self.grid_dir])
        queue = mp.Queue()
        queue.put((args, "0000"))
        queue.put(None)
        failed = mp.Value("b", 0)

        with mock.patch.object(mesa_go, "start_mesa") as mock_start:
            mesa_go.queue_start_mesa(queue, self.logger, failed)

        mock_start.assert_called_once()
        self.assertEqual(mock_start.call_args.args[1], "0000")
        self.assertEqual(failed.value, 0)

    def test_queue_start_mesa_failure(self):
        args = self.make_args([self.grid_dir])
        queue = mp.Queue()
        queue.put((args, "0000"))
        queue.put(None)
        failed = mp.Value("b", 0)

        with mock.patch.object(mesa_go, "start_mesa", side_effect=Exception("boom")):
            mesa_go.queue_start_mesa(queue, self.logger, failed)

        self.assertEqual(failed.value, 1)

    def test_get_slurm_task_info_no_task_share(self):
        self.assertEqual(mesa_go.get_slurm_task_info(False), (1, 0))

    def test_get_slurm_task_info_basic(self):
        env = {
            "SLURM_ARRAY_TASK_ID": "2",
            "SLURM_ARRAY_TASK_COUNT": "4",
            "SLURM_ARRAY_TASK_MIN": "1",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual(mesa_go.get_slurm_task_info(True), (4, 1))

    def test_get_slurm_task_info_task_list(self):
        env = {
            "SLURM_ARRAY_TASK_ID": "3",
            "SLURM_ARRAY_TASK_LIST": "1,3,5",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual(mesa_go.get_slurm_task_info(True), (3, 1))

    def test_get_slurm_task_info_task_id_not_in_list(self):
        env = {
            "SLURM_ARRAY_TASK_ID": "4",
            "SLURM_ARRAY_TASK_LIST": "1,3,5",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            with self.assertRaises(ValueError):
                mesa_go.get_slurm_task_info(True)

    def test_get_slurm_task_info_missing_env(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                mesa_go.get_slurm_task_info(True)

    def test_get_slurm_task_info_invalid_count(self):
        env = {
            "SLURM_ARRAY_TASK_ID": "1",
            "SLURM_ARRAY_TASK_COUNT": "0",
            "SLURM_ARRAY_TASK_MIN": "1",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            with self.assertRaises(ValueError):
                mesa_go.get_slurm_task_info(True)

    def test_get_slurm_task_info_task_id_below_min(self):
        env = {
            "SLURM_ARRAY_TASK_ID": "0",
            "SLURM_ARRAY_TASK_COUNT": "4",
            "SLURM_ARRAY_TASK_MIN": "1",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            with self.assertRaises(ValueError):
                mesa_go.get_slurm_task_info(True)

    def test_main_failure_raises(self):
        args = self.make_args([self.grid_dir])
        args.grid_dir = self.grid_dir
        args.num_mesa = 1

        queue = mock.MagicMock()
        pool = mock.MagicMock()
        failed = mock.MagicMock()
        failed.value = 1

        with mock.patch.object(mesa_go, "check_cores"):
            with mock.patch.object(mesa_go, "get_subdirs", return_value=["0000"]):
                with mock.patch.object(mesa_go, "get_slurm_task_info", return_value=(1, 0)):
                    with mock.patch.object(mesa_go.mp, "Queue", return_value=queue):
                        with mock.patch.object(mesa_go.mp, "Value", return_value=failed):
                            with mock.patch.object(mesa_go.mp, "Pool", return_value=pool):
                                with mock.patch.object(mesa_go, "run_cmd"):
                                    with self.assertRaises(RuntimeError):
                                        mesa_go.main(args, self.logger)

    def test_main_keyboard_interrupt(self):
        args = self.make_args([self.grid_dir])
        args.grid_dir = self.grid_dir
        args.num_mesa = 1

        queue = mock.MagicMock()
        pool = mock.MagicMock()
        pool.join.side_effect = KeyboardInterrupt
        failed = mock.MagicMock()
        failed.value = 0

        with mock.patch.object(mesa_go, "check_cores"):
            with mock.patch.object(mesa_go, "get_subdirs", return_value=["0000"]):
                with mock.patch.object(mesa_go, "get_slurm_task_info", return_value=(1, 0)):
                    with mock.patch.object(mesa_go.mp, "Queue", return_value=queue):
                        with mock.patch.object(mesa_go.mp, "Value", return_value=failed):
                            with mock.patch.object(mesa_go.mp, "Pool", return_value=pool):
                                with mock.patch.object(mesa_go, "run_cmd"):
                                    with self.assertRaises(KeyboardInterrupt):
                                        mesa_go.main(args, self.logger)

        queue.cancel_join_thread.assert_called_once()
        pool.terminate.assert_called_once()

    def test_main_task_share_filters_subdirs(self):
        args = self.make_args([self.grid_dir])
        args.grid_dir = self.grid_dir
        args.num_mesa = 1
        args.task_share = True

        queue = mock.MagicMock()
        pool = mock.MagicMock()
        failed = mock.MagicMock()
        failed.value = 0

        with mock.patch.object(mesa_go, "check_cores"):
            with mock.patch.object(mesa_go, "get_subdirs", return_value=["0000", "0001", "0002"]):
                with mock.patch.object(mesa_go, "get_slurm_task_info", return_value=(2, 1)):
                    with mock.patch.object(mesa_go.mp, "Queue", return_value=queue):
                        with mock.patch.object(mesa_go.mp, "Value", return_value=failed):
                            with mock.patch.object(mesa_go.mp, "Pool", return_value=pool):
                                with mock.patch.object(mesa_go, "run_cmd"):
                                    mesa_go.main(args, self.logger)

        put_args = [call.args[0] for call in queue.put.call_args_list]
        self.assertIn((args, "0001"), put_args)
        self.assertNotIn((args, "0000"), put_args)
        self.assertNotIn((args, "0002"), put_args)

    def test_run_success(self):
        args = self.make_args([self.grid_dir])

        parser = mock.MagicMock()
        parser.parse_args.return_value = args

        with mock.patch.object(mesa_go, "get_parser", return_value=parser):
            with mock.patch.object(mesa_go, "process_args", return_value=args):
                with mock.patch.object(mesa_go.mp, "get_logger", return_value=self.logger):
                    with mock.patch.object(mesa_go, "main") as mock_main:
                        with mock.patch.object(sys, "argv", ["mesa-go", self.grid_dir]):
                            self.assertEqual(mesa_go.run(), 0)

        mock_main.assert_called_once()

    def test_run_keyboard_interrupt(self):
        args = self.make_args([self.grid_dir])

        parser = mock.MagicMock()
        parser.parse_args.return_value = args

        with mock.patch.object(mesa_go, "get_parser", return_value=parser):
            with mock.patch.object(mesa_go, "process_args", return_value=args):
                with mock.patch.object(mesa_go.mp, "get_logger", return_value=self.logger):
                    with mock.patch.object(mesa_go, "main", side_effect=KeyboardInterrupt):
                        with mock.patch.object(sys, "argv", ["mesa-go", self.grid_dir]):
                            self.assertEqual(mesa_go.run(), 130)

    def test_run_exception(self):
        args = self.make_args([self.grid_dir])

        parser = mock.MagicMock()
        parser.parse_args.return_value = args

        with mock.patch.object(mesa_go, "get_parser", return_value=parser):
            with mock.patch.object(mesa_go, "process_args", return_value=args):
                with mock.patch.object(mesa_go.mp, "get_logger", return_value=self.logger):
                    with mock.patch.object(mesa_go, "main", side_effect=Exception("boom")):
                        with mock.patch.object(sys, "argv", ["mesa-go", self.grid_dir]):
                            self.assertEqual(mesa_go.run(), 1)
