#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""MESA grid overseer."""

import functools
import itertools
import multiprocessing as mp
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
from argparse import ArgumentParser


def get_parser():
    """Return command-line parser."""
    # TODO: Add option to compile star once and then copy the executable?
    parser = ArgumentParser(description="""
    This program will run a MESA grid created with ``wsssss.inlists.create_grid`` module.
    """)
    try:
        env_OMP_NUM_THREADS = os.environ["OMP_NUM_THREADS"]
    except KeyError:
        print("Environment variable ``OMP_NUM_THREADS`` not set, using ``nproc``.")
        env_OMP_NUM_THREADS = mp.cpu_count()

    parser.add_argument(
        "grid_dir",
        type=str,
        default=".",
        nargs="?",
        help="The directory which contains each run for MESA to run in a separate sub-directory.",
    )

    parser.add_argument(
        "--sub-dirs",
        "-d",
        nargs="*",
        type=str,
        default=[],
        help="The names of the sub directories if they are not named 0000 0001 etc.",
    )

    parser.add_argument(
        "--base-work-dir",
        "-b",
        type=str,
        default="$MESA_DIR/star/work",
        help=(
            "Directory to use as the base work directory. Missing files from "
            "this directory will be copied into each run directory. Existing "
            "files in the run directory are not overwritten."
        ),
    )

    parser.add_argument(
        "--restart",
        "-re",
        default=False,
        const=True,
        nargs="?",
        help=(
            "Restart from the last photo in each run's photos directory. "
            "If no photo is found, start a new run. If a check-grid "
            "grid_restart file is passed, use that instead."
        ),
    )
    parser.add_argument(
        "--source",
        type=str,
        default="",
        help="Source this file before running ``star``.",
    )

    parser.add_argument(
        "--cmd-pre",
        type=str,
        default="",
        help="Command to run before starting the grid. Can be a file.",
    )

    parser.add_argument(
        "--cmd-pre-each",
        type=str,
        default="",
        help="Command to run directly before starting ``star`` in each directory. Can be a file.",
    )

    parser.add_argument(
        "--cmd-main",
        type=str,
        default="./rn",
        help="Command to run MESA. Defaults to ``./rn``.",
    )

    parser.add_argument(
        "--cmd-post-each",
        type=str,
        default="",
        help="Command to run directly after ``star`` has finished in each directory. Can be a file.",
    )

    parser.add_argument(
        "--cmd-post",
        type=str,
        default="",
        help="Command to run after the grid is finished. Can be a file.",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_const",
        const=True,
        default=False,
        help="Print info about what the script is doing.",
    )

    parser.add_argument(
        "--num-mesa",
        "-n",
        type=int,
        default=1,
        help="Number of instances of MESA to run at once.",
    )

    parser.add_argument(
        "--OMP_NUM_THREADS",
        "-j",
        type=int,
        default=env_OMP_NUM_THREADS,
        help=(
            "Value of the OMP_NUM_THREADS environment variable to set before "
            "running ``star``. ``num-mesa`` times ``OMP_NUM_THREADS`` cannot "
            "be larger than ``nproc``. Defaults to ``os.cpu_count()`` if "
            "``OMP_NUM_THREADS`` environment variable is not set."
        ),
    )

    parser.add_argument(
        "--skip-if-file-exists",
        type=str,
        default="",
        help=(
            "Skip running MESA in this work_dir if this file exists in "
            "work_dir. Can use MESAGO_LOG_FILE to look for log file generated "
            "by this script. WORK_DIR and RUN_NAME will be replaced with the "
            "run directory and run name, respectively."
        ),
    )

    parser.add_argument(
        "--log-path",
        type=str,
        default="out_RUN_NAME",
        help="Log file path. WORK_DIR and RUN_NAME will be replaced by their corresponding values.",
    )

    parser.add_argument(
        "--task-share",
        action="store_const",
        const=True,
        default=False,
        help="Set the slurm task-id and only do runs assigned to that task-id.",
    )

    return parser


def expand_path(path):
    """Expand user, environment variables, and return an absolute path."""
    if path == "":
        return ""
    return os.path.abspath(os.path.expanduser(os.path.expandvars(path)))


def resolve_path_template(template, grid_dir, run_name):
    """Resolve WORK_DIR/RUN_NAME templates."""
    if template == "":
        return ""

    path = template.replace("WORK_DIR", os.path.join(grid_dir, run_name)).replace("RUN_NAME", run_name)

    if not os.path.isabs(path):
        path = os.path.join(grid_dir, path)

    return os.path.abspath(path)


def check_cores(args):
    """Check and process core count arguments."""
    nproc = os.cpu_count()
    if nproc is None:
        raise ValueError("Cannot determine number of available CPUs.")

    if args.num_mesa < 1:
        raise ValueError("num-mesa must be at least 1.")

    if args.OMP_NUM_THREADS < 1:
        raise ValueError("OMP_NUM_THREADS must be at least 1.")

    req_cores = args.num_mesa * args.OMP_NUM_THREADS

    if req_cores > nproc:
        raise ValueError(f"Number of required cores ({req_cores}) larger than available cores ({nproc}).")

    return req_cores, nproc


def process_args(args):
    """Process and validate parsed arguments."""
    args.base_work_dir = expand_path(args.base_work_dir)
    args.grid_dir = expand_path(args.grid_dir)

    if not os.path.isdir(args.grid_dir):
        raise FileNotFoundError(f"grid_dir does not exist: {args.grid_dir}")

    if args.base_work_dir and not os.path.isdir(args.base_work_dir):
        raise FileNotFoundError(f"base_work_dir does not exist: {args.base_work_dir}")

    if args.source != "":
        args.source = expand_path(args.source)
        if not os.path.isfile(args.source):
            raise FileNotFoundError(f"source file does not exist: {args.source}")

    if args.log_path != "":
        if not (("WORK_DIR" in args.log_path) or ("RUN_NAME" in args.log_path)):
            raise ValueError("log-path must contain one or both of WORK_DIR or RUN_NAME in its path.")

    if args.skip_if_file_exists != "":
        if args.skip_if_file_exists != "MESAGO_LOG_FILE":
            if not (("WORK_DIR" in args.skip_if_file_exists) or ("RUN_NAME" in args.skip_if_file_exists)):
                raise ValueError(
                    "skip-if-file-exists must contain one or both of WORK_DIR "
                    "or RUN_NAME in its path, unless it is MESAGO_LOG_FILE."
                )

    args.restart_settings = None
    if isinstance(args.restart, str):
        args.restart = expand_path(args.restart)

        if not os.path.isfile(args.restart):
            raise FileNotFoundError(f"restart file does not exist: {args.restart}")

        restart_settings = {}

        with open(args.restart, "r") as handle:
            for lineno, line in enumerate(handle, start=1):
                line = line.strip()

                if not line or line.startswith("#"):
                    continue

                parts = line.split("#", maxsplit=1)[0].strip().split()
                if len(parts) != 2:
                    raise ValueError(f"Bad restart file line {lineno}: {line!r}. Expected '<dirname> <photo>'.")

                dirname, photo = parts
                restart_settings[dirname] = photo

        args.restart_settings = restart_settings
        args.restart = True

    return args


def get_subdirs(args):
    """Return subdirectories to run."""
    if args.sub_dirs:
        sub_dirs = list(args.sub_dirs)

        for name in sub_dirs:
            path = os.path.join(args.grid_dir, name)
            if not os.path.isdir(path):
                raise FileNotFoundError(f"sub-dir does not exist: {path}")

        return sub_dirs

    sub_dirs = []

    for item in os.listdir(args.grid_dir):
        item_path = os.path.join(args.grid_dir, item)
        if os.path.isdir(item_path) and item.isdigit():
            sub_dirs.append(item)

    sub_dirs.sort()
    return sub_dirs

def choose_restart_photo(args, photos, run_name):

def copy_base_work_dir(args, run_dir, logger):
    """
    Copy missing files from the base work directory into the run directory.
    Existing files in the run directory are not overwritten.
    """
    if not args.base_work_dir:
        return

    base_dir = args.base_work_dir

    if os.path.abspath(base_dir) == os.path.abspath(run_dir):
        return

    pid = os.getpid()
    logger.info(f"{pid}: Copying missing files from base work dir: {base_dir}")

    for root, dirs, files in os.walk(base_dir):
        rel_root = os.path.relpath(root, base_dir)
        target_root = os.path.join(run_dir, rel_root)

        os.makedirs(target_root, exist_ok=True)

        for directory in dirs:
            target_dir = os.path.join(target_root, directory)
            os.makedirs(target_dir, exist_ok=True)

        for filename in files:
            source_path = os.path.join(root, filename)
            target_path = os.path.join(target_root, filename)

            if not os.path.exists(target_path):
                shutil.copy2(source_path, target_path)


    """
    Return (photo, run_new).

    photo is None if a new run should be started.
    """
    if not args.restart:
        return None, True

    if len(photos) == 0:
        return None, True

    # User supplied restart file.
    if args.restart_settings is not None:
        if run_name not in args.restart_settings:
            return None, True

        photo = args.restart_settings[run_name]
        if photo == 'full_restart':
            return None, True
        return photo, False

    # Auto-select latest numeric photo.
    numeric_photos = []

    for photo in photos:
        cleaned = photo.replace("x", "0")
        if cleaned.isdigit():
            numeric_photos.append((int(cleaned), photo))

    if not numeric_photos:
        return None, True

    numeric_photos.sort(key=lambda item: item[0])
    photo = numeric_photos[-1][1]

    return photo, False


def photo_exists(photo):
    """Check whether a restart photo exists."""
    return os.path.exists(os.path.join("photos", photo)) or os.path.exists(photo)

def start_mesa(args, run_name, logger):
    pid = os.getpid()
    logger.info(f"{pid}: Starting {run_name}")

    os.chdir(os.path.join(args.grid_dir, run_name))
    if args.verbose:
        logger.info(f"{pid}: Current directory: {os.getcwd()}")

    log_file = resolve_path_template(args.log_path, args.grid_dir, run_name)

    if log_file == "":
        log_file = os.path.join(args.grid_dir, "out_" + run_name)

    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    if args.skip_if_file_exists:
        if args.skip_if_file_exists == "MESAGO_LOG_FILE":
            skip_check_fpath = log_file
        else:
            skip_check_fpath = resolve_path_template(
                args.skip_if_file_exists,
                args.grid_dir,
                run_name,
            )

        if os.path.exists(skip_check_fpath):
            logger.info(f"{pid}: Skipping {run_name}, skip file found {skip_check_fpath}.")
            return run_name, "Skip file found."

    with open(
        log_file, "w"
    ) as handle:  # Create file so other processes can find it before someting actually writes to it
        handle.write("")
        handle.flush()
        os.fsync(handle)

    pre_cmd_str = f"export OMP_NUM_THREADS={args.OMP_NUM_THREADS}; "
    if args.source != "":
        pre_cmd_str = f". {args.source}; " + pre_cmd_str

    if args.cmd_pre_each != "":
        run_cmd(args.cmd_pre_each, shell=True)

    copy_base_work_dir(args, run_dir, logger)

    if not os.path.exists("star"):
        run_cmd("./clean && ./mk", shell=True)

    if os.path.exists("photos"):
        photos = os.listdir("photos")
    else:
        photos = []

    photo, run_new = choose_restart_photo(args, photos, run_name)

    if not run_new:
        if photo_exists(photo):
            cmd = pre_cmd_str + f"./re {shlex.quote(photo)}" + " 2>&1"
            file_mode = "a"
        else:
            logger.info(f"{pid}: Restart photo {photo} not found for {run_name}; starting new run.")
            photo = None
            run_new = True

    if run_new:
        cmd = pre_cmd_str + args.cmd_main + " 2>&1"
        file_mode = "w"

    out = run_cmd(
        cmd,
        split=False,
        shell=True,
        to_file=log_file,
        file_mode=file_mode,
    )

    if args.cmd_post_each != "":
        run_cmd(args.cmd_post_each, shell=True)

    logger.info(f"{pid}: Finished {run_name}")

    return run_name, out


def queue_start_mesa(queue, logger):
    while True:
        item = queue.get(block=True)
        if item is None:
            break

        start_mesa(*item, logger=logger)


def run_cmd(cmd, capture_output=False, split=False, to_file="", file_mode="w", **kwargs):
    pid = os.getpid()
    if file_mode not in ["a", "-a", "w"]:
        raise ValueError("`file_mode` must be one of `a`, `-a`, or `w`.")

    if capture_output and to_file != "":
        logger.warning(
            f"{pid}: Cannot simultaneously capture output and write to file if `split=True`, setting to False."
        )
        split = False

    if split:
        cmd = cmd.split()

    if to_file != "" and not capture_output:
        with open(to_file, file_mode) as f:
            return subprocess.run(cmd, stdout=f, stdin=f, **kwargs)
    if capture_output:
        if to_file != "":
            if file_mode == "a":
                file_mode = "-a"
            elif file_mode == "w":
                file_mode = ""
            return subprocess.run(
                cmd + f"| tee {file_mode} {to_file}",
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=True,
                **kwargs,
            )
        return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **kwargs)
    return subprocess.run(cmd, **kwargs)




def main(args, logger):
    # TODO: include walltime?

    if args.verbose:
        print("Parsed inputs:")
        for key, value in args.__dict__.items():
            print(f"{key:<16}= {value}")
        print("")

    check_cores(args)

    if args.cmd_pre != "":
        run_cmd(args.cmd_pre, shell=True)

    sub_dirs = get_subdirs(args)

    arguments = list(zip(itertools.repeat(args), sub_dirs))
    if args.task_share:
        try:
            n_tasks = int(os.environ["SLURM_ARRAY_TASK_COUNT"])
            task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
            task_id_min = int(os.environ["SLURM_ARRAY_TASK_MIN"])
            task_id = task_id - task_id_min
        except KeyError:  # Not running with a task array
            n_tasks = 1
            task_id = 0
        print(f"n_tasks={n_tasks}")
        print(f"task_id={task_id}")
    else:
        n_tasks = 1
        task_id = 0

    if args.verbose:
        print("Sub-directories:")
        for subdir in sub_dirs[task_id::n_tasks]:
            print(subdir)
        print("")

    _queue_start_mesa = functools.partial(queue_start_mesa, logger=logger)

    queue = mp.Queue()
    pool = mp.Pool(args.num_mesa, _queue_start_mesa, (queue,))
    for item in arguments[task_id::n_tasks]:
        queue.put(item)
    for i in range(args.num_mesa):
        queue.put(None)
    queue.close()
    queue.join_thread()
    pool.close()
    pool.join()

    if args.cmd_post != "":
        run_cmd(args.cmd_post, shell=True)


def run():
    try:
        import setproctitle

        setproctitle.setproctitle("MESAgo")
    except ModuleNotFoundError:
        print("Module setproctitle not found, will not set process name.")
        pass

    signal.signal(signal.SIGTERM, signal.default_int_handler)
    signal.signal(signal.SIGINT, signal.default_int_handler)

    t_start = time.time()

    parser = get_parser()
    args = parser.parse_args()

    args = process_args(args)

    logger = mp.get_logger()

    if args.debug:
        args.verbose = True
        logger.setLevel("DEBUG")
    else:
        logger.setLevel("INFO")

    main(args, logger)
    return 0


if __name__ == "__main__":
    run()
