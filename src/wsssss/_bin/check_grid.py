#!/usr/bin/env python
# -*- coding: utf-8 -*-

import ast
import glob
import os
from argparse import ArgumentParser
from collections import Counter
from collections import defaultdict

import numpy as np
import warnings

from wsssss import load_data as ld
from wsssss import functions as uf

warnings.filterwarnings("error")


def get_parser():
    """
    Return argument parser for checking a MESA grid.
    """
    _parser = ArgumentParser(description="""
    This program will check whether a grid has run successfully.
    """)
    _parser.add_argument(
        "--verbose",
        "-v",
        action="store_const",
        const=True,
        default=False,
    )
    _parser.add_argument(
        "grid_dir",
        type=str,
        default=".",
        nargs="?",
        help="The directory which contains each run for MESA to run in a separate sub-directory.",
    )
    _parser.add_argument(
        "--subdirs",
        "-d",
        nargs="*",
        type=str,
        default=[],
        help="The names of the sub directories if they are not named 0000 0001 etc.",
    )
    _parser.add_argument(
        "--history-file",
        "-f",
        type=str,
        default="history.data",
        help=(
            "Name of the MESA history file in the LOGS directory. {} will be "
            "expanded to the directory name. If it contains a star will glob match."
        ),
    )
    _parser.add_argument(
        "--exclude",
        nargs="*",
        type=str,
        default=["figs", "slurm*"],
        help="Directory names to exclude from sub-dirs (i.e. do not contain MESA runs).",
    )
    _parser.add_argument(
        "--out-file",
        "-o",
        type=str,
        default="../out_{}",
        help=(
            "File format for MESA out files (from eg ./rn | tee out_0000). "
            "Curly braces will be expanded with the directory name."
        ),
    )
    _parser.add_argument(
        "--make-restart-file",
        "-r",
        type=str,
        default="",
        help=(
            "Make a file which contains which subdir needs to restart from what photo. "
            "Pass `pre-CHeX` to only restart runs which did not finish CHeX, pass `redoRC` to "
            "restart before CHeB or `all` to do for all."
        ),
    )
    _parser.add_argument(
        "--min-photo-delta",
        type=int,
        default=0,
        help="Minimum model number difference between last model and photo model number.",
    )
    _parser.add_argument(
        "--list-all",
        action="store_const",
        const=True,
        default=False,
        help="Print all termination reasons.",
    )
    _parser.add_argument(
        "--full-termcode",
        action="store_const",
        const=True,
        default=False,
        help=(
            "If set, scan the full MESA terminal output file for a termination code, "
            "otherwise only search in the last part of the file."
        ),
    )
    return _parser


def read_hist_first_last_row(fpath):
    """
    Read the first and last full row of a MESA history file.
    """
    with open(fpath, "r") as f:
        for _ in range(6):
            columns = f.readline()
        first_row = f.readline()
        f.seek(0, 2)
        size = f.tell()
        f.seek(max(0, size - 3 * len(first_row)))
        last_rows = f.readlines()
    # Last row incomplete or mangled, use second to last row
    if len(last_rows[-1]) == len(first_row):
        last_row = last_rows[-1]
    else:
        last_row = last_rows[-2]

    first_row = tuple(first_row.strip().split())
    last_row = tuple(last_row.strip().split())
    columns = columns.strip().split()
    formats = [np.array(ast.literal_eval(_)).dtype if _ != "NaN" else np.float64 for _ in first_row]
    return np.rec.array([first_row, last_row], dtype=list(zip(columns, formats)))


def get_termination_code(lines):
    """
    Extract a MESA termination code from the last part of the terminal output.
    """
    lines = lines[-min(100, len(lines)) :]
    term_code = "Running"
    for line in lines[::-1]:
        if "termination code" in line:
            term_code = line.split(":")[1].strip()
            break

    if term_code == "Running":  # Didn't find normal termination code
        for line in lines:
            if "STOP" in line:
                term_code = line.split("STOP")[1].strip()
                break
            if "Backtrace" in line:
                term_code = "Error"
            if ("adjust_mesh_failed" in line) or ("mesh_plan problem" in line):
                term_code = "mesh_failed"
                break
            if "stopping because of problems dt < min_timestep_limit" in line:
                term_code = "min_timestep_limit"
                break
            if "stop because hit EOS limits" in line:
                term_code = "EOS_limits"
                break
            if "stopping because of problems" in line:
                term_code = line.split("--")[-1].strip()
                break
            if "terminated evolution: nonzero_ierr" in line:
                term_code = "nonzero_ierr"

    return term_code


def get_mesa_termcode(sub_dir, args):
    """
    Get the MESA termination code for a given run.
    """
    out_file = os.path.join(args.grid_dir, sub_dir, args.out_file).format(sub_dir)
    if os.path.isfile(out_file):
        with open(out_file, "r") as handle:
            if not args.full_termcode:
                # Seek to end of file and look in end of file for termination code
                handle.seek(0, os.SEEK_END)
                size = handle.tell()
                handle.seek(max(0, size - 4096 * 2))  # Go back atleast ~28 lines of full MESA output (146 chars)
                out_file_lines = handle.readlines()[1:]  # First line might be garbled
            else:
                out_file_lines = handle.readlines()
        term_code = get_termination_code(out_file_lines)
    else:
        term_code = "NotRun"
    return sub_dir, term_code


def count_in_dict(info_dict, key, filter_mainkeys=None):
    """
    Count values in a nested dictionary, optionally filtered by main keys.
    """
    if filter_mainkeys is not None:
        filtered = dict((k, v[key]) for k, v in info_dict.items() if k in filter_mainkeys)
    else:
        filtered = dict((k, v[key]) for k, v in info_dict.items())
    cnt = Counter(filtered.values())
    dct = defaultdict(list)
    for main_key, value in filtered.items():
        dct[value].append(main_key)
    return cnt, dct


def get_history_path(subdir, args):
    """
    Return the history file path for a run.
    """
    if "*" in args.history_file:
        matches = glob.glob(os.path.join(args.grid_dir, subdir, args.history_file))

        if len(matches) > 1:
            raise ValueError(f"Found too many files:\n{matches}")
        elif len(matches) == 0:
            return ""
        else:
            return matches[0]

    return os.path.join(
        args.grid_dir,
        subdir,
        f"LOGS/{args.history_file.format(subdir, subdir)}",
    )


def run():
    parser = get_parser()
    args = parser.parse_args()

    args.grid_dir = os.path.abspath(args.grid_dir)

    # Expand items in args.exclude.
    _ = []
    for exclude in args.exclude:
        _ += glob.glob(os.path.join(args.grid_dir, exclude))
    args.exclude = [os.path.split(_)[-1] for _ in _]

    # Get subdirs and remove items from args.exclude.
    if args.subdirs == []:
        subdirs = os.listdir(args.grid_dir)
        subdirs = sorted(subdirs)
        subdirs = [subdir for subdir in subdirs if os.path.isdir(os.path.join(args.grid_dir, subdir))]
        args.subdirs = subdirs

    expanded_subdirs = []
    for subdir in args.subdirs:
        if subdir in args.exclude:
            continue

        expanded = glob.glob(os.path.join(args.grid_dir, subdir))
        expanded_subdirs += expanded

    args.subdirs = [os.path.split(path)[-1] for path in expanded_subdirs]
    args.subdirs = list(dict.fromkeys(args.subdirs))  # Remove duplicates

    run_info = {}
    run_data = {}

    for subdir in args.subdirs:
        hist_path = get_history_path(subdir, args)
        mesa_termcode = get_mesa_termcode(subdir, args)[1]

        if os.path.isfile(hist_path) and mesa_termcode != "NotRun":
            data = read_hist_first_last_row(hist_path)
            run_info[subdir] = {
                "mesa_termcode": mesa_termcode,
                "mesa_last_model": data.model_number[-1],
                "CHeX": ["pre-CHeX", "post-CHeX"][int(data.center_he4[-1] < 1e-6)],
                "CHX": ["pre-CHX", "post-CHX"][int(data.center_h1[-1] < 1e-6)],
            }

            run_data[subdir] = data
        else:
            run_info[subdir] = {
                "mesa_termcode": mesa_termcode,
                "mesa_last_model": "-1",
                "CHeX": mesa_termcode,
                "CHX": mesa_termcode,
            }

            run_data[subdir] = mesa_termcode

    cnt_term_codes_mesa, dct_term_codes_mesa = count_in_dict(run_info, key="mesa_termcode")

    # Make output string.
    part_len = 44
    string_separator = "-" * part_len + "\n"
    status_str = "" + string_separator
    status_str += f'{"  termination_code":<32}{"count":>10}\n'
    status_str += string_separator
    for mesa_tc, c_mesa_tc in cnt_term_codes_mesa.items():
        status_str += f"{mesa_tc:<32}{c_mesa_tc:>12}\n"

        # Count CHeX status for each MESA termination code.
        cnt_CHeX, dct_CHeX = count_in_dict(
            run_info,
            key="CHeX",
            filter_mainkeys=dct_term_codes_mesa[mesa_tc],
        )

        for i, che_x in enumerate(sorted(cnt_CHeX)):
            part = f"{che_x:<29}{cnt_CHeX[che_x]:>9}\n"

            start_part = " ├ "
            if i == len(cnt_CHeX) - 1:
                start_part = " └ "

            status_str += start_part + part

            # For pre-CHeX runs, count CHX status.
            if che_x == "pre-CHeX":
                cnt_CHX, dct_CHX = count_in_dict(
                    run_info,
                    key="CHX",
                    filter_mainkeys=dct_CHeX[che_x],
                )

                for j, ch_x in enumerate(sorted(cnt_CHX)):
                    part = f"{ch_x:<26}{cnt_CHX[ch_x]:>6}\n"

                    start_part = "    ├ "
                    if j == len(cnt_CHX) - 1:
                        start_part = "    └ "

                    status_str += start_part + part

    status_str += string_separator

    if args.list_all:
        for subdir in args.subdirs:
            status_str += (
                f"{subdir:<16} "
                f'{run_info[subdir]["CHeX"]:>10} '
                f'{run_info[subdir]["mesa_last_model"]:>7} '
                f'{run_info[subdir]["mesa_termcode"]:<12}\n'
            )

        status_str += string_separator

    print(status_str)

    if args.make_restart_file:
        restart_reason = {}

        for subdir in args.subdirs:
            info = run_info[subdir]

            reason = ""
            photo = ""
            mesa_termcode = info["mesa_termcode"]

            if info["CHeX"] == "post-CHeX":
                reason = "redoRC"
                photo = "preRC"
            elif mesa_termcode == "Running":
                reason = "Running"
                photo = "last"
            elif info["CHeX"] == "pre-CHeX":
                reason = "None"
                photo = "last"
            elif args.make_restart_file == "redoRC":
                reason = "redoRC"
                photo = "preRC"
            elif mesa_termcode in ["logQ_min_limit", "FileNotFound", "NotRun"]:
                reason = mesa_termcode
                photo = "full_restart"

            restart_reason[subdir] = (photo, reason)

        grid_restart_photos = []
        for subdir, (photo, reason) in restart_reason.items():
            photodir = os.path.join(args.grid_dir, subdir, "photos")
            if os.path.exists(photodir):
                photos = sorted(os.listdir(photodir))
            else:
                photos = []
            photos = [_ for _ in photos if _.replace("x", "").isdigit()]
            photo_modelnum = np.array([int(_.replace("x", "")) for _ in photos])

            if args.make_restart_file == "pre-CHeX":
                info = run_info[subdir]
                if info["CHeX"] == "post-CHeX":
                    continue

            restart_photo = ""
            try:
                if photo == "last":
                    restart_photo = photos[
                        np.where((photo_modelnum[-1] - photo_modelnum) >= args.min_photo_delta)[0][-1]
                    ]
                elif photo == "preRC":
                    if "*" in args.history_file:
                        matches = glob.glob(os.path.join(args.grid_dir, subdir, args.history_file))
                        if len(matches) > 1:
                            raise ValueError(f"Found too many files:\n{matches}")
                        elif len(matches) == 0:
                            hist_path = ""
                        else:
                            hist_path = matches[0]
                    else:
                        hist_path = os.path.join(
                            args.grid_dir,
                            subdir,
                            f"LOGS/{args.history_file.format(subdir, subdir)}",
                        )
                    if os.path.exists(hist_path) and hist_path != "":
                        hist = ld.History(hist_path, index_name=None)
                        cheb_mask = uf.get_cheb_mask(hist)
                        first_model_rc = hist.data.model_number[cheb_mask][0]
                        restart_photo = photos[
                            np.where((first_model_rc - photo_modelnum) >= args.min_photo_delta)[0][-1]
                        ]
                    else:
                        restart_photo = "full_restart"
                elif photo == "full_restart":
                    restart_photo = "full_restart"
                elif photo == "":
                    restart_photo = "full_restart"
            except IndexError as e:
                print(subdir, photo, reason, e)
                restart_photo = "full_restart"
            if args.verbose:
                print(subdir, restart_photo)

            if restart_photo == "":
                print(f"Empty restart_photo for {subdir}!")
            grid_restart_photos.append((subdir, restart_photo))

        with open(os.path.join(args.grid_dir, "grid_restart"), "w") as f:
            f.writelines((" ".join(line) + "\n" for line in grid_restart_photos))
    return 0


if __name__ == "__main__":
    run()
