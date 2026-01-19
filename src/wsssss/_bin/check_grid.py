#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import ast
import glob
import os
import sys
from argparse import ArgumentParser
from collections import Counter
from collections import defaultdict
from datetime import datetime

import numpy as np
import warnings

from wsssss import load_data as ld
from wsssss import functions as uf

warnings.filterwarnings("error")
def get_parser():
    """"""
    _parser = ArgumentParser(description="""
    This program will check whether a grid has run successfully.
    """)
    _parser.add_argument('--verbose', '-v', action='store_const', const=True, default=False, )
    _parser.add_argument('grid_dir', type=str, default='.', nargs='?',
                         help='The directory which contains each run for MESA to run in a separate sub-directory.')
    _parser.add_argument('--subdirs', '-d', nargs='*', type=str, default='',
                         help='The names of the sub directories if they are not named 0000 0001 etc.')
    _parser.add_argument('--history-file', '-f', type=str, default='history.data',
                         help='Name of the MESA history file in the LOGS directory. {} will be expanded to the '
                              'directory name.')
    _parser.add_argument('--exclude', nargs='*', type=str, default=['figs', 'template_11701', 'slurm*'],
                         help='Directory names to exclude from sub-dirs (i.e. do not contain MESA runs).')
    _parser.add_argument('--out-file', '-o', type=str, default='../out_{}',
                         help='File format for MESA out files (from eg ./rn | tee out_file). Curly braces will be '
                              'expanded with with the directory name.')
    _parser.add_argument('--slurm-stats-dir', '-s', type=str, default='slurm_out',
                         help='Directory containing the slurm output (.err, .stats, .out files) for a job.')
    _parser.add_argument('--make-restart-file', '-r', type=str, default='',
                         help='Make a file which contains which subdir needs to restart from what photo.'
                              'Pass `pre-CHeX` to only restart runs which did not finish CHeX, pass `redoRC` to restart'
                              'before CHeB or `all` to do for all.')
    # _parser.add_argument('--good-termination-codes', '-t', nargs='*', type=list, default=['xa_central_lower_limit', 'stop_at_TP'], )
    _parser.add_argument('--no-slurm', action='store_const', const=True, default=False,
                         help='Run the check ignoring any slurm output.')
    _parser.add_argument('--list-all', action='store_const', const=True, default=False,
                         help='Print all termination reasons.')
    return _parser


def read_hist_first_last_row(fpath):
    with open(fpath, 'r') as f:
        for _ in range(6):
            columns = f.readline()
        first_row = f.readline()
        f.seek(0, 2)
        size = f.tell()
        f.seek(size - len(first_row)*2)
        last_row = f.readline()#.split('\n')[1]
    first_row = tuple(first_row.strip().split())
    last_row = tuple(last_row.strip().split())
    columns = columns.strip().split()
    formats = [np.array(ast.literal_eval(_)).dtype if _ != 'NaN' else np.float64 for _ in first_row]
    return np.rec.array([first_row, last_row], dtype=list(zip(columns, formats)))#dtype={'names': columns, 'formats': formats})


def get_termination_code(lines):
    # last 100 lines of MESA output
    lines = lines[-min(100, len(lines)):]
    term_code = 'Running'
    for line in lines[::-1]:
        if 'termination code' in line:
            term_code = line.split(':')[1].strip()
            break

    if term_code == 'Running':  # Didn't find normal termination code
        for line in lines:
            if 'STOP' in line:
                term_code = line.split('STOP')[1].strip()
                break
            if 'Backtrace' in line:
                term_code = 'Error'
            if ('adjust_mesh_failed' in line) or ('mesh_plan problem') in line:
                term_code = 'mesh_failed'
                break
            if 'stopping because of problems dt < min_timestep_limit' in line:
                term_code = 'min_timestep_limit'
                break
            if 'stop because hit EOS limits' in line:
                term_code = 'EOS_limits'
                break
            if 'stopping because of problems' in line:
                term_code = line.split('--')[-1].strip()
                break
            if 'terminated evolution: nonzero_ierr' in line:
                term_code = 'nonzero_ierr'

    return term_code


def get_timedelta(lines):
    first = lines[0].strip()
    last = lines[-1].strip()

    dts = []
    for timestamp in (first, last):
        try:
            _, day, month, hms, tz, year = timestamp.split()
        except ValueError:
            return dts[0] - dts[0]

        day = int(day)
        month = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
                 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}[month]
        year = int(year)

        hour, minute, second = (int(_) for _ in hms.split(':'))
        dts.append(datetime(year, month, day, hour, minute, second))
    timedelta = (dts[1] - dts[0])

    return timedelta


def get_mesa_termcode(sub_dir, args):
    out_file = os.path.join(args.grid_dir, sub_dir, args.out_file).format(sub_dir)
    if os.path.exists(out_file):
        with open(out_file, 'r') as handle:
            out_file_lines = handle.readlines()

        # slurm_stats_file = os.path.join(args.grid_dir, sub_dir, args.slurm_stats_file).format(sub_dir)

        term_code = get_termination_code(out_file_lines)
        # walltime = get_timedelta(out_file_lines)
    else:
        term_code = 'NotRun'

    return sub_dir, term_code


def get_slurm_stats(stats_lines, args):
    info_dict = {}
    if args.no_slurm:
        info_dict['jobid'] = ''
        info_dict['jobname'] = 'noslurm'
        info_dict['job_task'] = ''
        info_dict['taskjob'] = ''
        info_dict['taskid'] = ''
        info_dict['walltime_requested'] = '24:0:0'
        info_dict['walltime_required'] = '24:0:0'
        info_dict['reason'] = 'None'
        return info_dict

    for line in stats_lines:
        if 'Identity jobid' in line:
            _, _, _, _jobid, _, _jobname = line.split()
            info_dict['jobid'] = _jobid
            info_dict['jobname'] = _jobname
        if 'Array Identity' in line:
            _, _, _, _job_task = line.split()
            _taskjob, _taskid = _job_task.split('_')
            info_dict['job_task'] = _job_task
            info_dict['taskjob'] = _taskjob
            info_dict['taskid'] = _taskid
        else:
            info_dict['job_task'] = ''
            info_dict['taskjob'] = ''
            info_dict['taskid'] = ''

        if 'Requested' in line:
            _, _, _hardware, _, _walltime_requested, _ = line.split()
            info_dict['hardware'] = _hardware
            info_dict['walltime_requested'] = _walltime_requested
        if 'Required' in line:
            _walltime_required = line.split()[-2]
            info_dict['walltime_required'] = _walltime_required
        if 'JobState' in line:
            _, _, _jobstate, _, _, _reason = line.split()
            info_dict['jobstate'] = _jobstate
            info_dict['reason'] = _reason

    return info_dict


def get_cheb_mask(data):
    """
    Defined as:
    have helium in core
    have convective core
    hydrogen depleted core


    :param data:
    :return:
    """

    mask = data.center_he4 > 1e-6
    mask = np.logical_and(mask, data.mass_conv_core > 0)
    mask = np.logical_and(mask, data.center_h1 < 1e-6)

    return mask


def count_in_dict(info_dict, key, filter_mainkeys=None):
    if filter_mainkeys is not None:
        filtered = dict((k, v[key]) for k, v in info_dict.items() if k in filter_mainkeys)
    else:
        filtered = dict((k, v[key]) for k, v in info_dict.items())
    cnt = Counter(filtered.values())
    dct = defaultdict(list)
    for main_key, value in filtered.items():
        dct[value].append(main_key)
    return cnt, dct


def invert_dict(dct, reference_key, filter_dct_keys=None, reverse_filter=False, sort=False):
    if filter_dct_keys is None:
        filter_dct_keys = []

    out_dict = defaultdict(list)
    for key, value in dct.items():
        if (key in filter_dct_keys) != reverse_filter:  # xor
            continue
        out_dict[value[reference_key]].append(key)

    if sort:
        for key in sorted(out_dict):
            out_dict[key] = sorted(out_dict[key])
    return out_dict




if __name__ == "__main__":
    run()

def run():
    parser = get_parser()
    args = parser.parse_args()

    args.grid_dir = os.path.abspath(args.grid_dir)

    # Expand items in args.exclude.
    _ = []
    for exclude in args.exclude:
        _ += glob.glob(os.path.join(args.grid_dir, exclude))
    args.exclude = _
    args.exclude = [os.path.split(_)[-1] for _ in args.exclude]

    # Get subdirs and remove items from args.exclude.
    if args.subdirs == '':
        subdirs = os.listdir(args.grid_dir)
        subdirs = sorted(subdirs)
        subdirs = [_ for _ in subdirs if os.path.isdir(os.path.join(args.grid_dir, _))]
        args.subdirs = subdirs

    _ = [glob.glob(os.path.join(args.grid_dir, _)) for _ in args.subdirs if _ not in args.exclude]
    args.subdirs = []
    for expanded in _:
        args.subdirs += expanded
    args.subdirs = [os.path.split(_)[-1] for _ in args.subdirs]

    # Get info from slurm stats files.
    jobid2task = {}
    task2jobid = {}
    run_info = {}
    run_data = {}
    slurm_stats_files = []
    if not args.no_slurm:
        if args.slurm_stats_dir == 'LOGS':

            for subdir in args.subdirs:
                for fname in os.listdir(os.path.join(args.grid_dir, subdir)):
                    if fname.endswith('.stats'):
                        slurm_stats_files.append((os.path.join(args.grid_dir, subdir, fname), subdir))
        else:
            for fname in os.listdir(os.path.join(args.grid_dir, args.slurm_stats_dir)):
                if fname.endswith('.stats'):
                    slurm_stats_files.append((os.path.join(args.grid_dir, args.slurm_stats_dir, fname), None))
            # slurm_stats_files = [os.path.join(args.grid_dir, args.slurm_stats_dir, _) )
    else:
        for subdir in args.subdirs:
            slurm_stats_files.append([None, subdir])
    for i, (slurm_stats_file, subdir) in enumerate(slurm_stats_files):
        if args.no_slurm:
            lines = None
        else:
            with open(slurm_stats_file, 'r') as handle:
                lines = handle.readlines()
        _ = get_slurm_stats(lines, args)
        jobid = _['jobid']
        taskjob = _['taskjob']
        taskid = _['taskid']
        if taskid == '':
            taskid = f'{i:>04}'

        if subdir is None:
            subdir = f'{taskid:>04}'

        if subdir in run_info:
            if int(run_info[subdir]['jobid']) < int(jobid):  # Take more recent run.
                run_info[subdir] = _
        else:
            run_info[subdir] = _

        jobid2task[jobid] = (taskjob, taskid)
        if taskid not in task2jobid:
            task2jobid[taskid] = jobid
        else:
            print(f'Key `{taskid}` already in `task2jobid` dictionary with value {task2jobid[taskid]} '
                  f'when inserting value {jobid}.')

    for subdir in args.subdirs:
        hist_path = os.path.join(args.grid_dir, subdir, f'LOGS/{args.history_file.format(subdir, subdir)}')
        mesa_termcode = get_mesa_termcode(subdir, args)[1]
        if os.path.exists(hist_path) and (mesa_termcode != 'NotRun'):
            data = read_hist_first_last_row(hist_path)
            run_info[subdir]['mesa_termcode'] = mesa_termcode
            run_info[subdir]['mesa_last_model'] = data.model_number[-1]
            # noinspection PyTypeChecker
            run_info[subdir]['CHeX'] = ['pre-CHeX', 'post-CHeX'][int(data.center_he4[-1] < 1e-6)]
            run_info[subdir]['CHX'] = ['pre-CHX', 'post-CHX'][int(data.center_h1[-1] < 1e-6)]
            _ = run_info[subdir]['walltime_required'].split(':')
            run_info[subdir]['short_runtime'] = ['', 'short_runtime'][
                int((3600 * int(_[0]) + 60 * int(_[1]) + int(_[0])) < 60)]

            run_data[subdir] = data
        else:
            run_info[subdir]['mesa_termcode'] = mesa_termcode
            run_info[subdir]['mesa_last_model'] = '-1'
            # noinspection PyTypeChecker
            run_info[subdir]['CHeX'] = mesa_termcode
            run_info[subdir]['CHX'] = mesa_termcode
            run_info[subdir]['short_runtime'] = mesa_termcode
            run_data[subdir] = mesa_termcode

    cnt_term_codes_mesa, dct_term_codes_mesa = count_in_dict(run_info, key='mesa_termcode')

    # Make output string.
    part_len = 44
    string_separator = '-' * part_len + '\n'
    status_str = '' + string_separator
    status_str += f'{"  termination_code":<32}{"count":>10}\n'
    status_str += string_separator
    for (mesa_tc, c_mesa_tc) in cnt_term_codes_mesa.items():
        part = f'{mesa_tc:<32}{c_mesa_tc:>12}\n'
        status_str += part
        if mesa_tc in ['None', 'min_timestep_limit']:
            cnt_term_codes_slurm, dct_term_codes_slurm = count_in_dict(run_info, key='reason',
                                                                       filter_mainkeys=dct_term_codes_mesa[mesa_tc])
            for i, (slurm_tc, c_slurm_tc) in enumerate(cnt_term_codes_slurm.items()):
                part = f'{slurm_tc:<29}{c_slurm_tc:>9}\n'
                start_part = ' ├ '
                if i == len(cnt_term_codes_slurm) - 1:
                    start_part = ' └ '
                part = start_part + part
                status_str += part

                # For each sub section count pre CHeX and post CHeX
                cnt_, dct_ = \
                    count_in_dict(run_info, key='CHeX', filter_mainkeys=dct_term_codes_slurm[slurm_tc])

                for j, key in enumerate(sorted(cnt_)):
                    part = f'{key:<26}{cnt_[key]:>6}\n'
                    start_part = '    ├ '
                    if j == len(cnt_) - 1:
                        start_part = '    └ '
                    part = start_part + part
                    status_str += part

                    # For each pre-CHeX section count pre CHX and post CHX
                    cnt_CHX, dct_CHX = \
                        count_in_dict(run_info, key='CHX', filter_mainkeys=dct_['pre-CHeX'])

                    for j, key in enumerate(sorted(cnt_CHX)):
                        part = f'{key:<23}{cnt_CHX[key]:>6}\n'
                        start_part = '       ├ '
                        if j == len(cnt_CHX) - 1:
                            start_part = '       └ '
                        part = start_part + part
                        status_str += part

    status_str += string_separator
    cnt, dct = count_in_dict(run_info, 'short_runtime')
    if not args.no_slurm:
        status_str += f'{"short_runtime":<32} {str(cnt["short_runtime"]):>12}\n'
        status_str += string_separator
    if args.list_all:
        for subdir in args.subdirs:
            status_str += f'{subdir:<16} {run_info[subdir]["CHeX"]:>10} {run_info[subdir]["mesa_last_model"]:>7} {run_info[subdir]["mesa_termcode"]:<12}\n'
        status_str += string_separator
    print(status_str)

    if args.make_restart_file:
        restart_reason = {}  # last photo, redoRC, etc
        for subdir in args.subdirs:
            info = run_info[subdir]
            data = run_data[subdir]
            reason = ''
            photo = ''
            mesa_termcode = info['mesa_termcode']

            # print(subdir, info['CHeX'], info['mesa_termcode'])
            if info['CHeX'] == 'post-CHeX':
                reason = 'redoRC'
                photo = 'preRC'
            elif mesa_termcode == 'Running':
                reason = 'Running'
                photo = 'last'
            elif info['CHeX'] == 'pre-CHeX':
                reason = 'None'
                photo = 'last'
            elif args.make_restart_file == 'redoRC':
                reason = 'redoRC'
                photo = 'preRC'
            elif mesa_termcode in (['logQ_min_limit', 'FileNotFound', 'NotRun']):
                reason = info['mesa_termcode']
                photo = 'full_restart'

            restart_reason[subdir] = (photo, reason)

        grid_restart_photos = []
        for subdir, (photo, reason) in restart_reason.items():
            photodir = os.path.join(args.grid_dir, subdir, 'photos')
            if os.path.exists(photodir):
                photos = sorted(os.listdir(photodir))
            else:
                photos = []
            photo_modelnum = np.array([int(_.replace('x', '')) for _ in photos])

            if args.make_restart_file == 'pre-CHeX':
                info = run_info[subdir]
                if info['CHeX'] == 'post-CHeX':
                    continue

            restart_photo = ''
            try:
                if photo == 'last':
                    restart_photo = photos[np.where((photo_modelnum[-1] - photo_modelnum) > 200)[0][-1]]
                elif photo == 'preRC':
                    histpath = os.path.join(args.grid_dir, subdir, f'LOGS/{args.history_file.format(subdir, subdir)}')
                    if os.path.exists(histpath):
                        hist = ld.History(histpath)
                        cheb_mask = uf.get_cheb_mask(hist)
                        first_model_rc = hist.data.model_number[cheb_mask][0]
                        restart_photo = photos[np.where((first_model_rc - photo_modelnum) > 200)[0][-1]]
                    else:
                        restart_photo = 'full_restart'
                elif photo == 'full_restart':
                    restart_photo = 'full_restart'
                elif photo == '':
                    restart_photo = 'full_restart'
            except IndexError as e:
                print(subdir, photo, reason, e)
                restart_photo = 'full_restart'
            if args.verbose:
                print(subdir, restart_photo)

            if restart_photo == '':
                print(f'Empty restart_photo for {subdir}!')
            grid_restart_photos.append((subdir, restart_photo))

        with open(os.path.join(args.grid_dir, 'grid_restart'), 'w') as f:
            f.writelines((' '.join(line) + '\n' for line in grid_restart_photos))
    return 0