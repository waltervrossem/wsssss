#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pathlib
import shutil
import subprocess
import sys
import time
from argparse import ArgumentParser

import numpy as np
from scipy.integrate import trapezoid as trapz

from wsssss import load_data as ld

np.seterr(all='ignore')

if '__file__' not in globals().keys():  # Otherwise doc generation breaks.
    import wsssss
    __file__ = os.path.join(os.path.dirname(wsssss.__file__), '_bin/gyre_driver/gyre_driver.py')

_version = '0.2.2'
_this_dir = pathlib.Path(__file__).parent

# MESA values
pi = 3.1415926535897932384626433832795028841971693993751
Msun = 1.9892E33
Rsun = 6.9598E10
Lsun = 3.8418E33
Teff_sun = 5777E0
G = 6.67428E-8
sigma_b = 5.670400E-5

# From old GYRE_driver
Dnu_sun = 137.
numax_sun = 3100.
nmode = 20


# noinspection PyPep8Naming
def get_nu_max_dnu_dp(args, fpath, l):
    """Calculate nu_max, Delta_nu, and Delta_P for a model."""

    if args.filetype == 'MESA':
        gp = ld.GyreProfile(fpath)
        header = gp.header
        columns = gp.columns
        data = gp.data

        M_star = header['star_mass'] / Msun
        R_star = header['star_radius']
        L_star = header['star_luminosity']
        Teff_star = (L_star / (4 * pi * R_star ** 2 * sigma_b)) ** 0.25
        R_star = R_star / Rsun

        r = data['radius']

        N = np.sqrt(data['brunt_N2'])

    elif args.filetype == 'FGONG':
        raise NotImplementedError('FGONG files not implemented.')
        # header, data = fgong.load_fgong(fpath, return_comment=False)
        #
        # M_star = header[0] / Msun
        # R_star = header[1] / Rsun
        # Teff_star = header[13]
        #
        # r = data[:, 0]
        # m = np.exp(data[:, 1]) * M_star * Msun
        # gamma1 = data[:, 14]
        # gamma1[gamma1 < 0] = 0
        #
        # N = np.sqrt((G * m / r**3) * gamma1)

    elif args.filetype == 'LOSC':
        with open(fpath, 'r') as handle:
            lines = handle.readlines()

        markers = []
        for i, line in enumerate(lines):
            if line.startswith('%'):
                markers.append(i)

        start_data = markers[-2] + 1
        phot_r_cgs, mass_cgs, G_cgs = map(float, lines[start_data].split())

        R_star = phot_r_cgs / Rsun
        M_star = mass_cgs / Msun

        num_zones = int(lines[start_data + 1].strip())
        zone_dat = []
        for i in range(num_zones):
            zone_start = start_data + 2 + i
            zone_dat.append(tuple(map(int, lines[zone_start].split())))

        num_mesh_points = int(lines[zone_start + 1])
        skip_rows = zone_start + 2
        print(fpath)
        r, m_div_r3, P, rho, gamma1, A_div_r = np.loadtxt(str(fpath), skiprows=skip_rows, usecols=[1, 2, 3, 4, 5, 6],
                                                          comments='%').T
        if len(r) != num_mesh_points:
            raise ValueError(f'Loading of {fpath} failed, not enough meshpoints loaded.')

        m_h = 1.67353276e-24
        k_B = 1.38064852e-16
        mu = 1.25  # Assume not very ionized and approx solar composition.
        Teff_star = (P * mu * m_h / (rho * k_B))[-1]

        N_div_r = np.sqrt(-G_cgs * m_div_r3 * A_div_r)
        N = N_div_r * r

    else:
        raise ValueError('Invalid filetype.')

    N_div_r = N / r
    N_div_r[np.isnan(N_div_r)] = 0
    N_div_r[N_div_r < 0] = 0

    nu_max = numax_sun * M_star / (R_star ** 2 * np.sqrt(Teff_star / Teff_sun))
    Dnu = Dnu_sun * np.sqrt(M_star / R_star ** 3)

    DP = np.abs(trapz(N_div_r, r))
    DP = 2 * pi ** 2 * (1 / DP) / np.sqrt(l * (l + 1))

    if args.verbose:
        print(f'M_star          = {M_star}')
        print(f'R_star          = {R_star}')
        print(f'Teff            = {Teff_star}')
        print(f'nu_max          = {nu_max}')
        print(f'Dnu             = {Dnu}')
        if l > 0:
            print(f'DP              = {DP}')
        print()

    return nu_max, Dnu, DP


def write_gyre_adin(model_name, l, file_type, suffix, save_modes, grid_type, freq_min, freq_max, n_freq, args):
    """Create the gyre inlist."""

    reduced_name = model_name.name
    gyre_adin = args.gyre_adin_template + reduced_name + f'_l{l}' + suffix
    summary_file = summary_path(model_name, l, suffix, args)
    if args.out_dir == '':
        mode_name_base = model_name
    else:
        mode_name_base = pathlib.Path(args.out_dir) / f'{reduced_name}'

    if type(n_freq) in [int, np.int64]:
        single_scan = True
    else:
        single_scan = False

    if args.verbose:
        print(f'gyre_adin       = {gyre_adin}')
        print(f'l               = {l}')
        print(f'summary_file    = {summary_file}')
        print(f'grid_type       = {grid_type}')
        print(f'file_type       = {file_type}')
        if single_scan:
            print(f'freq_min        = {freq_min:.5f}')
            print(f'freq_max        = {freq_max:.5f}')
            print(f'n_freq          = {n_freq}')
        else:
            print(f'freq_min        = {freq_min:}')
            print(f'freq_max        = {freq_max:}')
            print(f'n_freq          = {n_freq}')
        print()

    if args.base_in == '':
        if args.gyre == 'G5':
            base_in = _this_dir / 'INPUT_GYRE_5.2_ad.in'
            base_in_exists = base_in.exists()
        elif args.gyre == 'G4':
            base_in = _this_dir / 'INPUT_GYRE_4.4_ad.in'
            base_in_exists = base_in.exists()
        elif args.gyre == 'G6':
            base_in = _this_dir / 'INPUT_GYRE_6_ad.in'
            base_in_exists = base_in.exists()
        elif args.gyre == 'G7':
            base_in = _this_dir / 'INPUT_GYRE_7_ad.in'
            base_in_exists = base_in.exists()
        elif args.gyre == 'G8':  # Can use the same base inlist
            base_in = _this_dir / 'INPUT_GYRE_8_ad.in'
            base_in_exists = base_in.exists()
        else:
            raise FileNotFoundError(f'No default base inlist associated with gyre {args.gyre}.')
    else:
        base_in = args.base_in
        base_in_exists = base_in.exists()
    if not base_in_exists:
        raise FileNotFoundError('Cannot find a base inlist for gyre.')

    shutil.copy2(base_in, gyre_adin)

    if save_modes:
        if file_type == 'LOSC':
            mode_item_list = 'l,n_pg,n_p,freq,E_norm,x,Gamma_1,prop_type,xi_r,xi_h,dE_dx,K'
        elif file_type == 'FGONG':
            mode_item_list = 'l,n_pg,n_p,freq,E_norm,x,m,p,rho,Gamma_1,prop_type,xi_r,xi_h,dE_dx,K'
        elif file_type == 'MESA':
            mode_item_list = 'M_r,l,n_pg,n_p,freq,E_norm,x,m,p,rho,Gamma_1,prop_type,xi_r,xi_h,dE_dx,K'
        else:
            raise ValueError('Invalid filetype.')
        if args.gyre >= 'G5':
            # Gyre 5 uses capital P for pressure and the normalized rotation kernel changed to unnormalized.
            mode_item_list = mode_item_list.replace(',p,', ',P,').replace(',K', ',dbeta_dx')

        mode_output = (f"   mode_template = '{mode_name_base}_l%l_%J_np%p_ng%g.mgyre'\n"
                       f"   mode_file_format = 'TXT'\n"
                       f"   mode_item_list = '{mode_item_list}'"
                       f"/")
        if args.gyre >= 'G7':
            mode_output = mode_output.replace('mode_', 'detail_').replace('%J', '%ID')
    else:
        mode_output = '/'

    freq_units = ("freq_min_units = 'UHZ'\n"
                  "   freq_max_units = 'UHZ'")
    ad_ = 'ad_'
    nad_output = ('&nad_output\n'
                  '/')
    if args.gyre == 'G4':
        freq_units = "freq_units = 'UHZ'"
        ad_ = ''
        nad_output = ''

    if single_scan:
        scan_str = (f"&scan\n"
                    f"   grid_type = '{grid_type}'\n"
                    f"   {freq_units}\n"
                    f"   freq_min =   {freq_min}\n"
                    f"   freq_max =   {freq_max}\n"
                    f"   n_freq =     {n_freq}\n"
                    f"/")
    else:
        num_scan = len(n_freq)
        scan_str = ''
        for i in range(num_scan):
            freq_min_i = freq_min[i]
            freq_max_i = freq_max[i]
            n_freq_i = n_freq[i]
            scan_str += (f"&scan\n"
                         f"   grid_type = '{grid_type}'\n"
                         f"   {freq_units}\n"
                         f"   freq_min =   {freq_min_i}\n"
                         f"   freq_max =   {freq_max_i}\n"
                         f"   n_freq =     {n_freq_i}\n"
                         f"/\n")

    s = (f"\n"
         f"! Generated by py_gyre_driver {_version}.\n\n"
         f"&model\n"
         f"   model_type = 'EVOL'\n"
         f"   file = '{model_name}'\n"
         f"   file_format = '{file_type}'\n"
         f"/\n"
         f"&{ad_}output\n"
         f"   summary_file = '{summary_file}'\n"
         f"   summary_file_format = 'TXT'\n"
         f"   summary_item_list = '{args.summary_item_list}'\n"
         f"   freq_units = 'UHZ'\n"
         f"{mode_output}\n"
         f"{nad_output}\n"
         f"{scan_str}"
         f"&mode\n"
         f"   l = {l}\n"
         f"/\n")

    with open(gyre_adin, 'a') as handle:
        handle.write(s)

    return


def run_gyre(model_name, l, suffix, args):
    """Run gyre an instance of gyre."""
    reduced_name = model_name.name
    gyre_adin = args.gyre_adin_template + reduced_name + f'_l{l}' + suffix
    summary_file = summary_path(model_name, l, suffix, args)

    gyre_exec = get_gyre(args, True)[0]

    if args.verbose:
        print(os.getcwd())
        print(f'Calling {gyre_exec} {gyre_adin}\n')

    if os.path.exists(summary_file):
        os.remove(summary_file)

    if not args.skip_calc or (args.skip_calc and l == 0):
        if args.source != '':
            output = subprocess.run(f'source {args.source}; {gyre_exec} {gyre_adin}',
                                    shell=True, executable="/bin/bash", stderr=subprocess.PIPE)
        else:
            output = subprocess.run(f'{gyre_exec} {gyre_adin}',
                                    shell=True, executable="/bin/bash", stderr=subprocess.PIPE)

        if not output.stderr == b'':
            if "ASSERT 'k == k_chk' failed at line 303" in output.stderr.decode():
                raise ValueError(f"Gyre 4.4 does not support the version of MESA used to generate {model_name}")

            raise ChildProcessError(f'The following command encountered an error:\n.'
                                    f'{output.args}\n\n'
                                    f'{output.stderr.decode()}\n')
    else:
        output = f'skipped {gyre_exec} {gyre_adin}'

    return output


def get_gyre(args, check, print_warning=False):
    """Get the path to the gyre executable and check if it is the correct version."""
    version = args.gyre
    environ = os.environ

    if f'GYRE_DIR_{version}' in environ:
        path = pathlib.Path(environ[f'GYRE_DIR_{version}'])
    else:
        path = pathlib.Path(environ['GYRE_DIR'])

        if check:  # Check in $GYRE_DIR/src/common/gyre_version.fpp what version of gyre is found.
            version_file = path / 'src' / 'common' / 'gyre_version.fpp'
            if not version_file.exists():  # Different file name for 7.2 and after
                version_file = path / 'src' / 'common' / 'version_m.fypp'

            with open(version_file, 'r') as handle:
                lines = handle.readlines()

            version_str = ''
            for line in lines:
                if 'VERSION = ' in line.upper():
                    line = line.upper()
                    version_str = line.split('VERSION =')[1].strip().replace("'", '').replace('(', '').replace(')', '')
                    break

            if version_str == '':
                raise ValueError(f'Could not find gyre version of $GYRE_DIR in file\n'
                                 f'{version_file}')

            if not version[1] == version_str[0]:
                if args.lenient:
                    if print_warning:
                        print('###################################################################')
                        print(f'Could not find the required gyre version `{version}` in $GYRE_DIR.\n'
                              f'Using gyre version {version_str} instead.')
                        print('###################################################################')
                        print()
                    version = f'G{version_str[0]}'
                else:
                    raise ValueError(f'Could not find the required gyre version `{version}` in $GYRE_DIR.\n'
                                     f'Found gyre version {version_str} in `{path}`.')

    if version <= 'G4':
        path = path / 'bin' / 'gyre_ad'
    else:
        path = path / 'bin' / 'gyre'

    return path, version


def merge_summary_parts(model_name, l, num_scan, keep_files=False):
    """Append all partial runs for a single `l` into one summary file."""

    all_lines = []
    summary_files = []
    set_header = False
    for i in range(num_scan):
        suffix = f'.part{i + 1}of{num_scan}'
        summary_file = summary_path(model_name, l, suffix, args)
        if not summary_file.exists():  # Skip summary files which found no modes.
            continue

        summary_files.append(summary_file)
        with open(summary_file, 'r') as handle:
            lines = handle.readlines()

        if not set_header:
            all_lines += lines
            set_header = True
        else:
            all_lines += lines[6:]

    new_summary_file = summary_path(model_name, l, '', args)

    with open(new_summary_file, 'w') as handle:
        handle.writelines(all_lines)

    if not keep_files:
        for summary_file in summary_files:
            os.remove(summary_file)


def merge_summary_l(model_name, args, keep_files=False):
    """Append all runs for all `l` into one summary file."""

    all_lines = []
    summary_files = []
    set_header = False
    for l in args.ll:
        summary_file = summary_path(model_name, l, '', args)
        if not summary_file.exists():  # Skip summary files which found no modes.
            continue

        summary_files.append(summary_file)
        with open(summary_file, 'r') as handle:
            lines = handle.readlines()

        if not set_header:
            all_lines += lines
            set_header = True
        else:
            all_lines += lines[6:]

    new_summary_file = summary_path(model_name, '', '', args)

    with open(new_summary_file, 'w') as handle:
        handle.writelines(all_lines)

    if not keep_files:
        for summary_file in summary_files:
            os.remove(summary_file)


def summary_path(model_name, l, suffix, args):
    reduced_name = model_name.name

    if args.out_dir == '':
        summary_file = pathlib.Path(f'{model_name}.sgyre_l{l}{suffix}')
    else:
        summary_file = pathlib.Path(args.out_dir) / f'{reduced_name}.sgyre_l{l}{suffix}'

    return summary_file


# noinspection PyPep8Naming
def calc_scan(model_name, l, args):
    nu_max, Dnu, DP = get_nu_max_dnu_dp(args, model_name, l)

    fmid = nu_max
    fsig = (0.66 * nu_max ** 0.88) / 2 / np.sqrt(2 * np.log(2.))  # Mosser 2012a

    fmin = max(1e-4, fmid - 2 * fsig)
    fmax = fmid + 2 * fsig

    n_freqDnu = int(np.ceil((fmax - fmin) / Dnu))

    if l == 0:
        n_freq = 3 * n_freqDnu
        num_scan = 1
    else:
        l0_summary_file = summary_path(model_name, 0, '', args)
        with open(l0_summary_file, 'r') as handle:
            lines = []
            for i in range(6):
                lines.append(handle.readline())
        colnames = lines[5].split()
        i_freq = colnames.index('Re(freq)')
        freqs_l0 = np.loadtxt(l0_summary_file, skiprows=6, usecols=i_freq)
        fmin = freqs_l0[0]
        fmax = freqs_l0[-1]

        n_freqDP = np.ceil((fmax - fmin) / (1e6 / (1e6 / fmin - DP) - fmin)).astype(int)

        if n_freqDnu > n_freqDP:
            n_freqDnu = np.ceil((fmax - fmin) / Dnu)
            n_freq = int(5 * n_freqDnu)
            num_scan = 1

        else:
            fmin = freqs_l0[:-1]
            fmax = freqs_l0[1:]

            if args.pmode:
                # See figure 4 in https://arxiv.org/abs/1108.4777 for origin of 0.13.
                if l % 2 == 0:  # even l
                    fmid = fmax - 0.13 * (fmax - fmin)
                if l % 2 == 1:  # odd l
                    fmid = np.mean([fmin, fmax], axis=0)

                fsig = np.maximum(0.15, np.minimum(0.5 * (fmax - fmin), nmode * (1e6 / (1e6 / fmid - DP) - fmid)))
                fmin = fmid - fsig
                fmax = fmid + fsig

                # Don't search outside l=0 mode frequencies
                fmin = np.maximum(fmin, freqs_l0[0])
                fmax = np.minimum(fmax, freqs_l0[-1])

            n_freq = 5 * np.ceil((fmax - fmin) / (1e6 / (1e6 / fmin - DP) - fmin)).astype(int)

            num_scan = len(fmin)

    return fmin, fmax, n_freq, num_scan, nu_max


def split_scan(fmin, fmax, n_freq, grid_type, args):
    if grid_type == 'LINEAR':

        fmin_new = []
        fmax_new = []
        n_freq_new = []
        for i, n in enumerate(n_freq):
            if n <= args.batch:
                fmin_new.append(fmin[i])
                fmax_new.append(fmax[i])
                n_freq_new.append(n)
            else:
                num_split = n // args.batch + 1
                sub_intervals = np.linspace(fmin[i], fmax[i], num_split + 1)
                sub_fmin = sub_intervals[:-1]
                sub_fmax = sub_intervals[1:]
                sub_n = (int(n / num_split) + 1) * np.ones_like(sub_fmin, dtype=int)

                fmin_new.extend(sub_fmin)
                fmax_new.extend(sub_fmax)
                n_freq_new.extend(sub_n)
        fmin = np.array(fmin_new)
        fmax = np.array(fmax_new)
        n_freq = np.array(n_freq_new, dtype=int)

    else:
        raise NotImplementedError(f'Only `grid_type = LINEAR` is implemented.')

    num_scan = len(n_freq)

    return fmin, fmax, n_freq, num_scan


def do_gyre_sim(fpath, args):
    grid_type = 'LINEAR'
    num_scan = 0
    try:
        for l in args.ll:
            fmin, fmax, n_freq, num_scan, nu_max = calc_scan(fpath, l, args)
            if (l > 0) and (nu_max < args.min_numax):
                print(f'Skipping {fpath} l={l} as numax {nu_max} below min-numax {args.min_numax}.')
                continue

            # Split each frequency scan into a different gyre run. Usually is a bit faster.
            if num_scan > 1 and args.parts:
                # Split into scans upto size `args.batch`. Can be slow if batch is too small.
                if args.batch is not None and any(n_freq > args.batch):
                    fmin, fmax, n_freq, num_scan = split_scan(fmin, fmax, n_freq, grid_type, args)

                for i in range(len(n_freq)):
                    fmin_i = fmin[i]
                    fmax_i = fmax[i]
                    n_freq_i = n_freq[i]

                    part_suffix = f'.part{i + 1}of{num_scan}'

                    write_gyre_adin(fpath, l, args.filetype, part_suffix, args.save_modes,
                                    grid_type, fmin_i, fmax_i, n_freq_i, args)
                    run_gyre(fpath, l, part_suffix, args)

            else:
                write_gyre_adin(fpath, l, args.filetype, '', args.save_modes, grid_type, fmin, fmax, n_freq, args)
                run_gyre(fpath, l, '', args)

            if args.parts and num_scan > 1:
                merge_summary_parts(fpath, l, num_scan)
        if not args.no_merge:
            merge_summary_l(fpath, args)
    except KeyboardInterrupt:
        print('Stopping GYRE')
        if num_scan > 1 and args.parts:
            merge_summary_parts(fpath, l, num_scan, keep_files=True)  # Merge the summaries for the interrupted l.
        merge_summary_l(fpath, args, keep_files=True)
        raise


def sort_files(files):
    path_lengths = np.array([a for a in enumerate(map(lambda x: len(str(x)), files))])
    i_sort = np.argsort(path_lengths[:, 1])
    files = np.asarray(files, dtype=object)[i_sort]
    path_lengths = path_lengths[i_sort]

    lengths = list(set(path_lengths[:, 1]))
    lengths.sort()
    for length in lengths:
        i_sub = np.where(path_lengths[:, 1] == length)[0]
        files[i_sub] = files[i_sub][np.argsort(files[i_sub])]

    files = list(files)
    return files


def check_args(args):
    if args.gyre >= 'G6':
        parts_set = False
        if args.parts:
            parts_set = True
        batch_set = False
        if args.batch is not None:
            batch_set = True

        if parts_set or batch_set:
            print("When using gyre 6 or above --parts and/or --batch are no longer required.")
            print("Turning off --parts and --batch")
            args.parts = False
            args.batch = None

    if args.ll.startswith('mode'):
        args.save_modes = True
        args.ll = args.ll.replace('mode', '')

    args.ll = [int(l) for l in args.ll]
    args.ll.sort()

    if 0 not in args.ll:
        args.ll = [0] + args.ll  # Need l=0 modes to estimate frequency scans.

    if type(args.files) == list:
        args.files = [pathlib.PurePath(fpath.strip()) for fpath in args.files]
    else:
        args.files = [pathlib.PurePath(args.files.strip())]

    if args.skip_existing:
        existing_out_files = list(pathlib.Path(args.out_dir).glob('*.sgyre_l'))
        existing_out_files = [_.absolute() for _ in existing_out_files]
        do_files = [model_name for model_name in args.files if
                    summary_path(model_name, '', '', args).absolute() not in existing_out_files]
        if args.verbose:
            print(f'num_prof_skipped= {len(args.files) - len(do_files)}')
        args.files = do_files

    # Sort files by length, then by filename.
    if args.sort and len(args.files) > 1:
        args.files = sort_files(args.files)

    # Create output directory
    if args.out_dir != '':
        args.out_dir = pathlib.Path(args.out_dir)
        if not args.out_dir.exists():
            os.mkdir(args.out_dir)

    # Create and/or clean input directory
    if args.in_dir != '':
        gyre_adin_template = args.in_dir + '/gyre_ad.in_'
    else:
        args.in_dir = 'gyre_ad.in'
        gyre_adin_template = 'gyre_ad.in/gyre_ad.in_'
    args.in_dir = pathlib.Path(args.in_dir)
    if args.in_dir.exists():
        shutil.rmtree(args.in_dir)
    os.mkdir(args.in_dir)

    if args.base_in != '':
        args.base_in = pathlib.PurePath(args.base_in)

    # If --lenient, switch to the version of gyre found.
    original_gyre = args.gyre
    checked_gyre = get_gyre(args, True, True)[1]
    args.gyre = checked_gyre

    args.gyre_adin_template = gyre_adin_template
    args.original_gyre = original_gyre
    return args


def get_parser():
    parser = ArgumentParser(description="""
    This program will run GYRE, estimating appropriate number of scan points for different angular orders l
    and range of frequency.
    """)

    parser.add_argument('ll', type=str,
                        help='Degrees of modes to scan. eg. `012` or `mode0`. '
                             'Including `mode` overrides `--save-modes`.')
    parser.add_argument('filetype', type=str, choices=['FGONG', 'MESA', 'LOSC'], default='MESA',
                        help='Filetype of profiles that gyre will read.')
    parser.add_argument('files', type=str, nargs='*',
                        help='Paths to profile files for gyre to use.')
    parser.add_argument('--gyre', type=str, choices=['G4', 'G5', 'G6', 'G7', 'G8'], default='G8',
                        help='Which version of gyre to use.')
    parser.add_argument('--pmode', action='store_const', const=True, default=False,
                        help='If set, scan for modes around the expected frequencies.')
    parser.add_argument('--save-modes', action='store_const', const=True, default=False,
                        help='Save mode profiles.')
    parser.add_argument('--verbose', '-v', action='store_const', const=True, default=False,
                        help='Print info about gyre runs.')
    parser.add_argument('--parts', '-p', action='store_const', const=True, default=False,
                        help='Run each frequency scan window individually. This is probably a bit faster.')
    parser.add_argument('--source', type=str, default='',
                        help='Source this file before running gyre.')
    parser.add_argument('--lenient', action='store_const', const=True, default=False,
                        help='If set, will not stop `gyre_driver` if the version of gyre given to `--gyre` is different'
                             ' than the version found in `$GYRE_DIR`.')
    parser.add_argument('--out-dir', type=str, default='gyre_out',
                        help='Directory in which to save output.')
    parser.add_argument('--in-dir', type=str, default='gyre_in',
                        help='Directory in which to save inlist files.')
    parser.add_argument('--base-in', type=str, default='',
                        help='Path to the base inlist to use.')
    parser.add_argument('--no-merge', '-m', action='store_const', const=True, default=False,
                        help="Don't merge the final summary files into a single file.")
    parser.add_argument('--batch', type=int, default=None,
                        help='Split scans into batches of up to N frequencies in order to'
                             ' get around memory issues with broken prune_modes in 5.2.')
    parser.add_argument('--skip-calc', action='store_const', const=True, default=False,
                        help='Do everything except run gyre for l >= 1. Useful to only generate inlists.')
    parser.add_argument('--skip-existing', action='store_const', const=True, default=False,
                        help='Skip running gyre for existing runs. Only works if summary files are merged.')
    parser.add_argument('--min-numax', type=float, default=0,
                        help='Models with numax in uHz lower than this will only calculate l=0 modes.')
    parser.add_argument('--version', action='version', version=f'gyre_driver {_version}')
    parser.add_argument('--summary-item-list', type=str, default='M_star,R_star,L_star,l,n_pg,n_p,n_g,freq,E,E_norm',
                        help='Summary item list for gyre.')
    return parser




if __name__ == '__main__':
    run()

def run():
    t_start = time.time()
    cwd = os.getcwd()

    parser = get_parser()
    args = parser.parse_args()

    args.sort = True
    if len(args.files) > 1:
        args.files = sort_files(args.files)

    if args.filetype == 'LOSC':
        print('Warning! LOSC does not supply an effective temperature, will use an estimate instead.')

    if args.verbose:
        print()
        print('Parsed inputs:')
        print(f'll              = {args.ll}')
        print(f'filetype        = {args.filetype}')
        print(f'files           = {" ".join([str(_) for _ in args.files])}')
        print(f'--gyre          = {args.gyre}')
        print(f'--pmode         = {args.pmode}')
        print(f'--save-modes    = {args.save_modes}')
        print(f'--verbose       = {args.verbose}')
        print(f'--parts         = {args.parts}')
        print(f'--source        = {args.source}')
        print(f'--lenient       = {args.lenient}')
        print(f'--out-dir       = {args.out_dir}')
        print(f'--in-dir        = {args.in_dir}')
        print(f'--base-in       = {args.base_in}')
        print(f'--sort          = {args.sort}')
        print(f'--no-merge      = {args.no_merge}')
        print(f'--batch         = {args.batch}')
        print(f'--skip-calc     = {args.skip_calc}')
        print(f'--skip-existing = {args.skip_existing}')
        print()

    # Load environment variables from --source and keep a copy of the old ones.
    environ_original = os.environ.copy()
    if args.source != '':
        env_out = subprocess.check_output(f"source {args.source}; env -0", shell=True, executable="/bin/bash")
        environ_used = dict(line.decode().partition('=')[::2] for line in env_out.split(b'\0'))
        if '' in environ_used:
            environ_used.pop('')
        os.environ.clear()
        os.environ.update(environ_used)
    environ_used = os.environ.copy()

    args = check_args(args)

    if args.verbose:
        print('Processed inputs:')
        print(f'll              = {args.ll}')
        print(f'files           = {" ".join([str(_) for _ in args.files])}')
        print(f'--save-modes    = {args.save_modes}')
        print(f'--gyre          = {args.gyre}')
        print(f'--out-dir       = {args.out_dir}')
        print(f'--in-dir        = {args.in_dir}')
        print(f'--base-in       = {args.base_in}')
        print()

    for file in args.files:
        do_gyre_sim(file, args)

    t_end = time.time()
    t_taken = t_end - t_start
    print(f"Total time taken: {int(t_taken // 3600)}h{int(t_taken // 60) % 60}m{t_taken % 60 :.2f}s\n")

    if args.original_gyre != args.gyre:
        print('###################################################################')
        print(f'Could not find the required gyre version `{args.original_gyre}` in $GYRE_DIR.\n'
              f'Used gyre version `{args.gyre}` instead.')
        print('###################################################################')
        print()

    # Reset to the old environment variables.
    os.environ.clear()
    os.environ.update(environ_original)
    return 0
