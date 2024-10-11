#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import itertools
import shutil
import f90nml
import importlib.util

import numpy as np
import make_merger_profiles as mk_mp
import make_core_env as mce

main_header = """! this is the master inlist that MESA reads when it starts.

! This file tells MESA to go look elsewhere for its configuration
! info. This makes changing between different inlists easier, by
! allowing you to easily change the name of the file that gets read."""

file_header = "! Created with a python script.\n"

non_mesa_keys = ['type', 'filename', 'unpacknumber', 'unpacknumber_total', 'use_extra_default_inlist', 'group_unpack',
                 'finalize_funcs', 'note', 'run_in_each_dir']

non_mesa_key_start = '!PY_KEY_'

def parse_value(value):
    """
    Takes `value` and makes it compatible for use in an inlist.
    """

    if type(value) == bool:
        parsed = '.{}.'.format(str(value).lower())
    elif type(value) == str:
        parsed = "'{}'".format(value)
    else:
        parsed = value
    return parsed


def generate_inlist_string(inlist_dict):
    """
    Generates an inlist string from `inlist_dict`. Which is an unpacked controls etc dict.
    """
    inlist_string = ""

    inlist_type = inlist_dict['type']
    if inlist_type == 'master':
        inlist_string += main_header + "\n\n"

        if 'use_extra_default_inlist' in inlist_dict:
            extra_inlists = inlist_dict['use_extra_default_inlist']
            if len(extra_inlists) > 4:
                raise ValueError("Too many extra inlists")
        else:
            extra_inlists = []

        for key, key_dat in inlist_dict.items():
            if (key in non_mesa_keys) or key.startswith(non_mesa_key_start):
                continue
            elif key.startswith('#'):
                continue
            sub_str = "&{}\n\n".format(key)

            if len(extra_inlists) > 0:
                for i, extra_inlist in enumerate(extra_inlists):
                    parsed_extra_inlist = parse_value(extra_inlist)
                    sub_str += "    read_extra_{}_inlist{} = .true.\n".format(key, i + 1)
                    sub_str += "    extra_{}_inlist{}_name = {}\n\n".format(key, i + 1, parsed_extra_inlist)

            for sub_key, sub_value in key_dat.items():
                parsed_sub_value = parse_value(sub_value)
                sub_str += '    {} = {}\n'.format(sub_key, parsed_sub_value)

            sub_str += '\n'
            sub_str += r'/ ! end of {} namelist'.format(key)
            sub_str += '\n\n\n'

            inlist_string += sub_str

    else:
        sub_str = "&{}\n\n".format(inlist_type)
        for key, value in inlist_dict.items():
            if (key in non_mesa_keys) or key.startswith(non_mesa_key_start):
                if key == 'note':
                    sub_str = '! {}\n'.format(value) + sub_str
                    continue
                else:
                    continue
            elif key.startswith('#'):
                continue
            parsed_value = parse_value(value)
            sub_str += '    {} = {}\n'.format(key, parsed_value)

        sub_str += '\n'
        sub_str += r'/ ! end of {} namelist'.format(inlist_type)

        inlist_string += sub_str

    return inlist_string


def write_to_file(dir_path, filename, inlist_string):
    """
    Write `inlist_string` to `dir_path`/`filename`.
    """
    inlist_string = file_header + inlist_string
    inlist_string = inlist_string.strip() + '\n'

    with open(os.path.join(dir_path, filename), 'w') as handle:
        handle.write(inlist_string)


def make_inlists(dir_path, inlist_options_path, coldef_dir):
    """
    Read `inlist_options_path` and generate all inlist strings and write them to files in `dir_path`.
    If the files already exist, overwrite them.
    If the directory `dir_path` does not exist, create it.
    """

    dir_path = os.path.expanduser(dir_path)
    inlist_options_path = os.path.expanduser(inlist_options_path)

    # Force reload of inlist_options
    spec = importlib.util.spec_from_file_location("inlist_options", inlist_options_path)
    inlist_options = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(inlist_options)

    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
    else:
        print("Deleting {}".format(dir_path))
        shutil.rmtree(dir_path)
        os.makedirs(dir_path)

    if inlist_options.star_job['filename'] == inlist_options.controls['filename']:
        star_job_controls_same_file = True
    else:
        star_job_controls_same_file = False

    write_to_file(dir_path, inlist_options.master_inlist['filename'],
                  generate_inlist_string(inlist_options.master_inlist))
    write_to_file(dir_path, inlist_options.pgstar['filename'], generate_inlist_string(inlist_options.pgstar))

    gen_star_job = unpack_inlist(inlist_options.star_job)
    gen_controls = unpack_inlist(inlist_options.controls)

    unpacked = []

    for i, (unpacked_star_job, unpacked_controls) in enumerate(itertools.product(gen_star_job, gen_controls)):
        if 'finalize_funcs' in list(inlist_options.master_inlist.keys()):
            for key, func in inlist_options.master_inlist['finalize_funcs'].items():
                if key == 'controls':
                    unpacked_controls = func(unpacked_controls, unpacked_star_job, key, i)
                elif key == 'star_job':
                    unpacked_star_job = func(unpacked_controls, unpacked_star_job, key, i)

        unpacked.append((unpacked_star_job, unpacked_controls))
        star_job_string = generate_inlist_string(unpacked_star_job)
        controls_string = generate_inlist_string(unpacked_controls)

        file_num_str = '_{:04}'.format(i)

        if star_job_controls_same_file:
            inlist_string = star_job_string + '\n\n\n' + controls_string
            write_to_file(dir_path, inlist_options.star_job['filename'] + file_num_str, inlist_string)
        else:
            write_to_file(dir_path, inlist_options.star_job['filename'] + file_num_str, star_job_string)
            write_to_file(dir_path, inlist_options.controls['filename'] + file_num_str, controls_string)

    shutil.copy2(inlist_options_path, dir_path)

    # columns for output files

    master_inlist = f90nml.read(os.path.join(dir_path, inlist_options.master_inlist['filename']))
    starjob_to_read = []
    for key, value in master_inlist['star_job'].items():
        if key.startswith('extra_star_job_inlist'):
            starjob_to_read.append((key, value))
    starjob_to_read = sorted(starjob_to_read, key=lambda x: x[0])  # sort by order which MESA reads the inlists

    histcols = []
    starcols = []
    for _, starjob_file in starjob_to_read:
        success = False
        for fname in [starjob_file, starjob_file+'_0000']:
            for search_dir in [dir_path, coldef_dir]:
                if os.path.isfile(os.path.join(search_dir, starjob_file)):
                    starjob = f90nml.read(os.path.join(search_dir, starjob_file))
                    histcols.append(starjob['star_job'].get('history_columns_file'))
                    starcols.append(starjob['star_job'].get('profile_columns_file'))
                    success = True
                    break
        if success:
            break
        if not success:
            raise FileNotFoundError(f"Can't find file {starjob_file} or {starjob_file}_0000 in directories\n"
                                    f"{dir_path}\n"
                                    f"{coldef_dir}")
    coldef_files = []
    for fname in histcols[::-1]:
        if fname is not None:
            coldef_files.append(fname)
            break
    for fname in starcols[::-1]:
        if fname is not None:
            coldef_files.append(fname)
            break
    for fname in coldef_files:
        shutil.copy2(os.path.join(coldef_dir, fname), os.path.join(dir_path, fname))

    return unpacked, coldef_files


def unpack_inlist(inlist_dict):
    """
    If an inlist contains any lists, tuples, or numpy arrays of length greater than 1
    then return all unique inlists.
    """

    if inlist_dict['type'] == 'master':
        raise ValueError("Cannot unpack master inlist")
    if inlist_dict['type'] == 'pgstar':
        raise ValueError("Cannot unpack pgstar inlist")

    inlist_dict['unpacknumber'] = 0
    inlist_dict['unpacknumber_total'] = 1

    contains_list = []  # Keys which need to be unpacked
    lengths = []  # Length of lists which need to be unpacked
    items = []  # Lists which need to be unpacked
    for key, item in inlist_dict.items():
        if (key in non_mesa_keys) or key.startswith(non_mesa_key_start):
            if key == 'group_unpack':
                groups = inlist_dict['group_unpack']
                for i, group in enumerate(groups):
                    contains_list.append('__group__{}'.format(i))
                    lengths.append(len(group))
                    items.append(group)
            elif f'{non_mesa_key_start}_to_unpack' in inlist_dict.keys():
                if key in inlist_dict[f'{non_mesa_key_start}_to_unpack']:
                    contains_list.append(key)
                    lengths.append(len(item))
                    items.append(inlist_dict[key])
            continue
        if type(item) in [list, tuple, np.ndarray] and len(item) > 1:
            contains_list.append(key)
            lengths.append(len(item))
            items.append(inlist_dict[key])

    if len(lengths) == 0:
        yield inlist_dict
        return

    tot_permutations = 1
    for n in lengths:
        tot_permutations *= n

    inlist_dict['unpacknumber_total'] = tot_permutations
    base_inlist = inlist_dict.copy()

    permutations = itertools.product(*items)
    for i, permutation in enumerate(permutations):
        new_inlist = base_inlist.copy()
        for j, value in enumerate(permutation):
            if contains_list[j].startswith('__group__'):
                new_inlist.update(value)
            else:
                new_inlist[contains_list[j]] = value
        new_inlist['unpacknumber'] = i
        yield new_inlist


def check_environ_var():
    """
    Check the required environment variables.
    """
    try:
        mesa_dir = os.environ["MESA_DIR"]
    except KeyError:
        raise ValueError("$MESA_DIR not set")

    # try:
    #    mesasdk_dir = os.environ["MESASDK_ROOT"]
    # except KeyError:
    #    raise ValueError("$MESASDK_ROOT not set")

    try:
        base_work_dir = os.environ["BASE_WORK_DIR"]
    except KeyError:
        raise ValueError("$BASE_WORK_DIR not set")

    # try:
    #    work_dir = os.environ["WORK_DIR"]
    # except KeyError:
    #    raise ValueError("$WORK_DIR not set")


def setup_dirs(unpacked, inlist_dir, sub_work_dir, master_work_dir='', copy_exe=False, walltime=12, max_job_count=4,
               nodes=1, inlist_options_filename='inlist_options.py', defaults_dir='', colddef_files=[]):
    """
    Prepare directories where mesa will run.
    The directory `inlist_dir` contains all required inlists and the inlist_options.py file from which they were generated.
    The directory `sub_work_dir` will be created (or cleaned) in $WORK_DIR/`sub_work_dir`.
    If `master_work_dir` is empty, use the $WORK_DIR environment variable.
    """
    inlist_dir = os.path.expanduser(inlist_dir)
    master_work_dir = os.path.expanduser(master_work_dir)



    if sub_work_dir == '':
        raise ValueError("Must give sub_work_dir a non-empty string")

    base_work_dir = defaults_dir
    
    if master_work_dir == '':
        master_work_dir = os.environ["WORK_DIR"]
    work_dir = os.path.join(master_work_dir, sub_work_dir)
    
    if defaults_dir == '':
        check_environ_var()
        mesa_dir = os.environ["MESA_DIR"]
        defaults_dir = os.path.join(mesa_dir, 'star')

    if work_dir == '/':
        raise ValueError("work_dir cannot be root directory")

    if os.path.isdir(work_dir):
        shutil.rmtree(work_dir)
        print("Deleting {}".format(work_dir))
        os.makedirs(work_dir)
    else:
        os.makedirs(work_dir)

    # Force reload of inlist_options
    spec = importlib.util.spec_from_file_location("inlist_options", os.path.join(inlist_dir, inlist_options_filename))
    inlist_options = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(inlist_options)
    
    if inlist_options.star_job['filename'] == inlist_options.controls['filename']:
        star_job_controls_same_file = True
        sjc_fname = inlist_options.star_job['filename']
    else:
        star_job_controls_same_file = False
        star_job_fname = inlist_options.star_job['filename']
        controls_fname = inlist_options.controls['filename']

    if 'use_extra_default_inlist' not in list(inlist_options.master_inlist.keys()):
        inlist_options.master_inlist['use_extra_default_inlist'] = []

    # Get number of runs
    inlist_files = os.listdir(inlist_dir)
    inlist_files.remove(inlist_options_filename)
    inlist_files.remove('inlist')
    inlist_files.remove('inlist_pgstar')
    if inlist_options_filename + 'c' in inlist_files:
        inlist_files.remove(inlist_options_filename + 'c')
    if '__pycache__' in inlist_files:
        inlist_files.remove('__pycache__')
    
    num_runs = -1
    for fname in inlist_files:
        if not fname.startswith('inlist'):
            continue
        elif fname.endswith('.py') or fname.endswith('.pyc'):
            continue
        val = int(fname.split('_')[-1]) + 1
        if val > num_runs:
            num_runs = val

    shutil.copy2(os.path.join(inlist_dir, inlist_options_filename), os.path.join(work_dir, inlist_options_filename))
    make_jobfile_grid(master_work_dir, sub_work_dir, walltime, num_runs, max_job_count, nodes)

    for i in range(num_runs):
        dir_num = '{:04}'.format(i)

        if not os.path.isdir(os.path.join(work_dir, dir_num)):
            os.mkdir(os.path.join(work_dir, dir_num))
            os.mkdir(os.path.join(work_dir, dir_num, 'LOGS'))
            os.mkdir(os.path.join(work_dir, dir_num, 'photos'))
            
        
        shutil.copy2(os.path.join(base_work_dir, 'rn'), os.path.join(work_dir, dir_num, 'rn'))
        shutil.copy2(os.path.join(base_work_dir, 're'), os.path.join(work_dir, dir_num, 're'))
        shutil.copy2(os.path.join(base_work_dir, 'clean'), os.path.join(work_dir, dir_num, 'clean'))
        shutil.copy2(os.path.join(base_work_dir, 'mk'), os.path.join(work_dir, dir_num, 'mk'))
        shutil.copytree(os.path.join(base_work_dir, 'make'), os.path.join(work_dir, dir_num, 'make'))
        shutil.copytree(os.path.join(base_work_dir, 'src'), os.path.join(work_dir, dir_num, 'src'))
        if copy_exe:
            shutil.copy2(os.path.join(base_work_dir, 'star'), os.path.join(work_dir, dir_num, 'star'))

        shutil.copy2(os.path.join(inlist_dir, 'inlist'), os.path.join(work_dir, dir_num, 'inlist'))
        shutil.copy2(os.path.join(inlist_dir, 'inlist_pgstar'), os.path.join(work_dir, dir_num, 'inlist_pgstar'))
        
        if f'{non_mesa_key_start}_extra_file' in inlist_options.master_inlist.keys():
            if isinstance(inlist_options.master_inlist[f'{non_mesa_key_start}_extra_file'], str):
                extra_fname = inlist_options.master_inlist[f'{non_mesa_key_start}_extra_file']
                shutil.copy2(os.path.join(base_work_dir, extra_fname), os.path.join(work_dir, dir_num, extra_fname))
            else:
                for extra_fname in inlist_options.master_inlist[f'{non_mesa_key_start}_extra_file']:
                    if extra_fname.startswith('/'):
                        extra_fname_path = extra_fname
                        extra_fname = os.path.split(extra_fname)[-1]
                    else:
                        extra_fname_path = os.path.join(base_work_dir, extra_fname)
                    # if os.path.isfile(os.path.join(base_work_dir, extra_fname)):
                    #     extra_fname_path = os.path.join(base_work_dir, extra_fname)
                    # elif os.path.isfile(extra_fname):
                    #     extra_fname_path = extra_fname
                    #     extra_fname = os.path.split(extra_fname)[-1]
                    # else:
                    #     raise FileNotFoundError(f'Cannot find {extra_fname}')
                    shutil.copy2(extra_fname_path, os.path.join(work_dir, dir_num, extra_fname))

        for fname in colddef_files:
            if fname is not None:
                shutil.copy2(os.path.join(inlist_dir, fname),
                             os.path.join(work_dir, dir_num, fname))

        copied = []
        if len(inlist_options.master_inlist['use_extra_default_inlist']) > 0:
            for extra_inlist in inlist_options.master_inlist['use_extra_default_inlist']:
                shutil.copy2(os.path.join(defaults_dir, extra_inlist),
                             os.path.join(work_dir, dir_num, extra_inlist))
                shutil.copy2(os.path.join(defaults_dir, extra_inlist),
                             os.path.join(inlist_dir, extra_inlist))
                copied.append(extra_inlist)

        for j in range(1, 5):  # xxx_inlist5 is reserved for the inlist generated by this script
            for inlist_type in ['controls', 'star_job', 'pgstar']:
                key = 'read_extra_{}_inlist{}'.format(inlist_type, j)
                if key in list(inlist_options.master_inlist[inlist_type].keys()):
                    if inlist_options.master_inlist[inlist_type][key]:
                        inlist_key = 'extra_{}_inlist{}_name'.format(inlist_type, j)
                        extra_inlist = inlist_options.master_inlist[inlist_type][inlist_key]
                        if not extra_inlist in copied:
                            shutil.copy2(os.path.join(defaults_dir, extra_inlist),
                                         os.path.join(work_dir, dir_num, extra_inlist))
                            shutil.copy2(os.path.join(defaults_dir, extra_inlist),
                                         os.path.join(inlist_dir, extra_inlist))
                            copied.append(extra_inlist)

        if star_job_controls_same_file:
            sjc_fname_num = sjc_fname + '_{:04}'.format(i)
            shutil.copy2(os.path.join(inlist_dir, sjc_fname_num), os.path.join(work_dir, dir_num, sjc_fname))
        else:
            star_job_fname_num = '_{:04}'.format(i)
            controls_fname_num = '_{:04}'.format(i)
            shutil.copy2(os.path.join(inlist_dir, star_job_fname + star_job_fname_num),
                         os.path.join(work_dir, dir_num, star_job_fname))
            shutil.copy2(os.path.join(inlist_dir, controls_fname + controls_fname_num),
                         os.path.join(work_dir, dir_num, controls_fname))

        make_jobfile_grid(master_work_dir, sub_work_dir, walltime, 1, 1, 1, dir_num)  # For single runs

        unpacked_star_job, unpacked_controls = unpacked[i]
        if 'relax_composition_filename' in list(unpacked_star_job.keys()):
            relax_composition_filename = unpacked_star_job['relax_composition_filename']

            fname_split = relax_composition_filename.replace('.data', '').split('_')
            mass_frac_Hecore = float(fname_split[1].replace('.data', ''))

            if f'{non_mesa_key_start}_mean_core_fracs' in list(unpacked_star_job.keys()):
                core_comp = unpacked_star_job[f'{non_mesa_key_start}_mean_core_fracs']
                env_comp = unpacked_star_job[f'{non_mesa_key_start}_mean_env_fracs']
                smoothing_window = unpacked_star_job[f'{non_mesa_key_start}_smoothing_window']
                smoothing = unpacked_star_job[f'{non_mesa_key_start}_smoothing']
                comp_func = unpacked_star_job[f'{non_mesa_key_start}_comp_func']
                comps = comp_func(core_comp, env_comp, mass_frac_Hecore + 3*smoothing_window,
                                  smoothing, smoothing_window, nz=4096)
    
                fpath = os.path.join(master_work_dir, sub_work_dir, '{:04}'.format(i), relax_composition_filename)
                mk_mp.write_comp_file(fpath, comps)
            else:
                if len(fname_split) > 2:
                    smoothing = float(fname_split[2].replace('smooth', ''))
                else:
                    smoothing = 0
    
                try:
                    mass_frac_y = unpacked_controls['initial_y']
                except KeyError:
                    master_inlist = f90nml.read(os.path.join(work_dir, dir_num, 'inlist'))
                    mass_frac_y = 0.28
        
                    for j, extra_inlist in enumerate(inlist_options.master_inlist['use_extra_default_inlist']):
                        if master_inlist['controls']['extra_controls_inlist{}_name'.format(j + 1)]:
                            try:
                                mass_frac_y = f90nml.read(os.path.join(work_dir, dir_num, extra_inlist))['controls'][
                                    'initial_y']
                            except KeyError:
                                pass
                    # if mass_frac_y is None:
                    #     raise ValueError('Cannot find `initial_y` in any supplied inlist.')
                try:
                    mass_frac_z = unpacked_controls['initial_z']
                except KeyError:
                    master_inlist = f90nml.read(os.path.join(work_dir, dir_num, 'inlist'))
                    mass_frac_z = 0.02
        
                    for j, extra_inlist in enumerate(inlist_options.master_inlist['use_extra_default_inlist']):
                        if master_inlist['controls']['extra_controls_inlist{}_name'.format(j + 1)]:
                            try:
                                mass_frac_z = f90nml.read(os.path.join(work_dir, dir_num, extra_inlist))['controls'][
                                    'initial_z']
                            except KeyError:
                                pass
                    # if mass_frac_z is None:
                    #     raise ValueError('Cannot find `initial_z` in any supplied inlist.')
    
                if 'mesh_delta_coeff' in unpacked_controls:
                    num_zones = int(820 / unpacked_controls['mesh_delta_coeff'])
                else:
                    num_zones = 4096
                    
                if f'{non_mesa_key_start}_merger_profile_dat_name' in unpacked_star_job:
                    nuc_net_for_profile = unpacked_star_job[f'{non_mesa_key_start}_merger_profile_dat_name']
                else:
                    try:
                        nuc_net_for_profile = unpacked_star_job['new_net_name']
                    except KeyError:
                        master_inlist = f90nml.read(os.path.join(work_dir, dir_num, 'inlist'))
                        nuc_net_for_profile = None
            
                        for j, extra_inlist in enumerate(inlist_options.master_inlist['use_extra_default_inlist']):
                            if master_inlist['star_job']['extra_star_job_inlist{}_name'.format(j + 1)]:
                                nuc_net_for_profile = f90nml.read(os.path.join(work_dir, dir_num, extra_inlist))['star_job'][
                                    'new_net_name']
                        if nuc_net_for_profile is None:
                            raise ValueError('Cannot find `new_net_name` in any supplied inlist.')
    
                fpath = os.path.join(master_work_dir, sub_work_dir, '{:04}'.format(i), relax_composition_filename)
                mk_mp.make_composition(fpath, mass_frac_Hecore, num_zones, mass_frac_y, mass_frac_z, nuc_net_for_profile,
                                       smooth=smoothing)
            if 'relax_entropy_filename' in list(unpacked_star_job.keys()):
                qDT = mce.make_DT_entropy(mass_frac_Hecore)
                qDT_string = '{}\n'.format(comps.shape[0])
                for line in qDT:
                    qDT_string += ' '.join(['{:.14e}'.format(_) for _ in line]) + '\n'

                relax_entropy_filename = unpacked_star_job['relax_entropy_filename']
                fpath = os.path.join(master_work_dir, sub_work_dir, '{:04}'.format(i), relax_entropy_filename)
                
                if not os.path.isdir(os.path.split(fpath)[0]):
                    os.makedirs(os.path.split(fpath)[0])
                with open(fpath, 'w') as handle:
                    handle.write(qDT_string)
        if 'run_in_each_dir' in inlist_options.master_inlist.keys():
            curdir = os.getcwd()
            os.chdir(os.path.join(work_dir, dir_num))
            inlist_options.master_inlist['run_in_each_dir'](work_dir, dir_num, unpacked_controls, unpacked_star_job)
            os.chdir(curdir)
    return inlist_files


def make_jobfile_grid(work_dir, sub_work_dir, walltime, num_jobs, max_job_count=4, nodes=1, dir_num=None):
    """
    """
    # Not using PBS anymore
    return
    gridjob_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'gridjob.pbs'))
    f = open(gridjob_file, 'r')
    lines_base = f.readlines()
    f.close()

    if num_jobs < nodes:
        print("Warning! Number of jobs less than number of nodes, num nodes set to num jobs.")
        nodes = num_jobs
        max_job_count = num_jobs

    if max_job_count > 16 * nodes:
        max_job_count = 16 * nodes
        print("Warning! max_job_count exceeded 16, set to 16 instead.")
    else:
        if max_job_count not in [x * nodes for x in [1, 2, 4, 8]]:
            print("Warning! max_job_count not a power of 2, cores will be wasted.")

    if max_job_count < nodes:
        print("Warning! max_job_count less than number of nodes. Set to number of nodes.")
        max_job_count = nodes

    is_single = not dir_num is None

    if nodes > 1:
        step = num_jobs // nodes
        jobnums = np.arange(num_jobs + 1)

        starts = jobnums[::step][:nodes]
        ends = np.hstack((starts[1:], num_jobs))
    else:
        nodes = 1
        starts = [0]
        ends = [num_jobs]

    for node_num in range(nodes):
        lines = lines_base[:]
        for i, line in enumerate(lines):
            if line.startswith('#PBS -l walltime'):
                num_hrs = int(walltime)
                num_min = int(60 * (walltime % 1))
                lines[i] = '#PBS -l walltime={:02}:{:02}:00'.format(num_hrs, num_min)

            elif line.startswith('#PBS -l nodes='):
                lines[i] = '#PBS -l nodes={}'.format(1)

            elif line.startswith('#PBS -N mesa_grid'):
                if is_single:
                    lines[i] = '#PBS -N ms_{}_{}'.format(sub_work_dir, dir_num)
                else:
                    lines[i] = '#PBS -N mg_{}'.format(sub_work_dir)

            elif line.startswith('SOURCE_DIR='):
                lines[i] = 'SOURCE_DIR="{}"'.format(os.path.abspath(os.path.join(work_dir, sub_work_dir)))

            elif line.startswith('DAT_DIR='):
                if is_single:
                    lines[i] = 'DAT_DIR="/home/$USER/data/single/{}"'.format(sub_work_dir)
                else:
                    lines[i] = 'DAT_DIR="/home/$USER/data/grids/{}"'.format(sub_work_dir)

            elif line.startswith('WORK_DIR='):
                lines[i] = 'WORK_DIR="$TMPDIR/{}"'.format(sub_work_dir)

            elif line.startswith('SINGLE_DIRNUM=""'):
                if is_single:
                    lines[i] = 'SINGLE_DIRNUM="{}"'.format(dir_num)

            elif line.startswith('NUM_START='):
                num_start = starts[node_num]
                lines[i] = 'NUM_START={}'.format(num_start)

            elif line.startswith('NUM_END='):
                num_end = ends[node_num]
                lines[i] = 'NUM_END={}'.format(num_end)

            elif line.startswith('MAX_JOB_COUNT='):
                lines[i] = 'MAX_JOB_COUNT={}'.format(max_job_count)

            elif line.startswith('export OMP_NUM_THREADS='):
                omp_threads = (16 * nodes) // max_job_count
                lines[i] = 'export OMP_NUM_THREADS={}'.format(omp_threads)

        for i, line in enumerate(lines):
            if not line.endswith('\n'):
                lines[i] += '\n'

        if is_single:
            filename = os.path.join(dir_num, 'single_{}_{}.pbs'.format(sub_work_dir, dir_num))
        else:
            if nodes > 1:
                filename = 'gj_{}_{}.pbs'.format(sub_work_dir, node_num)
            else:
                filename = 'gj_{}.pbs'.format(sub_work_dir)

        new_contents = "".join(lines)
        f = open(os.path.join(work_dir, sub_work_dir, filename), 'w')
        f.write(new_contents)
        f.close()


def make_ready(inlist_dir, sub_work_dir, master_work_dir='', inlist_options_path='inlist_options.py',
               coldef_dir='', copy_exe=False, walltime=720, max_job_count=4, nodes=1, defaults_dir=''):
    """
    Make inlists, setup directories, and make job file.
    """
    unpacked, coldef_files = make_inlists(inlist_dir, inlist_options_path, coldef_dir)
    inlist_files = setup_dirs(unpacked, inlist_dir, sub_work_dir, master_work_dir, copy_exe, walltime,
               max_job_count, nodes, inlist_options_filename=os.path.split(inlist_options_path)[1], defaults_dir=defaults_dir, colddef_files=coldef_files)
    return inlist_files

if sys.platform == 'win32':
    os.environ['MESA_DIR'] = r'C:\Users\walter\Documents\GitHub\MasterProject\code'
    os.environ["BASE_WORK_DIR"] = r'C:\Users\walter\Documents\GitHub\MasterProject\testing'

if __name__ == "__main__":
    args = sys.argv[1:]
    args = [_.strip() for _ in args]
    if len(args) < 4:
        print("Order of arguments:")
        print("grid_name, walltime (hrs), max_job_count, nodes, [-i inlist_options_path], [-c column_def_dir_path], "
              "[-d defaults_dir_path]")
        sys.exit(1)

    grid_name = args[0]
    walltime = int(args[1])
    max_job_count = int(args[2])
    nodes = int(args[3])

    inlist_options_path = 'inlist_options.py'
    coldef_dir = ''

    if len(args) > 4:
        extra_args = args[4:]

        if '-i' in extra_args:
            indx = extra_args.index('-i')
            arg_str = extra_args[indx + 1]
            inlist_options_path = os.path.expanduser(arg_str)

        if '-c' in extra_args:
            indx = extra_args.index('-c')
            arg_str = extra_args[indx + 1]
            coldef_dir = os.path.expanduser(arg_str)
            
        if '-d' in extra_args:
            indx = extra_args.index('-d')
            arg_str = extra_args[indx + 1]
            defaults_dir = os.path.expanduser(arg_str)
        else:
            defaults_dir = os.path.join(os.environ['MESA_DIR'], 'star')

    inlist_dir = os.path.expanduser(os.path.join('~/data/inlists/grids', grid_name))
    sub_work_dir = grid_name
    master_work_dir = os.path.expanduser('~/work/grids')

    copy_exe = False
    print('todo: allow for inlist_options files with different names.')
    print('')
    print('------{}-----'.format(grid_name))
    print('inlist_dir = ', inlist_dir)
    print('sub_work_dir  = ', sub_work_dir)
    print('master_work_dir = ', master_work_dir)
    print('inlist_options_path = ', inlist_options_path)
    print('coldef_dir = ', coldef_dir)
    print('defaults_dir = ', defaults_dir)
    print('copy_exe = ', copy_exe)
    print('walltime (hrs) = ', walltime)
    print('max_job_count = ', max_job_count)
    print('nodes = ', nodes)
    print('')

    inlist_files = make_ready(inlist_dir, sub_work_dir, master_work_dir, inlist_options_path, coldef_dir,
               copy_exe, walltime, max_job_count, nodes, defaults_dir)

    inlist_options_filename = os.path.split(inlist_options_path)[1]
    
    # print(inlist_files)
    # inlist_files = os.listdir(inlist_dir)
    # inlist_files.remove(inlist_options_filename)
    # inlist_files.remove('inlist')
    # inlist_files.remove('inlist_pgstar')
    # if inlist_options_filename + 'c' in inlist_files:  # .pyc files'
    #     inlist_files.remove(inlist_options_filename + 'c')
    # print(inlist_files)
    num_runs = -1
    for fname in inlist_files:
        if not fname.startswith('inlist'):
            continue
        val = int(fname.split('_')[-1]) + 1
        if val > num_runs:
            num_runs = val

    print('num_runs = ', num_runs)

    print('')


def fortran2py(inlist_type, options):
    """
    Utility function that allows you to copy/paste inlist options from fortran to python.
    """

    if inlist_type not in ['star_job', 'controls', 'pgstar']:
        raise ValueError('inlist_type must be one of star_job, controls, or pgstar')

    options = options.replace('!', '#')

    options = options.replace('.true.', 'True')
    options = options.replace('.false.', 'False')

    options = options.replace('"', "'")  # use single quotes

    lines = options.split('\n')
    new_lines = []
    for line in lines:
        if '=' in line and not line.strip().startswith('#'):

            var, val = line.split('=', maxsplit=1)
            if '#' in val:
                val, comment = val.split('#', maxsplit=1)
                comment = ' # ' + comment
            else:
                comment = ''

            if "'" in val:
                pass
            else:
                if 'd' in val:
                    val = val.replace('d', 'E')
            new_line = "{}['{}'] = {}{}".format(inlist_type, var.strip(), val.strip(), comment)
        else:
            new_line = line.strip()
        new_lines.append(new_line)

    new_options = '\n'.join(new_lines)
    print(new_options)
