#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

import f90nml
import numpy as np

from ..functions import get_mesa_version, compare_version
from ..constants import post15140 as const

parser = f90nml.Parser()
parser.global_start_index = 1

# Locations of *.defaults files in $MESA_DIR.
defaults = {'star_job': 'star/defaults/star_job.defaults',
            'eos': 'eos/defaults/eos.defaults',
            'kap': 'kap/defaults/kap.defaults',
            'controls': 'star/defaults/controls.defaults',
            'pgstar': 'star/defaults/pgstar.defaults',
            'astero': ['star/astero/defaults/astero_search.defaults', 'astero/defaults/astero_search.defaults']
            }


def inlist_diff(dict1, dict2):
    """
    Compare two inlists and show which items change or remain the same.

    Args:
        dict1 (dict):
        dict2 (dict):

    Returns:
        dict: Items which remain the same are in ``result['same']``,
            and those which changed in ``result['changed']``.
    """
    keys1 = list(dict1.keys())
    keys2 = list(dict2.keys())

    unique_keys = []
    for k in keys1 + keys2:
        if k not in unique_keys:
            unique_keys.append(k)

    result = {'same': {}, 'changed': {}}

    for k in unique_keys:
        v1 = None
        v2 = None

        if k in keys1:
            v1 = dict1[k]
        if k in keys2:
            v2 = dict2[k]

        if isinstance(v1, (f90nml.Namelist, dict)) or isinstance(v2, (f90nml.Namelist, dict)):
            raise ValueError(f'Cannot compare nested dicts for key {k}.')

        if v1 == v2:
            result['same'][k] = v1
        else:
            result['changed'][k] = (v1, v2)

    return result


def evaluate_inlist_str(inlist_str, inlist_dir):
    """
    Fully evaluate `inlist_str` as if it were written in `inlist_dir`, following all ``read_extra_*_inlist`` if
    they are ``.true..``.

    Args:
        inlist_str (str): `str` form of an inlist.
        inlist_dir (str): Reference directory for reading other files.

    Returns:
        dict: Evaluated inlist.

    """
    inlist = parser.reads(inlist_str)
    return _evaluate_inlist(inlist, inlist_dir)


def evaluate_inlist(path):
    """
    Fully evaluate the inlist at `path`, following all ``read_extra_*_inlist`` if they are ``.true..``.
    Args:
        path (str): Path to `inlist`.

    Returns:
        dict: Evaluated inlist.

    """
    inlist = parser.read(path)
    inlist_dir = os.path.dirname(path)
    return _evaluate_inlist(inlist, inlist_dir)


def get_inlist_type(inlist):
    key_type = 'old'
    for kind in defaults.keys():
        if kind not in inlist.keys():
            continue

        for key, item in list(inlist[kind].items()):
            if isinstance(item, list):
                key_type = 'new'
                return key_type
    return key_type

def _evaluate_inlist(inlist, inlist_dir):
    """
    Fully evaluate `inlist` as if it were written in `inlist_dir`, following all ``read_extra_*_inlist`` if
    they are ``.true..``.

    Args:
        inlist (dict): Inlist dictionary.
        inlist_dir (str): Directory which contains `inlist`.

    Returns:
        dict: Evaluated inlist.

    """
    out = {}
    key_type = 'old'
    for kind in defaults.keys():
        if kind not in inlist.keys():
            continue

        for key, item in list(inlist[kind].items()):
            if isinstance(item, list):
                key_type = 'new'
                inlist[kind].pop(key)
                for i in range(len(item)):
                    if item[i] is not None:
                        inlist[kind][f'{key}({i + 1})'] = item[i]

        out[kind] = dict(inlist[kind])

    if key_type == 'old':
        max_extra_inlists = 5  # Hard coded in MESA
        read_extra = 'read_extra_{}_inlist{}'
        read_extra_filename = 'extra_{}_inlist{}_name'
    else:
        max_extra_inlists = const.max_extra_inlists
        read_extra = 'read_extra_{}_inlist({})'
        read_extra_filename = 'extra_{}_inlist_name({})'

    to_read = {}
    for kind in defaults.keys():
        if kind not in inlist.keys():
            continue
        to_read[kind] = []

        for i in range(max_extra_inlists):
            key = read_extra.format(kind, i+1)
            if key in inlist[kind].keys():
                if inlist[kind][key]:
                    to_read[kind].append(inlist[kind][read_extra_filename.format(kind, i+1)])

    inlists = {}
    for kind, fnames in to_read.items():
        inlists[kind] = []
        for fname in fnames:
            sub_inlist = parser.read(os.path.join(inlist_dir, fname))
            sub_inlist = _evaluate_inlist(sub_inlist, inlist_dir)

            if kind in sub_inlist.keys():
                inlists[kind].append(sub_inlist[kind])

    for kind, sub_inlists in inlists.items():
        if kind not in inlist.keys():
            out[kind] = {}
        for sub_inlist in sub_inlists:
            out[kind].update(sub_inlist)

    for kind in out.keys():
        for key, value in list(out[kind].items()):  # Consume whole generator as we need to change the underlying dict.
            if '_inlist' in key:  # Don't want read_extra inlist keys anymore.
                _ = out[kind].pop(key)
    return out


def variable_to_string(value):
    """
    Convert `value` to a fortran-compatible string. Can be a `string`, `bool`, `int`, or `float`.

    Args:
        value (str, bool, int, or float): Value to convert.

    Returns:
        str: Fortran-compatible string.
    """

    if type(value) == bool:
        parsed = f'.{str(value).lower()}.'
    elif type(value) == str:
        parsed = f"'{value}'"
    else:
        if isinstance(value, (int, np.int_)):
            parsed = str(value)
        else:
            parsed = f'{value:g}'.replace('e', 'd').replace('d+', 'd')
            if 'd' not in parsed:
                parsed += 'd0'
    return parsed


def write_inlist(inlist, path, header='', mode='w'):
    """
    Write the `inlist` to `path`. `header` is prepended to the string which is written. `mode` is passed to ``open``.

    Args:
        inlist (dict):
        path (str): Inlist file path.
        header (str):
        mode (str): File mode passed to ``open``.

    """
    if header and not header.startswith('!'):
        header = '!' + header
    s = '' + header

    for name in inlist.keys():
        s += f'&{name} ! start\n'
        for key, value in inlist[name].items():
            s += f'    {key} = {variable_to_string(value)}\n'

        s += f'/ !{name} end\n'

    with open(path, mode) as handle:
        handle.write(s)

def print_dict(to_print):
    """
    Print the dict `to_print` with a key and value on every line.

    Args:
        to_print (dict):
    """
    for k, v in to_print.items():
        print(f'{k}: {v}')


def load_MESA_defaults(mesa_dir):
    mesa_version = get_mesa_version(mesa_dir)
    if compare_version(mesa_version, 'r23.05.1', '<'):
        max_extra_inlists = 5  # Hard coded in MESA
        read_extra = 'read_extra_{}_inlist{}'
        read_extra_filename = 'extra_{}_inlist{}_name'
    else:
        max_extra_inlists = const.max_extra_inlists
        read_extra = 'read_extra_{}_inlist({})'
        read_extra_filename = 'extra_{}_inlist_name({})'

    if compare_version(mesa_version, '15140', '>='):
        astero_index = 1
    else:
        astero_index = 0

    inl_str = ''
    default_keys = get_mesa_defaults(mesa_dir)
    for namelist in default_keys.keys():
        if namelist == 'astero':
            inl_str += (f"&{namelist}\n"
                        f"   {read_extra.format(namelist, 1)} = .true.\n"
                        f"   {read_extra_filename.format(namelist, 1)} = '{mesa_dir}/{defaults['astero'][astero_index]}'\n"
                        f"/\n\n")
        else:
            inl_str += (f"&{namelist}\n"
                        f"   {read_extra.format(namelist, 1)} = .true.\n"
                        f"   {read_extra_filename.format(namelist, 1)} = '{mesa_dir}/{defaults[namelist]}'\n"
                        f"/\n\n")
    print(inl_str)
    inlist_defaults = evaluate_inlist_str(inl_str, '')
    return inlist_defaults

def compare_inlist(path1, path2, show_same=False):
    inlist1 = evaluate_inlist(path1)
    inlist2 = evaluate_inlist(path2)

    star_job = inlist_diff(inlist1['star_job'], inlist2['star_job'])
    controls = inlist_diff(inlist1['controls'], inlist2['controls'])

    print(f'left : {path1}')
    print(f'right: {path2}')
    print('')
    print('####### star_job differences #######')
    print_dict(star_job['changed'])
    if show_same:
        print('####### star_job same #######')
        print_dict(star_job['same'])
    print('')

    print('####### controls differences #######')
    print_dict(controls['changed'])
    if show_same:
        print('####### controls same #######')
        print_dict(controls['same'])

def get_mesa_defaults(mesa_dir):
    """
    Get the available MESA inlist options for the version of MESA installed in `mesa_dir`.

    Args:
        mesa_dir (str): ``$``MESA_DIR`` root directory.

    Returns:
        dict: List of keys for each inlist type (e.g. ``star_job``, ``controls``, etc.).
    """
    version = get_mesa_version(mesa_dir)
    if compare_version(version, '15140', '>='):
        namelists = ('star_job', 'eos', 'kap', 'controls', 'astero', 'pgstar')
        separate_eoskap = True
        astero_index = 1
    else:
        namelists = ('star_job', 'controls', 'pgstar')
        separate_eoskap = False
        astero_index = 0

    if compare_version(version, 'r23.05.1', '>='):
        extra_inlist_as_list = True
    else:
        extra_inlist_as_list = False


    keys_defaults = {}
    for nml_type in namelists:
        if nml_type == 'astero':
            path_to_file = os.path.join(mesa_dir, defaults[nml_type][astero_index])
        else:
            path_to_file = os.path.join(mesa_dir, defaults[nml_type])

        with open(path_to_file, 'r') as handle:
            s_raw = handle.read()

        # Get rid of empty lines and comments
        s = s_raw.lower().splitlines()
        s = [_ for _ in s if not _.strip().startswith('!')]  # Get rid of fortran comments
        s = [_ for _ in s if _.strip()]  # Get rid of ''
        s_no_com = '\n'.join(s)

        keys = [_.split('=', maxsplit=1)[0].strip() for _ in s]
        values = [_.split('=', maxsplit=1)[1].split('!')[0].strip() for _ in s]
        keys_defaults[nml_type] = keys
    return keys_defaults


def check_inlist(path, mesa_dir):
    """
    Check whether the options in the inlist at `path` are available in the version of mesa specified in `mesa_dir`.
    Will try to detect if the options have been moved to ``eos`` or ``kap`` from ``star_job`` or ``controls``.

    Args:
        path (str): Path to an inlist.
        mesa_dir (str): ``$``MESA_DIR`` root directory.

    Returns:
        dict: For each inlist option, states whether it is available in the version of mesa specified in `mesa_dir`.
    """
    version = get_mesa_version(mesa_dir)
    if compare_version(version, '15140', '>='):
        separate_eoskap = True
    else:
        separate_eoskap = False

    nml = evaluate_inlist(path)
    keys_defaults = get_mesa_defaults(mesa_dir)

    all_checked = {}
    for nml_type, sub_nml in nml.items():
        if nml_type not in keys_defaults.keys():
            all_checked[nml_type] = {f'!namelist {nml_type}': 'not in defaults'}
            continue
        key_arr = [key for key in sub_nml.keys()]

        def_key = keys_defaults[nml_type]

        in_default = {key: key in def_key for key in key_arr}

        in_default = {}
        for key in key_arr:
            key_to_check = key
            if nml_type == 'controls':
                if 'ctrl(' in key:
                    key_to_check = key.split('(')[0] + '(1:num_x_ctrls)'
            in_default[key] = key_to_check in def_key

        to_remove = [key for key in key_arr if not in_default[key]]

        orders = {key: 'not in defaults' for key in to_remove}
        # Check if have to move from star_job/controls to eos or kap
        if (nml_type in ['star_job', 'controls']) and separate_eoskap:
            for key in to_remove:
                for new_nml_type in ['eos', 'kap']:
                    if key in keys_defaults[new_nml_type]:
                        orders[key] = f'move to {new_nml_type}'
                        break
                    else:
                        if key.startswith('kappa'):
                            new_key = 'kap' + key[5:]
                            if new_key in keys_defaults[new_nml_type]:
                                orders[key] = f'rename to {new_key} and move to {new_nml_type}'
        all_checked[nml_type] = orders

    # Print nicely
    for nml_type, orders in all_checked.items():
        if len(orders) == 0:
            continue
        print(nml_type, len(orders))
        for key, order in orders.items():
            print(f'    {key}: {order}')
        print()

    return all_checked
