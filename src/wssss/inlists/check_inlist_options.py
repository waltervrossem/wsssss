#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import f90nml

version = '15140'
mesa_dir = rf'\\wsl$\Debian\home\walterLP2WSL\mesa\mesa-r{version}'

path_defaults = {'star_job': 'star/defaults/star_job.defaults',
                 'controls': 'star/defaults/controls.defaults',
                 'pgstar': 'star/defaults/pgstar.defaults',
                 'kap': 'kap/defaults/kap.defaults',
                 'eos': 'eos/defaults/eos.defaults'}


keys_defaults = {}
for nml_type, path in path_defaults.items():
    path_to_file = os.path.join(mesa_dir, path)

    with open(path_to_file, 'r') as handle:
        s_raw = handle.read()

    # Get rid of empty lines and comments
    s = s_raw.lower().splitlines()
    s = [_ for _ in s if not _.strip().startswith('!')]  # Get rid of fortran comments
    s = [_ for _ in s if _]  # Get rid of ''
    s_no_com = '\n'.join(s)

    keys = [_.split('=')[0].strip() for _ in s]
    keys_defaults[nml_type] = keys


def compare_inlist(path_to_inlist):
    nml = f90nml.read(path_to_inlist)

    all_orders = {}
    for nml_type, sub_nml in nml.items():
        key_arr = list(sub_nml.keys())
        key_arr = [key for key in key_arr]
        def_key = keys_defaults[nml_type]

        in_default = {key: key in def_key for key in key_arr}
        to_remove = [key for key in key_arr if not in_default[key]]

        orders = {key: 'del' for key in to_remove}
        # Check if have to move from star_job/controls to eos or kap
        if nml_type in ['star_job', 'controls']:
            for key in to_remove:
                for new_nml_type in ['eos', 'kap']:
                    if key in keys_defaults[new_nml_type]:
                        orders[key] = f'move to {new_nml_type}'
                        break
                    else:
                        if key.startswith('kappa'):
                            new_key = 'kap'+key[5:]
                            if new_key in keys_defaults[new_nml_type]:
                                orders[key] = f'rename to {new_key} and move to {new_nml_type}'
        all_orders[nml_type] = orders

    #Print nicely
    for nml_type, orders in all_orders.items():
        if len(orders) == 0:
            continue
        print(nml_type, len(orders))
        for key, order in orders.items():
            print(f'    {key}: {order}')
        print()

    return all_orders

