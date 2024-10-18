#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

import f90nml

parser = f90nml.Parser()
parser.global_start_index = 1

defaults = {'star_job': 'star/defaults/star_job.defaults',
            'eos': 'eos/defaults/eos.defaults',
            'kap': 'kap/defaults/kap.defaults',
            'controls': 'star/defaults/controls.defaults',
            'pgstar': 'star/defaults/pgstar.defaults',
            }


def inlist_diff(dict1, dict2):
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
    inlist = parser.reads(inlist_str)
    return _evaluate_inlist(inlist, inlist_dir)


def evaluate_inlist(path):
    inlist = parser.read(path)
    inlist_dir = os.path.dirname(path)
    return _evaluate_inlist(inlist, inlist_dir)


def _evaluate_inlist(inlist, inlist_dir):
    to_read = {}

    old_read_extra = 'read_extra_{}_inlist{}'
    old_read_extra_filename = 'extra_{}_inlist{}_name'
    new_read_extra = 'read_extra_{}_inlist'  # (:) part has been stripped off in inlist
    new_read_extra_filename = 'extra_{}_inlist_name'

    for kind in defaults.keys():
        if kind not in inlist.keys():
            continue
        to_read[kind] = []

        for i in range(5):
            old_key = old_read_extra.format(kind, i+1)
            new_key = new_read_extra.format(kind)
            if old_key in inlist[kind].keys():
                if inlist[kind][old_key]:
                    to_read[kind].append(inlist[kind][old_read_extra_filename.format(kind, i+1)])
            elif new_key in inlist[kind].keys():
                if inlist[kind][new_key][i]:
                    to_read[kind].append(inlist[kind][new_read_extra_filename.format(kind)][i])

    inlists = {}
    for kind, fnames in to_read.items():
        inlists[kind] = []
        old_key = old_read_extra.format(kind, i + 1)
        new_key = new_read_extra.format(kind)
        for fname in fnames:
            sub_inlist = parser.read(os.path.join(inlist_dir, fname))
            if (old_key in sub_inlist.keys()) or (new_key in sub_inlist.keys()):
                sub_inlist = _evaluate_inlist(sub_inlist, inlist_dir)

            if kind in sub_inlist.keys():
                inlists[kind].append(sub_inlist[kind])

    out = f90nml.Namelist(inlist)
    for kind, sub_inlists in inlists.items():
        for sub_inlist in sub_inlists:
            out[kind].update(sub_inlist)
        old_key = old_read_extra.format(kind, i + 1)
        new_key = new_read_extra.format(kind)
        for key in [old_key, new_key, old_read_extra_filename.format(kind, i+1), new_read_extra_filename.format(kind)]:
            if key in out[kind].keys():
                _ = out[kind].pop(key)
    return out


def print_dict(to_print):
    for k, v in to_print.items():
        print(f'{k}: {v}')


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
