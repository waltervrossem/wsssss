#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import f90nml
import sys


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
    to_read = []
    use_kinds = []
    for i, kind in enumerate(defaults.keys()):
        to_read.append([])
        if kind not in inlist.keys():
            continue
        else:
            use_kinds.append(kind)
        for j in range(5):
            if f'read_extra_{kind}_inlist{j+1}' in inlist[kind].keys():
                if inlist[kind][f'read_extra_{kind}_inlist{j+1}']:
                    to_read[i].append(inlist[kind][f'extra_{kind}_inlist{j+1}_name'])
            elif f'read_extra_{kind}_inlist' in inlist[kind].keys():
                if inlist[kind][f'read_extra_{kind}_inlist'][j]:
                    to_read[i].append(inlist[kind][f'extra_{kind}_inlist_name'][j])

    inlists = []
    for sub_to_read in to_read:
        sub_inlists = []
        for fname in sub_to_read:
            if fname is None:
                sub_inlists.append({})
            else:
                sub_inlists.append(parser.read(f'{inlist_dir}/{fname}'))
        inlists.append(sub_inlists)

    out = f90nml.Namelist()
    for i, kind in enumerate(defaults):
        if kind not in use_kinds:
            continue
        out[kind] = f90nml.Namelist()
        for nml in inlists[i]:
            for k, v in nml[kind].items():
                out[kind][k] = v

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
