#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import f90nml
import sys


def dict_diff(dict_a, dict_b, show_value_diff=True):
  result = {}
  result['added']   = {k: dict_b[k] for k in set(dict_b) - set(dict_a)}
  result['removed'] = {k: dict_a[k] for k in set(dict_a) - set(dict_b)}
  if show_value_diff:
    common_keys =  set(dict_a) & set(dict_b)
    result['value_diffs'] = {
      k:(dict_a[k], dict_b[k])
      for k in common_keys
      if dict_a[k] != dict_b[k]
    }
  return result


def dict_diff(dict1, dict2):
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

def evaluate_inlist(path):
    parser = f90nml.Parser()
    parser.global_start_index = 1
    inlist = parser.read(path)
    base_dir = os.path.dirname(path)
    to_read = [[], []]
    for i, kind in enumerate(['star_job', 'controls']):
        for j in range(5):
            j += 1
            if f'read_extra_{kind}_inlist{j}' in inlist[kind].keys():
                if inlist[kind][f'read_extra_{kind}_inlist{j}']:
                    to_read[i].append(inlist[kind][f'extra_{kind}_inlist{j}_name'])

    inlists = [[parser.read(f'{base_dir}/{fname}') for fname in to_read[0]],
               [parser.read(f'{base_dir}/{fname}') for fname in to_read[1]]]

    star_job = f90nml.Namelist()
    for nml in inlists[0]:
        for k, v in nml['star_job'].items():
            star_job[k] = v
    controls = f90nml.Namelist()
    for nml in inlists[1]:
        for k, v in nml['controls'].items():
            controls[k] = v

    out = f90nml.Namelist()
    out['star_job'] = star_job
    out['controls'] = controls

    return out

def print_dict(to_print):
    for k, v in to_print.items():
        print(f'{k}: {v}')

def compare_inlist(path1, path2, show_same=False):

    inlist1 = evaluate_inlist(path1)
    inlist2 = evaluate_inlist(path2)

    star_job = dict_diff(inlist1['star_job'], inlist2['star_job'])
    controls = dict_diff(inlist1['controls'], inlist2['controls'])

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


# if __name__ == '__main__':
#     args = sys.argv
#     path1 = sys.argv[1]
#     path2 = sys.argv[2]
#     compare_inlist(path1, path2)
