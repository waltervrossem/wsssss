#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import itertools
import os
import shutil

import numpy as np

from .inlists import defaults, evaluate_inlist, variable_to_string
from ..functions import get_mesa_version, compare_version

non_mesa_key_start = '!PY_KEY_'


class MesaGrid:
    def __init__(self, mesa_dir='', inlist_filename='inlist', starjob_filename='inlist_project',
                 controls_filename='inlist_project', eos_filename='inlist_project', kap_filename='inlist_project',
                 pgstar_filename='inlist_project'):

        if inlist_filename in [starjob_filename, controls_filename, eos_filename, kap_filename, pgstar_filename]:
            raise ValueError(
                '`inlist_filename` cannot be the same as `starjob_filename`, `controls_filename`, `eos_filename`, `kap_filename`, or `pgstar_filename`.')

        if mesa_dir == '':
            self.mesa_dir = os.environ['MESA_DIR']
            self.version = get_mesa_version(self.mesa_dir)
        else:
            self.version = get_mesa_version(mesa_dir)
            self.mesa_dir = mesa_dir

        if compare_version(self.version, '15140', '>='):
            self.namelists = ('star_job', 'eos', 'kap', 'controls', 'pgstar')
            self.separate_eoskap = True
        else:
            self.namelists = ('star_job', 'controls', 'pgstar')
            self.separate_eoskap = False

        if compare_version(self.version, 'r23.05.1', '>='):
            self.extra_inlist_as_list = True
        else:
            self.extra_inlist_as_list = False

        # Setup namelist objects
        self.star_job = {f'{non_mesa_key_start}filename': starjob_filename,
                         f'{non_mesa_key_start}type': 'star_job',
                         f'{non_mesa_key_start}group_unpack': []}
        self.controls = {f'{non_mesa_key_start}filename': controls_filename,
                         f'{non_mesa_key_start}type': 'controls',
                         f'{non_mesa_key_start}group_unpack': []}
        if self.separate_eoskap:
            self.eos = {f'{non_mesa_key_start}filename': eos_filename,
                        f'{non_mesa_key_start}type': 'eos',
                        f'{non_mesa_key_start}group_unpack': []}
            self.kap = {f'{non_mesa_key_start}filename': kap_filename,
                        f'{non_mesa_key_start}type': 'kap',
                        f'{non_mesa_key_start}group_unpack': []}
        self.pgstar = {f'{non_mesa_key_start}filename': pgstar_filename,
                       f'{non_mesa_key_start}type': 'pgstar', }
        self.inlist = {f'{non_mesa_key_start}filename': inlist_filename,
                       f'{non_mesa_key_start}type': 'master',
                       f'{non_mesa_key_start}group_unpack': []}
        for namelist in self.namelists:
            namelist_filename = self.__dict__[namelist][f'{non_mesa_key_start}filename']
            i = 5
            if self.extra_inlist_as_list:
                read_extra_inlist_key = f'read_extra_{namelist}_inlist({i})'
                read_extra_inlist_name_key = f'extra_{namelist}_inlist_name({i})'
            else:
                read_extra_inlist_key = f'read_extra_{namelist}_inlist{i}'
                read_extra_inlist_name_key = f'extra_{namelist}_inlist{i}_name'
            self.inlist[namelist] = {read_extra_inlist_key: False,
                                     read_extra_inlist_name_key: namelist_filename}

        self.extra_files = []
        self.extra_dirs = []
        self.inlist_finalize_function = lambda unpacked_namelist: unpacked_namelist
        self.griddir_finalize_function = lambda dirname: None
        self.name_funcion = None
        self.unpacked = False

        self.inlist_option_files_to_validate = [('star_job', 'history_columns_file'),
                                                ('star_job', 'profile_columns_file')]

    def __repr__(self):
        s = f'MESA Grid for version {self.version}.'
        if self.unpacked:
            s += f' {len(self.unpacked)} runs.'
        return s

    def add_file(self, path):
        """Add a file which will be copied into each grid directory."""
        if os.path.isfile(path):
            self.extra_files.append(path)
        else:
            raise FileNotFoundError(f'{path} is not a file.')

    def add_inlist_option_file_check(self, namelist, filename):
        """Check that the `filename` is in a grid directory."""
        if namelist not in self.namelists:
            raise ValueError(f'`namelist` {namelist} must be one of {", ".join(self.namelists)}.')
        self.inlist_option_files_to_validate.append((namelist, filename))

    def add_dir(self, path):
        """Add a directory which will be copied into each grid directory."""
        if os.path.isdir(path):
            self.extra_dirs.append(path)
        else:
            raise NotADirectoryError(f'{path} is not a directory.')

    def set_inlist_finalize_function(self, function):
        """Set the function which is applied to all unpacked namelists.
        Each namelist will have the `type` key to identify its type."""
        self.inlist_finalize_function = function

    def set_griddir_finalize_function(self, function):
        """Set the function which is called in each grid directory."""
        self.griddir_finalize_function = function

    def create_grid(self, grid_path):
        """Create a grid for MESA in `grid_path`."""
        self.validate_inlists()

        self.unpack_inlists()

        self.check_copy(grid_path)

        self._setup_directories(grid_path)

        self._write_inlists(grid_path)

        self._copy_extra_files_and_dirs(grid_path)

        curdir = os.path.abspath('.')
        abs_grid_path = os.path.abspath(grid_path)
        for dirname in self.dirnames:
            os.chdir(os.path.join(abs_grid_path, dirname))
            self.griddir_finalize_function(self)
        os.chdir(curdir)

        self.validate_files(grid_path)

    def validate_inlists(self, mesa_dir=None):
        """Check if all options in the inlists are valid MESA keys."""
        # Check if all extra namelists filenames are unique and different from the main inlist name.
        for namelist in self.namelists:
            read_extra_names = []
            for i in range(1, 6):
                if self.extra_inlist_as_list:
                    read_extra_inlist_key = f'read_extra_{namelist}_inlist({i})'
                    read_extra_inlist_name_key = f'extra_{namelist}_inlist_name({i})'
                else:
                    read_extra_inlist_key = f'read_extra_{namelist}_inlist{i}'
                    read_extra_inlist_name_key = f'extra_{namelist}_inlist{i}_name'

                if not read_extra_inlist_key in self.inlist[namelist].keys():
                    continue

                if i == 5:  # If any options have been added, set read_extra_inlist_key to True
                    nml_keys = [key for key in self.__dict__[namelist].keys() if not key.startswith(non_mesa_key_start)]
                    if (len(nml_keys) > 0) and (self.inlist[namelist][read_extra_inlist_key] == False):
                        self.inlist[namelist][read_extra_inlist_key] = True

                if self.inlist[namelist][read_extra_inlist_key]:
                    read_extra_names.append(self.inlist[namelist][read_extra_inlist_name_key])
                    if self.inlist[f'{non_mesa_key_start}filename'] == self.inlist[namelist][
                        read_extra_inlist_name_key]:
                        raise ValueError(
                            f'Main inlist filename {self.inlist[f"{non_mesa_key_start}filename"]} cannot be the same as an extra inlist filename {self.inlist[namelist][read_extra_inlist_name_key]}.')

            if len(read_extra_names) != len(np.unique(read_extra_names)):
                raise ValueError(f'Extra {namelist} inlist names must be unique.')

        # Get available MESA options from mesa_dir
        if mesa_dir is None:
            mesa_dir = self.mesa_dir
        available_options = {}
        for namelist in self.namelists:
            available_options[namelist] = self._get_available_options(f'{mesa_dir}/{defaults[namelist]}')

        # Find which options are not in MESA options.
        failed_options = {}
        failed = False
        for namelist in self.namelists:
            failed_options[namelist] = []
            for key in self.__dict__[namelist].keys():
                if not key.startswith(non_mesa_key_start):
                    if key.endswith(')'):
                        compare_key = key.split('(')[0]  # Cut off array index part
                    else:
                        compare_key = key
                    if compare_key.lower() not in available_options[namelist]:
                        failed_options[namelist].append(key)
                        failed = True

        if failed:
            s = []
            for namelist in self.namelists:
                if len(failed_options[namelist]) > 0:
                    s.append(f'Option(s) not in available {namelist} keys: {", ".join(failed_options[namelist])}.')
            s = ' '.join(s)
            raise KeyError(s)

    def validate_files(self, grid_path):

        file_not_found = []
        # Check for history/profile column files.
        for i, dirname in enumerate(self.dirnames):
            run_dir = os.path.join(grid_path, dirname)
            inlist = evaluate_inlist(os.path.join(run_dir, self.inlist[f'{non_mesa_key_start}filename']))
            for (kind, key) in self.inlist_option_files_to_validate:
                if kind in inlist.keys():
                    if key in inlist[kind].keys():
                        fname = inlist[kind][key]
                        if not os.path.isfile(os.path.join(run_dir, fname)):
                            file_not_found.append(fname)

        s = ''
        if file_not_found:
            s += f'Missing {len(file_not_found)} files: ' + ' '.join(file_not_found)
            s += '\n'

        if s:
            raise FileNotFoundError(s)

    def _get_available_options(self, path):
        with open(path, 'r') as handle:
            lines = handle.readlines()
        lines = [_.strip() for _ in lines]
        lines = [_ for _ in lines if not _.startswith('!') and _]
        lines = [_.split('=')[0].strip() for _ in lines]
        lines = [_.lower() for _ in lines]
        lines = [_.replace('(:)', '') for _ in lines]
        return lines

    def check_copy(self, grid_path):
        """Check if all files specified exists."""
        file_not_found = []
        for fpath in self.extra_files:
            if not os.path.isfile(fpath):
                file_not_found.append(fpath)

        dir_not_found = []
        for dpath in self.extra_dirs:
            if not os.path.isdir(dpath):
                dir_not_found.append(dpath)

        # Check for name collisions

        s = ''
        if file_not_found:
            s += f'Missing {len(file_not_found)} files: ' + ' '.join(file_not_found)
            s += '\n'

        if dir_not_found:
            s += f'Missing {len(dir_not_found)} directories: ' + ' '.join(dir_not_found)

        if s:
            raise FileNotFoundError(s)

    def unpack_inlists(self):
        generators = []
        for namelist in ['inlist', *self.namelists]:
            generators.append(self._make_inlist_generator(self.__dict__[namelist]))

        all_unpacked = []
        for i, unpacked in enumerate(itertools.product(*generators)):
            unpacked = list(unpacked)
            for j, unpacked_namelist in enumerate(unpacked):
                unpacked[j] = self.inlist_finalize_function(unpacked_namelist)
            all_unpacked.append(dict(zip(['inlist', *self.namelists], unpacked)))
        self.unpacked = tuple(all_unpacked)

    def _make_inlist_generator(self, inlist_dict):
        """
        Create a generator which yields all unique inlists with all combinations of any lists, tuples, or numpy arrays of length greater than 1.
        Can handle groups of variables which must change together by appending a list of `dict`s in the `group_unpack` key. If multiple lists of
        `dict`s are appended, they are treated as separate variables.
        """

        if inlist_dict[f'{non_mesa_key_start}type'] == 'master':
            yield inlist_dict
            return
        if inlist_dict[f'{non_mesa_key_start}type'] == 'pgstar':
            yield inlist_dict
            return

        inlist_dict[f'{non_mesa_key_start}unpacknumber'] = 0
        inlist_dict[f'{non_mesa_key_start}unpacknumber_total'] = 1

        contains_list = []  # Keys which need to be unpacked
        lengths = []  # Length of lists which need to be unpacked
        items = []  # Lists which need to be unpacked
        for key, item in inlist_dict.items():
            if key.startswith(non_mesa_key_start):
                if key == f'{non_mesa_key_start}group_unpack':
                    groups = inlist_dict[f'{non_mesa_key_start}group_unpack']
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
            if type(item) in [list, tuple, np.ndarray]:
                if len(item) > 1:
                    contains_list.append(key)
                    lengths.append(len(item))
                    items.append(inlist_dict[key])
                elif len(item) == 1:  # Remove item from list
                    inlist_dict[key] = item[0]


        if len(lengths) == 0:
            out_inlist = inlist_dict.copy()
            for namelist in self.namelists:
                if f'{non_mesa_key_start}group_unpack' in out_inlist.keys():
                    out_inlist.pop(f'{non_mesa_key_start}group_unpack')
            yield out_inlist
            return

        tot_permutations = 1
        for n in lengths:
            tot_permutations *= n

        inlist_dict[f'{non_mesa_key_start}unpacknumber_total'] = tot_permutations
        base_inlist = inlist_dict.copy()
        for namelist in self.namelists:
            if f'{non_mesa_key_start}group_unpack' in base_inlist.keys():
                base_inlist.pop(f'{non_mesa_key_start}group_unpack')

        permutations = itertools.product(*items)
        for i, permutation in enumerate(permutations):
            new_inlist = base_inlist.copy()
            for j, value in enumerate(permutation):
                if contains_list[j].startswith('__group__'):
                    new_inlist.update(value)
                else:
                    new_inlist[contains_list[j]] = value
            new_inlist[f'{non_mesa_key_start}unpacknumber'] = i
            yield new_inlist

    def _setup_directories(self, grid_path):
        num_unpacked = len(self.unpacked)
        self.dirnames = []
        if self.name_funcion is None:
            num_digits = max(4, np.ceil(np.log10(num_unpacked)).astype(int))
            for i in range(num_unpacked):
                dirname = f'{i:0{num_digits}}'
                self.dirnames.append(dirname)
        else:
            for i in range(num_unpacked):
                dirname = self.name_funcion(self.unpacked[i])
                self.dirnames.append(dirname)
        self.dirnames = tuple(self.dirnames)

        if os.path.isfile(grid_path):
            raise FileExistsError(f'Expected `grid_path` {grid_path} is a file.')

        if os.path.exists(grid_path) and os.path.isdir(grid_path):  # Remove existing grid.
            shutil.rmtree(grid_path)

        os.makedirs(grid_path)
        for dirname in self.dirnames:
            os.makedirs(os.path.join(grid_path, dirname))

    def _write_inlists(self, grid_path):
        for i, dirname in enumerate(self.dirnames):
            dirpath = os.path.join(grid_path, dirname)
            unpacked = self.unpacked[i]
            for namelist in unpacked.keys():
                file_path = os.path.join(dirpath, unpacked[namelist][f'{non_mesa_key_start}filename'])
                if not os.path.exists(file_path):
                    prepend = 'Created using wssss.create_grid module.\n'
                else:
                    prepend = ''
                with open(file_path, 'a') as handle:
                    handle.write(prepend + self._generate_inlist_string(self.unpacked[i][namelist]))

    def _generate_inlist_string(self, inlist_dict):
        """
        Generates an inlist string from `inlist_dict`. Which is an unpacked controls etc dict.
        """
        inlist_string = ''

        if len([_ for _ in inlist_dict.keys() if not _.startswith(non_mesa_key_start)]) == 0:
            return inlist_string

        inlist_type = inlist_dict[f'{non_mesa_key_start}type']
        if inlist_type == 'master':
            for key in inlist_dict.keys():
                if key.startswith(non_mesa_key_start):
                    continue
                elif key.startswith('#'):
                    continue
                key_dat = inlist_dict[key]
                sub_str = f'\n&{key}\n'

                for sub_key, sub_value in key_dat.items():
                    if sub_key.startswith(non_mesa_key_start):
                        continue
                    parsed_sub_value = variable_to_string(sub_value)
                    sub_str += f'    {sub_key} = {parsed_sub_value}\n'

                sub_str += rf'/ ! end of {key} namelist'
                sub_str += '\n'

                inlist_string += sub_str
        else:
            sub_str = f'\n&{inlist_type}\n'
            for key, value in inlist_dict.items():
                if key.startswith(non_mesa_key_start):
                    if key == 'note':
                        sub_str = f'! {value}\n' + sub_str
                        continue
                    else:
                        continue
                elif key.startswith('#'):
                    continue
                parsed_value = variable_to_string(value)
                sub_str += f'    {key} = {parsed_value}\n'

            sub_str += f'/ ! end of {inlist_type} namelist\n'

            inlist_string += sub_str

        return inlist_string



    def _copy_extra_files_and_dirs(self, grid_path):
        for dirname in self.dirnames:
            dirpath = os.path.join(grid_path, dirname)
            for fpath in self.extra_files:
                fname = os.path.basename(fpath)
                shutil.copy2(fpath, os.path.join(dirpath, fname))
            for dpath in self.extra_dirs:
                dname = os.path.basename(dpath)
                shutil.copytree(dpath, os.path.join(dirpath, dname))

    def summary(self):
        """Print a summary of which variables change."""
        raise NotImplementedError()
        if not self.unpacked:
            self.unpack_inlists()

        for kind in self.namelists:
            pass
