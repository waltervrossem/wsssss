#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import itertools
import os
import shutil
import copy
import sys
import inspect

import numpy as np

from .inlists import defaults, evaluate_inlist, variable_to_string
from ..functions import get_mesa_version, compare_version

non_mesa_key_start = '!PY_KEY_'
kap_user_params = ['user_num_kap_Xs', 'user_kap_Xs', 'user_num_kap_Zs', 'user_kap_Zs',
                   'user_num_kap_Xs_for_this_Z',
                   'user_num_kap_CO_Xs', 'user_kap_CO_Xs', 'user_num_kap_CO_Zs', 'user_kap_CO_Zs',
                   'user_num_kap_CO_Xs_for_this_Z',
                   'user_num_kap_lowT_Xs', 'user_kap_lowT_Xs', 'user_num_kap_lowT_Zs', 'user_kap_lowT_Zs',
                   'user_num_kap_lowT_Xs_for_this_Z']
class MesaGrid:
    def __init__(self, mesa_dir='', inlist_filename='inlist', inlists_index=5, starjob_filename='inlist_project',
                 controls_filename='inlist_project', eos_filename='inlist_project', kap_filename='inlist_project',
                 pgstar_filename='inlist_project', add_base_workdir=False):
        """
        `MesaGrid` class which contains all inlist settings for a grid. When the `create_grid` method is called,
        a copy of the script which called it is copied into the grid directory.

        Args:
            mesa_dir (str): ``$MESA_DIR`` root directory to be used with this grid.
            inlist_filename (str, optional): Defaults to 'inlist'.
            inlists_index (int, optional): Which read_extra_*_inlist(inlists_index) to use. Defaults to 5.
            starjob_filename (str, optional): Cannot be the same as `inlist_filename`. Defaults to 'inlist_project'.
            controls_filename (str, optional): Cannot be the same as `inlist_filename`. Defaults to 'inlist_project'.
            eos_filename (str, optional): Cannot be the same as `inlist_filename`. Defaults to 'inlist_project'.
            kap_filename (str, optional): Cannot be the same as `inlist_filename`. Defaults to 'inlist_project'.
            pgstar_filename (str, optional): Cannot be the same as `inlist_filename`. Defaults to 'inlist_project'.
            add_base_workdir (bool, optional): Add minimum required files from $MESA_DIR/star/work to each run directory (make/, src/, mk, re, rn, clean).

        Attributes:
            star_job (dict): Options for ``star_job``.
            controls (dict): Options for ``controls``.
            pgstar (dict): Options for ``pgstar``.
            kap (dict): Options for ``kap``. Only exists if `mesa_dir` version is later than or equal to 15140.
            eos (dict): Options for ``eos``. Only exists if `mesa_dir` version is later than or equal to 15140.

        Examples:
            Creating a grid of MESA runs with initial masses of 1 and 2 Msol.

            >>> import os
            >>> from wsssss.inlists import create_grid as cg
            >>> grid = cg.MesaGrid()
            >>> grid.controls['initial_mass'] = [1, 2]
            >>> grid.create_grid('path/to/grid')
            >>> os.listdir('path/to/grid')
            ['0000', '0001]
        TODO:
            * Add astero namelist and include in tests.
        """

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
        # self.astero = {f'{non_mesa_key_start}filename': astero_filename,
        #                  f'{non_mesa_key_start}type': 'astero',
        #                  f'{non_mesa_key_start}group_unpack': []}
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

        self.inlists_index = inlists_index
        for namelist in self.namelists:
            namelist_filename = self.__dict__[namelist][f'{non_mesa_key_start}filename']
            i = self.inlists_index
            if self.extra_inlist_as_list:
                read_extra_inlist_key = f'read_extra_{namelist}_inlist({i})'
                read_extra_inlist_name_key = f'extra_{namelist}_inlist_name({i})'
            else:
                read_extra_inlist_key = f'read_extra_{namelist}_inlist{i}'
                read_extra_inlist_name_key = f'extra_{namelist}_inlist{i}_name'

            # Create entry and default to True.
            self.inlist[namelist] = {read_extra_inlist_key: True,
                                     read_extra_inlist_name_key: namelist_filename}

        self.extra_files = []
        self.extra_dirs = []
        self.inlist_finalize_function = None
        self.griddir_finalize_function = None
        self.name_funcion = None
        self.unpacked = False

        self.inlist_option_files_to_validate = [('star_job', 'history_columns_file'),
                                                ('star_job', 'profile_columns_file')]

        if add_base_workdir:
            for fname in ['mk', 'clean', 'rn', 're']:
                self.extra_files.append(os.path.join(f'{self.mesa_dir}/star/work', fname))
            for dname in ['make', 'src']:
                self.extra_dirs.append(os.path.join(f'{self.mesa_dir}/star/work', dname))

        no_expand = {namelist: [] for namelist in self.namelists}
        no_expand['kap'] = ['user_kap_Xs', 'user_kap_Zs', 'user_num_kap_Xs_for_this_Z', 'user_kap_CO_Xs',
                             'user_kap_CO_Zs', 'user_num_kap_CO_Xs_for_this_Z', 'user_kap_lowT_Xs', 'user_kap_lowT_Zs',
                             'user_num_kap_lowT_Xs_for_this_Z']
        no_expand['kap'] = [_.lower() for _ in no_expand['kap']]
        self.no_expand = no_expand
        self.expand_non_mesa_keys = {namelist:[] for namelist in self.namelists}

    def __repr__(self):
        s = f'MESA Grid for version {self.version}.'
        if self.unpacked:
            s += f' {len(self.unpacked)} runs.'
        return s

    def add_file(self, path):
        """
        Add a file which will be copied into each run directory. Will raise a FileNotFoundError if the file at
        `path` does not exist.

        Args:
            path (str): Path to file to be added to every run directory.

        Examples:
            Copy the file at ``path/to/file`` to each run directory when the grid is created.

            >>> from wsssss.inlists import create_grid as cg
            >>> grid = cg.MesaGrid()
            >>> grid.add_file('path/to/file')
        """
        if os.path.isfile(path):
            self.extra_files.append(path)
        else:
            raise FileNotFoundError(f'{path} is not a file.')

    def add_dir(self, path):
        """
        Add a directory which will be copied into each run directory.

        Args:
            path (str): Path to directory to be added to every run directory.

        Examples:
            Copy the directory at ``path/to/directory`` to each run directory when the grid is created.

            >>> from wsssss.inlists import create_grid as cg
            >>> grid = cg.MesaGrid()
            >>> grid.add_dir('path/to/directory')
        """
        if os.path.isdir(path):
            self.extra_dirs.append(path)
        else:
            raise NotADirectoryError(f'{path} is not a directory.')

    def add_inlist_option_file_check(self, namelist, option):
        """
        Add a check that the file specified in the `namelist[option]` must exist in each run directory,
        e.g. controls['history_columns_filename'], which is included by default as well as ``profile_columns_filename``.

        Args:
            namelist (str):
            option (str):

        Examples:

            >>> from wsssss.inlists import create_grid as cg
            >>> grid = cg.MesaGrid()
            >>> grid.add_inlist_option_file_check('controls', 'history_columns_file')

        """
        if namelist not in self.namelists:
            raise ValueError(f'`namelist` {namelist} must be one of {", ".join(self.namelists)}.')
        self.inlist_option_files_to_validate.append((namelist, option))


    def add_no_expand_key(self, namelist, key):
        """
        Add a key which will not be expanded. For example &kap's ``user_kap_Xs``
        Args:
            namelist (str): Which namelist the key is part of. E.g. 'kap'.
            key (str): Name of parameter. E.g. 'user_kap_Xs'.

        Examples:

            >>> from wsssss.inlists import create_grid as cg
            >>> grid = cg.MesaGrid()
            >>> grid.add_no_expand_key('kap', f'user_kap_Xs')

        """

        self.no_expand[namelist].append(key)

    def add_non_mesa_key(self, namelist, key):
        """
        Add a key which is not in MESA's keys and expand it. For example including a parameter which is used in
        ``inlist_finalize_function``.
        Args:
            namelist (str): Which namelist the key is part of. E.g. controls.
            key (str): Name of parameter. Must start with ``'!PY_KEY_'``.
        """

        if not key.startswith(non_mesa_key_start):
            raise ValueError(f'Key {key} must start with "{non_mesa_key_start}"')

        self.expand_non_mesa_keys[namelist].append(key)

    def set_inlist_finalize_function(self, function):
        """
        Set the function which is applied to all unpacked inlists.

        The function must accept a single `inlist` and return a single `inlist`.

        Args:
            function (function):

        Examples:
            Change an option in `star_job` depending on ``initial_mass`` in ``controls``. This could also be accomplished using the
            ``f'{non_mesa_key_start}group_unpack'`` key.
            
            >>> import numpy as np
            >>> from wsssss.inlists import create_grid as cg
            >>> grid = cg.MesaGrid()
            >>> grid.star_job['change_initial_net'] = True
            >>> grid.star_job['new_net_name'] = 'pp_extras.net'
            >>> grid.controls['initial_mass'] = np.linspace(1, 2, 6)
            >>> def finalize_function(unpacked_inlist):
            ...     if unpacked_inlist['controls']['initial_mass'] > 1.3:
            ...         unpacked_inlist['star_job']['new_net_name'] = 'pp_cno_extras.net'
            ...     return unpacked_inlist
            >>> grid.set_inlist_finalize_function(finalize_function)
            >>> grid.unpack_inlists()
            >>> grid.unpacked[0]['star_job']['new_net_name']
            'pp_extras.net'
            >>> grid.unpacked[-1]['star_job']['new_net_name']
            'pp_cno_extras.net'

        """
        self.inlist_finalize_function = function

    def set_griddir_finalize_function(self, function):
        """
        Set the function which is called in each grid directory. The arguments to the function are a MesaGrid object and
        the run index.

        Args:
            function (function):

        Examples:

            >>> import os
            >>> from wsssss.inlists import create_grid as cg
            >>> grid = cg.MesaGrid()
            >>> grid.controls['initial_mass'] = [1, 2]
            >>> def griddir_finalize_function(grid, i):
            ...     os.system("pwd")
            >>> grid.set_griddir_finalize_function(griddir_finalize_function)
            >>> grid.create_grid('path/to/grid')
            /home/walter/Github/wsssss/path/to/grid/0000
            /home/walter/Github/wsssss/path/to/grid/0001

        """
        self.griddir_finalize_function = function

    def set_name_function(self, function):
        """
        Set the function which is used to generate the name of each run directory.
        It recieves a fully unpacked inlist (dict of dicts) or grid index and unpacked inlist and returns a string.
        It should generate a unique name for each inlist.

        Args:
            function (function):

        Examples:

            >>> import os
            >>> from wsssss.inlists import create_grid as cg
            >>> grid = cg.MesaGrid()
            >>> grid.controls['initial_mass'] = [1, 2]
            >>> def name_function(unpacked_inlist):
            ...     return f'm{unpacked_inlist["controls"]["initial_mass"]:.3f}'
            >>> grid.set_name_function(name_function)
            >>> grid.create_grid('path/to/grid')
            >>> os.listdir('path/to/grid')
            ['m1.000', 'm2.000']
        """
        self.name_funcion = function

    def create_grid(self, grid_path, rm_dir=True):
        """
        Create the MESA grid in `grid_path`. The inlist options are validated and extra files and directories copied to
        each run directory.
        Args:
            grid_path: Path to grid directory.
            rmdir (bool, optional): Remove grid directory if it exists. Defaults to `True`.

        Examples:

            >>> import os
            >>> from wsssss.inlists import create_grid as cg
            >>> grid = cg.MesaGrid()
            >>> grid.controls['initial_mass'] = [1, 2]
            >>> grid.create_grid('path/to/grid')
            >>> os.listdir('path/to/grid')
            ['0000', '0001']
        """
        self.validate_inlists()

        self.unpack_inlists()

        self.check_copy()

        self._setup_directories(grid_path, rm_dir)

        self._write_inlists(grid_path)

        self._copy_extra_files_and_dirs(grid_path)

        curdir = os.path.abspath('.')
        abs_grid_path = os.path.abspath(grid_path)
        for i, dirname in enumerate(self.dirnames):
            os.chdir(os.path.join(abs_grid_path, dirname))
            if self.griddir_finalize_function is not None:
                self.griddir_finalize_function(self, i)
        os.chdir(curdir)

        self._validate_files(grid_path)

        # Copy file which called create_grid into grid_path
        if len(inspect.stack()) > 1:
            calling_file = inspect.stack()[1].filename
            if os.path.isfile(calling_file) and not calling_file.endswith('IPython/core/interactiveshell.py'):
                shutil.copy2(calling_file, grid_path)
            else:
                print(f"Cannot copy calling file {calling_file} to grid directory.")
        else:
            pass


    def validate_inlists(self, mesa_dir=None):
        """
        Check if all options in the inlists are valid MESA keys.

        Args:
            mesa_dir (str, optional): MESA root directory to check against. Defaults to None, which will use ``$MESA_DIR``.

        Examples:

            >>> from wsssss.inlists import create_grid as cg
            >>> grid = cg.MesaGrid()
            >>> grid.controls['this_option_does_not_exist'] = 1
            KeyError: 'Option(s) not in available controls keys: this_option_does_not_exist.'
        """
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

                if read_extra_inlist_key not in self.inlist[namelist].keys():
                    continue

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
            available_options[namelist] = self._get_available_options(f'{mesa_dir}/{defaults[namelist]}')  # TODO: replace with inlists.get_mesa_defaults

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
                        if namelist == 'kap':  # These options are commented out in kap.defaults
                            if compare_key.lower() in [_.lower() for _ in kap_user_params]:
                                continue
                        failed_options[namelist].append(key)
                        failed = True

        if failed:
            s = []
            for namelist in self.namelists:
                if len(failed_options[namelist]) > 0:
                    s.append(f'Option(s) not in available {namelist} keys: {", ".join(failed_options[namelist])}.')
            s = ' '.join(s)
            raise KeyError(s)

    def _validate_files(self, grid_path):
        """
        Check if all specified files are accounted for.
        Args:
            grid_path (str): Grid directory.

        """

        file_not_found = []

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
        """
        TODO: Replace with inlists.get_mesa_defaults
        Args:
            path:

        Returns:

        """
        with open(path, 'r') as handle:
            lines = handle.readlines()
        lines = [_.strip() for _ in lines]
        lines = [_ for _ in lines if not _.startswith('!') and _]
        lines = [_.split('=')[0].strip() for _ in lines]
        lines = [_.lower() for _ in lines]
        lines = [_.split('(')[0] for _ in lines]
        return lines

    def check_copy(self):
        """
        Check if all files and directories which are to be copied exist.

        """

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
        """
        Unpack all inlists. Will place them in the `unpacked` attribute as a tuple.

        """
        generators = []
        for namelist in ['inlist', *self.namelists]:
            generators.append(self._make_inlist_generator(self.__dict__[namelist]))

        all_unpacked = []
        for i, unpacked in enumerate(itertools.product(*generators)):
            unpacked = dict(zip(['inlist', *self.namelists], list(unpacked)))
            if self.inlist_finalize_function is not None:
                finalized_unpacked = self.inlist_finalize_function(copy.deepcopy(unpacked))
            else:
                finalized_unpacked = unpacked
            all_unpacked.append(finalized_unpacked)
        self.unpacked = tuple(all_unpacked)

        num_unpacked = len(self.unpacked)
        self.dirnames = []
        if self.name_funcion is None:
            num_digits = max(4, np.ceil(np.log10(num_unpacked)).astype(int))
            for i in range(num_unpacked):
                dirname = f'{i:0{num_digits}}'
                self.dirnames.append(dirname)
        else:
            for i in range(num_unpacked):
                if self.name_funcion.__code__.co_argcount == 1:
                    dirname = self.name_funcion(self.unpacked[i])
                else:
                    dirname = self.name_funcion(i, self.unpacked[i])
                if dirname in self.dirnames:
                    i_already_exists = self.dirnames.index(dirname)
                    raise ValueError(f'`dirname` {dirname} for inlist {i} has already been generated for inlist {i_already_exists}.')
                self.dirnames.append(dirname)
        self.dirnames = tuple(self.dirnames)

    def _make_inlist_generator(self, inlist_dict):
        """
        Create a generator which yields all unique inlists with all combinations of any lists, tuples, or numpy arrays of length greater than 1.
        Can handle groups of variables which must change together by appending a list of `dict`s in the `group_unpack` key. If multiple lists of
        `dict`s are appended, they are treated as separate variables.
        Args:
            inlist_dict (dict):

        Yields:
            dict: Unpacked inlist.

        """
        inlist_dict = copy.deepcopy(inlist_dict)
        if inlist_dict[f'{non_mesa_key_start}type'] == 'master':
            yield inlist_dict
            return
        if inlist_dict[f'{non_mesa_key_start}type'] == 'pgstar':
            yield inlist_dict
            return

        contains_list = list()  # Keys which need to be unpacked
        lengths = list()  # Length of lists which need to be unpacked
        items = list()  # Lists which need to be unpacked
        for key, item in inlist_dict.items():
            if key.lower() in self.no_expand[inlist_dict[f'{non_mesa_key_start}type']]:
                continue

            if key.startswith(non_mesa_key_start):
                if key == f'{non_mesa_key_start}group_unpack':
                    groups = inlist_dict[f'{non_mesa_key_start}group_unpack']
                    for i, group in enumerate(groups):
                        contains_list.append(f'{non_mesa_key_start}__group__{i}')
                        lengths.append(len(group))
                        items.append(group)
                        continue
                elif key in self.expand_non_mesa_keys[inlist_dict[f'{non_mesa_key_start}type']]:  # Continue as if it is a normal key and expand it
                    pass
                else:
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

        base_inlist = copy.deepcopy(inlist_dict)
        if f'{non_mesa_key_start}group_unpack' in base_inlist.keys():
            base_inlist.pop(f'{non_mesa_key_start}group_unpack')

        permutations = itertools.product(*items)
        for i, permutation in enumerate(permutations):
            new_inlist = base_inlist.copy()
            for j, value in enumerate(permutation):
                if contains_list[j].startswith(f'{non_mesa_key_start}__group__'):
                    new_inlist.update(value)
                else:
                    new_inlist[contains_list[j]] = value
            yield new_inlist

    def _setup_directories(self, grid_path, rm_dir):
        """
        Create the directory structure for a grid at ``grid_path``.

        Args:
            grid_path (str): Path which will contain the grid.
            rm_dir (bool): If it exists, first remove the grid directory.

        """

        if os.path.isfile(grid_path):
            raise FileExistsError(f'Expected `grid_path` {grid_path} is a file.')

        if os.path.exists(grid_path) and os.path.isdir(grid_path) and rm_dir:  # Remove existing grid.
            shutil.rmtree(grid_path)

        os.makedirs(grid_path)
        for dirname in self.dirnames:
            os.makedirs(os.path.join(grid_path, dirname))

    def _write_inlists(self, grid_path):
        """
        Write the unpacked inlists to ``grid_path`` in their respective directories.

        Args:
            grid_path (str): Path which will contain the grid.

        """
        for i, dirname in enumerate(self.dirnames):
            dirpath = os.path.join(grid_path, dirname)
            unpacked = self.unpacked[i]
            for namelist in unpacked.keys():
                file_path = os.path.join(dirpath, unpacked[namelist][f'{non_mesa_key_start}filename'])
                if not os.path.exists(file_path):
                    prepend = '!Created using wsssss.inlists.create_grid module.\n'
                else:
                    prepend = ''
                with open(file_path, 'a') as handle:
                    handle.write(prepend + self._generate_inlist_string(self.unpacked[i][namelist]))

    def _generate_inlist_string(self, inlist_dict):
        """
        Generates an inlist string from `inlist_dict`. Which is an unpacked controls etc dict.

        Args:
            inlist_dict (dict): Dictionary containing a set of inlist options.

        Returns:
            inlist_str (str): A string representation of ``inlist_dict`` readable by MESA.
        """
        inlist_string = ''
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
                if key.lower() in self.no_expand[inlist_type]:
                    parsed_value = ', '.join([variable_to_string(val) for val in value])
                else:
                    parsed_value = variable_to_string(value)
                sub_str += f'    {key} = {parsed_value}\n'

            sub_str += f'/ ! end of {inlist_type} namelist\n'

            inlist_string += sub_str

        return inlist_string

    def _copy_extra_files_and_dirs(self, grid_path):
        """
        Copy extra files and directories specified in ``MesaGrid.extra_files`` and ``MesaGrid.extra_dirs`` to each
        run directory.

        Args:
            grid_path:
        """
        for dirname in self.dirnames:
            dirpath = os.path.join(grid_path, dirname)
            for fpath in self.extra_files:
                fname = os.path.basename(fpath)
                shutil.copy2(os.path.abspath(fpath), os.path.join(dirpath, fname))
            for dpath in self.extra_dirs:
                dname = os.path.basename(dpath)
                shutil.copytree(os.path.abspath(dpath), os.path.join(dirpath, dname), dirs_exist_ok=True)

            # Also copy files specified in inlist_option_files_to_validate if they are missing in the run directory.
            for namelist, option in self.inlist_option_files_to_validate:
                if not option in self.__dict__[namelist]:
                    continue
                fpath = self.__dict__[namelist][option]
                if fpath == '':
                    pass
                else:
                    fname = os.path.basename(fpath)
                    if os.path.isfile(os.path.join(dirpath, fname)):
                        pass
                    else:
                        shutil.copy2(os.path.abspath(fpath), os.path.join(dirpath, fname))

            # Copy extra inlist files if specified.
            for namelist in self.namelists:
                for i in range(5):
                    i += 1
                    if i == self.inlists_index:  # Skip files generated by this class
                        continue
                    if self.extra_inlist_as_list:
                        read_extra_inlist_key = f'read_extra_{namelist}_inlist({i})'
                        read_extra_inlist_name_key = f'extra_{namelist}_inlist_name({i})'
                    else:
                        read_extra_inlist_key = f'read_extra_{namelist}_inlist{i}'
                        read_extra_inlist_name_key = f'extra_{namelist}_inlist{i}_name'

                    if read_extra_inlist_key in self.inlist[namelist].keys():
                        if self.inlist[namelist][read_extra_inlist_key]:
                            fname = os.path.basename(self.inlist[namelist][read_extra_inlist_name_key])
                            if fname not in [os.path.basename(p) for p in self.extra_files]:
                                shutil.copy2(os.path.abspath(self.inlist[namelist][read_extra_inlist_name_key]),
                                         os.path.join(dirpath, fname))


    def summary(self):
        """
        Print a summary of which variables change.
        """
        raise NotImplementedError()
        if not self.unpacked:
            self.unpack_inlists()

        for kind in self.namelists:
            pass
