#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This module contains various functions to load MESA and/or GYRE data.

Examples:
    Load a history file and the first profile:

    >>> import wsssss.load_data as ld
    >>> hist = ld.History('path/to/LOGS/history.data')
    >>> prof = ld.Profile('path/to/LOGS/profile1.data')
    >>> # or load all associated profiles:
    >>> profs = ld.load_profs(hist)

"""

import ast
import copy
import os
import shutil

import dill
import numpy as np

from . import functions as uf
from .constants import post15140
from .constants import pre15140


class _LazyProperty(object):
    def __init__(self, func):
        """
        Lazily load the attribute decorated by this class.
        Args:
            func: function
                Function which returns the property when called.
        """
        self._func = func
        self.__name__ = func.__name__
        self.__doc__ = func.__doc__

    def __get__(self, obj, klass=None):
        if obj is None:
            return None
        result = obj.__dict__[self.__name__] = self._func(obj)
        return result


class _Data:

    def __init__(self, path, keep_columns='all', save_dill=False, reload=False, verbose=False,
                 nanval=-1e99, nanclip=None, empty_on_error=False):
        """
        Common methods and attributes for History, Profile, and GyreSummary.

        Args:
            path (str): Path to the MESA history data file. If ending with `.dill`, will strip it and set that as `path`.
            keep_columns (list of str, optional): Which columns of the history data file to keep.
            save_dill (bool, optional): If True, will write a `.dill` file containing the `History` data.
            reload (bool, optional): If True, will ignore a pre-existing `.dill` file and reload from the history file.
            verbose (bool, optional): Print extra information.
            nanval (float, optional): Set all values equal to this to NaN.
            nanclip (2 floats, optional): Set all values outside this range to NaN.
        """
        self.path = os.path.abspath(path)
        self.directory = os.path.dirname(self.path)
        self.dill_only = False
        if os.path.isfile(self.path):
            self.fname = os.path.basename(self.path)
        elif os.path.isfile(self.path + '.dill') and not os.path.isfile(self.path):
            self.fname = os.path.basename(self.path)
            self.dill_only = True
        else:
            if empty_on_error:
                self.header = {}
                self.columns = []
                self.data = np.array([])
                self.loaded = False
                return
            else:
                raise FileNotFoundError(self.path)

        self.keep_columns = keep_columns
        self.save_dill = save_dill
        if self.path.endswith('.dill'):
            self.dill_path = path
        elif self.dill_only:
            self.dill_path = path + '.dill'
        else:
            self.dill_path = os.path.join(self.directory, self.fname + '.dill')

        self.loaded = False
        self.verbose = verbose
        self.nanval = nanval
        if nanclip is not None:
            self.nanclip = sorted(nanclip)
        else:
            self.nanclip = None

        if os.path.isfile(self.dill_path) and not reload:
            with open(self.dill_path, 'rb') as handle:
                try:
                    if not self.dill_only and (os.path.getmtime(self.dill_path) < os.path.getmtime(self.path)):
                        if self.verbose:
                            print('.dill file is older than loaded file! Reloading.')
                        self.save_dill = True
                        raise ValueError()
                    tmp = dill.load(handle)
                    self.header = tmp.header
                    self.columns = tmp.columns
                    self.data = tmp.data
                    self._first_row = tmp._first_row
                    if self.keep_columns != 'all':
                        self.columns = self.keep_columns
                        self.data = self._discard_columns_rec_array(self.data, self.columns)
                    self.loaded = True
                except Exception:
                    if self.verbose:
                        print("Failed to load dill from: \n{}".format(self.dill_path))
                    header, columns, first_row = self._read_data_file_header_columns()
                    self.header = header
                    self.columns = columns
                    self._first_row = first_row

        else:
            header, columns, first_row = self._read_data_file_header_columns()
            self.header = header
            self.columns = columns
            self._first_row = first_row

        if self.keep_columns != 'all':
            for col_keep in self.keep_columns:  # Check if columns present
                missing_cols = []
                if col_keep not in self.columns:
                    missing_cols.append(col_keep)
                if len(missing_cols) > 0:
                    raise ValueError(f'Columns in `keep_columns` not present in data file:\n'
                                     f'{" ".join(missing_cols)}')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, mask):
        new_self = copy.copy(self)
        new_self.data = self._discard_rows_rec_array(self.data, mask)
        return new_self

    def _read_data_file_header_columns(self):
        """
        Read a data file and return the header, columns, and first row of data.

        Returns:
            A tuple (header, columns, first_line), where header is a dict containing the header of the data file,
            columns is a list of column names, and first_line a np.rec.array containing the first row of data.
        """
        lines = []
        with open(self.path, 'r') as handle:
            lines.extend(handle.readline() for _ in range(7))

        if lines[0].strip() == '':
            lines[:3] = lines[1:4]
            lines[3] = '\n'

        header_columns = lines[1].split()
        header_data = []
        for item in lines[2].split():
            try:
                val = ast.literal_eval(item)
            except ValueError:
                val = item
            header_data.append(val)
        header = {k: v for k, v in zip(header_columns, header_data)}
        if 'version_number' in header.keys():
            header['version_number'] = str(header['version_number'])

        columns = lines[5].split()
        first_row = lines[6].split()

        formats = [np.array(ast.literal_eval(_)).dtype if _ != 'NaN' else np.float64 for _ in first_row]
        first_line = np.rec.array(first_row, dtype=list(zip(columns, formats)))

        return header, columns, first_line

    def _fix_datafile(self):
        """
        Check for corrupted data in history. This can happen if, for example, MESA was killed when writing to the history.

        Returns:
            good_lines (list of str): Lines which do not have unexpected data.
        """
        with open(f'{self.path}', 'rb') as handle:
            lines = handle.readlines()

        expected_len = len(lines[6])

        bad_lines = []
        good_lines = []
        for i, line in enumerate(lines):
            if line.startswith(b'\x00'):
                bad_lines.append(i)
                bad_lines.append(i + 1)  # Line after line with \x00\x00... is garbled
            else:
                if len(line) != expected_len and i > 6:
                    bad_lines.append(i)
                else:
                    good_lines.append(line)
        bad_lines = [i for i in bad_lines if i >= 6]  # skip header for bad lines
        bad_lines = np.unique(bad_lines)

        print(f'Removed {len(bad_lines)} lines:\n{bad_lines}')
        return good_lines

    def _read_data_file(self):
        """
        Read a data file and return the header, columns, and data.

        Returns:
            A tuple (header, columns, data), where header is a dict containing the header of the data file,
            columns is a list of column names, and data a np.rec.array containing the data.
        """
        lines = []
        with open(self.path, 'r') as handle:
            lines.extend(handle.readline() for _ in range(7))

        if lines[0].strip() == '':
            lines[:2] = lines[1:3]
            lines[2] = '\n'

        header_columns = lines[1].split()
        header_data = [ast.literal_eval(_) for _ in lines[2].split()]
        header = {k: v for k, v in zip(header_columns, header_data)}

        columns = lines[5].split()
        first_row = lines[6].split()

        formats = [np.array(ast.literal_eval(_)).dtype if _ != 'NaN' else np.float64 for _ in first_row]
        try:
            data = np.rec.array(np.loadtxt(self.path, skiprows=6, dtype=list(zip(columns, formats))))
        except ValueError as exc:
            print(f"File {self.path} gave ValueError when reading:\n{exc.args[0]}\nTrying to fix.")

            shutil.copy2(self.path, f'{self.path}_original')

            lines = self._fix_datafile()

            with open(f'{self.path}', 'wb') as handle:  #
                handle.writelines(lines)

            data = np.rec.array(np.loadtxt(f'{self.path}', skiprows=6, dtype=list(zip(columns, formats))))

        return header, columns, data

    def _discard_columns_rec_array(self, rec_array, to_keep):
        """
        Recreate a record array from `rec_array` keeping only columns `to_keep`, and discarding other columns.

        Args:
            rec_array (np.rec.array): Record array from which to discard columns.
            to_keep (list of str): Columns names to keep.

        Returns:
            np.rec.array: New ``np.rec.array`` without discarded columns.
        """
        columns, formats = np.array(rec_array.dtype.descr).T
        mask = (columns != '') & np.isin(columns, to_keep)
        columns = columns[mask]
        formats = formats[mask]
        return np.rec.array(rec_array[columns].tolist(), dtype=list(zip(columns, formats)))

    def _discard_rows_rec_array(self, rec_array, mask):
        """
        Recreate a rec array from `rec_array` keeping only rows masked in `mask`, and discarding other rows.

        Args:
            rec_array (np.rec.array):
            mask (np.array): Mask to apply to column(s).

        Returns:
            np.rec.array: New record array without discarded rows.
        """
        columns, formats = np.array(rec_array.dtype.descr).T
        return np.rec.array(rec_array[mask].tolist(), dtype=list(zip(columns, formats)))

    @_LazyProperty
    def data(self):
        """
        Data loading function. Sets the ``header`` and ``column_names`` attributes and returns the main data.

        Returns:
            np.rec.array: Record array of the main data.
        """
        header, columns, data = self._read_data_file()

        if self.nanclip is not None:
            for col in columns:
                if data[col].dtype == np.float64:
                    data[col][data[col] <= self.nanclip[0]] = np.nan
                    data[col][data[col] >= self.nanclip[1]] = np.nan

        if self.nanval is not None:
            for col in columns:
                if data[col].dtype == np.float64:
                    data[col][data[col] == self.nanval] = np.nan

        self.data = data
        if isinstance(self, History):
            data = self._scrub_hist()

        self.data = data
        self.loaded = True

        if self.save_dill:
            self.dump()

        if self.keep_columns != 'all':
            self.columns = self.keep_columns
            self.data = self._discard_columns_rec_array(self.data, self.columns)
            data = self.data

        return data

    def dump(self, path_to_dump=''):
        """
        Dump a ``_Data`` object to disk as a ``.dill`` file.

        Args:
            path_to_dump (str, optional): Path where to write the file. If ``''``, will write at ``self.dill_path``.
        """
        if not self.loaded:  # Temporarily turn off save_dill so the lazy load of data won't write a dill file.
            save_dill = self.save_dill
            self.save_dill = False
            _ = self.data
            self.save_dill = save_dill

        if path_to_dump == '':
            path_to_dump = self.dill_path

        with open(path_to_dump, 'wb') as handle:
            dill.dump(self, handle)

    def get(self, *args, mask=None):
        """
        Get a single or multiple columns from ``data``.

        Args:
            *args (str): Column name(s) to get.
            mask (np.array): Mask to apply to column(s).

        Returns:
            ``np.rec.array`` or list of ``np.rec.array``: Column(s) of ``data``.

        """
        if mask is None:
            mask = ...

        if len(args) == 1:
            return self.data[args[0]][mask]
        else:
            return [self.data[cname][mask] for cname in args]


class _Mesa(_Data):

    def __init__(self, path, index_name='profiles.index', keep_columns='all', save_dill=False, reload=False,
                 verbose=False, nanval=-1e99, nanclip=None, empty_on_error=False):
        """
        Methods specific to History and Profile.

        Args:
            path (str): Path to the MESA history data file. If ending with `.dill`, will strip it and set that as `path`.
            index_name (str, optional): Filename of the profile index.
            keep_columns (list of str, optional): Which columns of the history data file to keep.
            save_dill (bool, optional): If True, will write a `.dill` file containing the `History` data.
            reload (bool, optional): If True, will ignore a pre-existing `.dill` file and reload from the history file.
            verbose (bool, optional): Print extra information.
            nanval (float, optional): Set all values equal to this to NaN.
            nanclip (2 floats, optional): Set all values outside this range to NaN.
        """
        super().__init__(path, keep_columns, save_dill, reload, verbose, nanval, nanclip, empty_on_error)

        self.LOGS = self.directory
        if os.path.isfile(index_name):
            self.index_path = os.path.abspath(index_name)
        else:
            self.index_path = os.path.join(self.LOGS, index_name)

        try:
            index = np.genfromtxt(self.index_path, skip_header=1, dtype=int)
            if index.shape == (3,):
                index = index.reshape((1, 3))

            # Scrub index of backups and retries
            max_model = index[-1, 0]
            index = index[index[:, 0] <= max_model]
            if isinstance(self, History):
                min_model = self._first_row.model_number
                index = index[index[:, 0] >= min_model]
            u, i = np.unique(index[:, 0][::-1], return_index=True)
            index = index[::-1][i]
        except OSError:
            if self.verbose:
                print('Index file not found, expected path {}'.format(self.index_path))
            index = None

        self.index = index


class History(_Mesa):
    def __init__(self, path, index_name='profiles.index', keep_columns='all', save_dill=True, reload=False,
                 verbose=False, nanval=-1e99, nanclip=None, empty_on_error=False):
        """
        Load a MESA history.

        Args:
            path (str): Path to the MESA history data file. If ending with `.dill`, will strip it and set that as `path`.
            index_name (str, optional): Filename of the profile index.
            keep_columns (list of str, optional): Which columns of the history data file to keep.
            save_dill (bool, optional): If True, will write a `.dill` file containing the `History` data. This file will
             include all columns, even those removed by keep_columns.
            reload (bool, optional): If True, will ignore a pre-existing `.dill` file and reload from the history file.
            verbose (bool, optional): Print extra information.
            nanval (float, optional): Set all values equal to this to NaN.
            nanclip (2 floats, optional): Set all values outside this range to NaN.
            empty_on_error (bool, optional): If loading fails, return an empty History.
        """
        super().__init__(path, index_name, keep_columns, save_dill, reload, verbose, nanval, nanclip, empty_on_error)

    def __getitem__(self, mask):
        new_self = copy.copy(self)
        new_self.data = self._discard_rows_rec_array(self.data, mask)
        if hasattr(self, 'index'):
            if self.index is not None:
                mnum0, mnum1 = self.data.model_number[[0, -1]]
                idx_mask = (self.index[:, 0] >= mnum0) & (self.index[:, 0] <= mnum1)
                new_self.index = self.index[idx_mask]
        return new_self

    def __repr__(self):
        initial_model = self._first_row['model_number']
        initial_mass = self._first_row['star_mass']
        initial_age = self._first_row['star_age']
        star_info = f'Initial model={initial_model}, mass={initial_mass:.2f}, age={initial_age}'
        return 'MESA history data file at {}'.format(self.path) + '\n' + star_info

    def get_profile_num(self, model_num, method='closest', earlier=True):
        """
        Returns the `closest` (by default) or `previous` or `next` profile number for a given model number.
        If earlier is True, and there are two closest profiles, return the one with a lower model number.

        Args:
            model_num (int):
            method (str): Must be one of `closest` or `previous`.
            earlier (bool):
        Returns:
            A tuple (header, columns, first_line), where header is a dict containing the header of the data file,
            columns is a list of column names, and first_line a np.rec.array containing the first row of data.
        Returns:
            A tuple (pnum, pmod, hist_ind), where pnum is the profile number, pmod is the model number of Profile, and
            hist_ind the index in History of Profile.
        """

        if method == 'closest':
            model_diff = np.abs(self.index[:, 0] - model_num)
            minval = min(model_diff)

            ind = np.where(model_diff == minval)[0]
            if len(ind) == 2:
                if earlier:
                    ind = ind[0]
                else:
                    ind = ind[1]
            else:
                ind = ind[0]
        elif method == 'previous':
            model_diff = self.index[:, 0] - model_num
            ind = np.where(model_diff <= 0)[0][-1]
        elif method == 'next':
            model_diff = self.index[:, 0] - model_num
            ind = np.where(model_diff >= 0)[0][-1]
        else:
            raise ValueError("method must be 'closest' or 'previous'.")
        pmod, _, pnum = self.index[ind]
        m_min, m_max = self.get('model_number')[[0, -1]]
        if (m_min <= pmod) and (m_max >= pmod):
            hist_ind = np.where(self.get('model_number') == pmod)[0][0]
        else:
            hist_ind = []
        return pnum, pmod, hist_ind


    def get_profile_index(self, profile_nums):
        """
        Returns the corresponding indeces of `profile_nums`.
        `profile_nums` can be an integer, a list of integers,
        a `Profile`, or a list of `Profile`\ s.

        Args:
            profile_nums (int or list of int): Profile numbers for which to calculate the indeces.

        Returns:
            int or np.array of int: Indeces of profile_nums in History.

        """
        if hasattr(profile_nums, '__len__'):
            if isinstance(profile_nums, Profile):
                profile_nums = [profile_nums]
        else:
            profile_nums = [profile_nums]
        if isinstance(profile_nums[0], Profile):
            profile_nums = [p.profile_num for p in profile_nums]
        idxs = np.where(np.isin(self.index[:, 2], profile_nums))
        model_nums = self.index[:, 0][idxs]
        data_idx = np.where(np.isin(self.data['model_number'], model_nums))[0]
        return data_idx

    def _scrub_hist(self):
        """
        Scrub history data for backups and retries.
        """
        max_model = self.data['model_number'][-1]
        scrubbed = self.data[self.data['model_number'] <= max_model]

        u, i = np.unique(scrubbed['model_number'][::-1], return_index=True)
        scrubbed = scrubbed[::-1][i]

        return scrubbed


class Profile(_Mesa):
    def __init__(self, path, index_name='profiles.index', keep_columns='all', load_GyreProfile=False,
                 suffix_GyreProfile='.GYRE', save_dill=False, reload=False, verbose=False, nanval=-1e99,
                 nanclip=None):
        """
        Load a MESA profile.

        Args:
            path (str): Path to the MESA profile data file. If ending with `.dill`, will strip it and set that as `path`.
            index_name (str, optional): Filename of the profile index.
            keep_columns (list of str, optional): Which columns of the history data file to keep.
            load_GyreProfile (bool): If True, will also load the corresponding GyreProfile into Profile.GyreProfile.
            suffix_GyreProfile (str, optional): Suffix of GyreProfile. Defaults to '.GYRE'.
            save_dill (bool, optional): If True, will write a `.dill` file containing the `History` data.
            reload (bool, optional): If True, will ignore a pre-existing `.dill` file and reload from the history file.
            verbose (bool, optional): Print extra information.
            nanval (float, optional): Set all values equal to this to NaN.
            nanclip (2 floats, optional): Set all values outside this range to NaN.
        """
        super().__init__(path, index_name, keep_columns, save_dill, reload, verbose, nanval, nanclip)
        self.LOGS = self.directory

        if self.index is not None:
            self.profile_num = self.index[np.where(self.index == self.header['model_number'])[0][0]][2]
        else:
            self.profile_num = None
        if load_GyreProfile:
            self.GyreProfile = GyreProfile(f'{self.path}{suffix_GyreProfile}')
        else:
            self.GyreProfile = None

    def __repr__(self):
        try_to_get = ['model_number', 'num_zones', 'star_mass', 'star_age', 'Teff', 'photosphere_L',
                      'center_h1', 'center_he4', 'date']
        return 'MESA profile data file at {}'.format(self.path) + '\n' + str(
            {key: self.header[key] for key in try_to_get if key in self.header.keys()})

    def get_hist_index(self, hist):
        """
        Get the index of this profile in the `History` hist.

        Args:
            hist (History):

        Returns:
            int: Index of profile in hist.
        """
        mod = self.header['model_number']
        return np.argwhere(hist.get('model_number') == mod)[0][0]


class _Gyre(_Data):

    def __init__(self, path, keep_columns='all', gyre_version='7', save_dill=False, reload=False, verbose=False,
                 nanval=-1e99, nanclip=None):
        """
        Common methods and attributes for GyreSummary and GyreMode.

        Args:
            path (str): Path to the Gyre summary or mode data file. If ending with `.dill`, will strip it and set that as `path`.
            keep_columns (list of str, optional): Which columns of the history data file to keep.
            gyre_version (str): Gyre version.
            save_dill (bool, optional): If True, will write a `.dill` file containing the `History` data.
            reload (bool, optional): If True, will ignore a pre-existing `.dill` file and reload from the history file.
            verbose (bool, optional): Print extra information.
            nanval (float, optional): Set all values equal to this to NaN.
            nanclip (2 floats, optional): Set all values outside this range to NaN.
        """
        super().__init__(path, keep_columns, save_dill, reload, verbose, nanval, nanclip)
        self.gyre_version = gyre_version


class GyreSummary(_Gyre):

    def __init__(self, path, keep_columns='all', gyre_version='7', save_dill=False, reload=False, verbose=False,
                 nanval=-1e99, nanclip=None):
        """
        Gyre summary output.

        Args:
            path (str): Path to the Gyre summary data file. If ending with `.dill`, will strip it and set that as `path`.
            keep_columns (list of str, optional): Which columns of the history data file to keep.
            gyre_version (str): Gyre version.
            save_dill (bool, optional): If True, will write a `.dill` file containing the `History` data.
            reload (bool, optional): If True, will ignore a pre-existing `.dill` file and reload from the history file.
            verbose (bool, optional): Print extra information.
            nanval (float, optional): Set all values equal to this to NaN.
            nanclip (2 floats, optional): Set all values outside this range to NaN.
        """
        super().__init__(path, keep_columns, gyre_version, save_dill, reload, verbose, nanval, nanclip)

    def __repr__(self):
        return f'GyreSummary at {self.path}'

    def _dimless_to_Hz(self):
        """
        Returns:
            float: Conversion factor between dimensionless frequency and Hz.

        """
        if 'M_star' in self.header.keys():
            M_star = self.header['M_star']
            R_star = self.header['R_star']
            G = pre15140.standard_cgrav
        else:
            M_star = self.get('M_star')[0]
            R_star = self.get('R_star')[0]
            G = post15140.standard_cgrav  # This changed in version 6.
        return 1.0 / (2 * np.pi) * ((G * M_star / (R_star) ** 3))

    def get_frequencies(self, freq_units, Re_freq_unit='uHz'):
        """
        Get frequencies in the specicied units. Will use 'Re(omega)' first and 'Re(freq)' otherwise.
        Args:
            freq_units (str): Unit to convert to, must be one of 'uHz', 'mHz', or 'Hz'.
            Re_freq_unit (str, optional): Unit of the 'Re(freq)' column, must be one of 'uHz', 'mHz', or 'Hz'.

        Returns:
            np.rec.array: Frequencies in unit specified by Re_freq_unit.
        """
        unit_dict = {'uHz': 1e6, 'mHz': 1e3, 'Hz': 1e0}
        if 'freq_units' in self.header.keys():
            header_freq_units = self.header['freq_units'].lower().replace('hz', 'Hz')
            if header_freq_units != Re_freq_unit:
                print(f'Warning! Frequency units in header ({header_freq_units}) and Re_freq_unit ({Re_freq_unit}) are '
                      f'not the same, using header frequency units.')
                Re_freq_unit = header_freq_units

        if 'Re(omega)' in self.columns:
            dimless_to_Hz = self._calc_dimless_to_Hz() * unit_dict[freq_units]
            freq_name = 'Re(omega)'
        elif 'Re(freq)' in self.columns:  # Assumes freq already in uHz.
            dimless_to_Hz = unit_dict[freq_units] / unit_dict[Re_freq_unit]
            freq_name = 'Re(freq)'
        return self.data[freq_name] * dimless_to_Hz


class GyreMode(_Gyre):

    def __init__(self, path, keep_columns='all', gyre_version='7', save_dill=False, reload=False, verbose=False,
                 nanval=-1e99, nanclip=None):
        super().__init__(path, keep_columns, gyre_version, save_dill, reload, verbose, nanval, nanclip)
        """
        Gyre mode detail file.

        Args:
            path (str): Path to the Gyre mode data file. If ending with `.dill`, will strip it and set that as `path`.
            keep_columns (list of str, optional): Which columns of the `GyreMode` data file to keep.
            gyre_version (str): Gyre version.
            save_dill (bool, optional): If True, will write a `.dill` file containing the `GyreMode` data.
            reload (bool, optional): If True, will ignore a pre-existing `.dill` file and reload from the mode file.
            verbose (bool, optional): Print extra information.
            nanval (float, optional): Set all values equal to this to NaN.
            nanclip (2 floats, optional): Set all values outside this range to NaN.
        """

    def __repr__(self):
        return f'GyreMode at {self.path}'

    def _dimless_to_Hz(self):
        """
        Returns:
            float: Conversion factor between dimensionless frequency and Hz.

        """
        M_star = self.header['M_star']
        R_star = self.header['R_star']
        if self.gyre_version < '6':
            G = pre15140.standard_cgrav
        else:
            G = post15140.standard_cgrav
        return 1.0 / (2 * np.pi) * ((G * M_star / (R_star) ** 3))

    def get_frequencies(self, freq_units):
        """
        Get frequencies in the specicied units. Will use 'Re(omega)' first and 'Re(freq)' otherwise.
        Args:
            freq_units (str): Unit to convert to, must be one of 'uHz', 'mHz', or 'Hz'.
            Re_freq_unit (str, optional): Unit of the 'Re(freq)' column, must be one of 'uHz', 'mHz', or 'Hz'.

        Returns:
            np.rec.array: Frequencies in unit specified by Re_freq_unit.
        """
        unit_dict = {'uHz': 1e6, 'mHz': 1e3, 'Hz': 1e0}
        if 'Re(omega)' in self.header:
            dimless_to_Hz = self._calc_dimless_to_Hz() * unit_dict[freq_units]
            freq_name = 'Re(omega)'
        elif 'Re(freq)' in self.header:  # Assumes freq already in uHz.
            dimless_to_Hz = unit_dict[freq_units] / unit_dict[Re_freq_unit]
            freq_name = 'Re(freq)'
        return self.data[freq_name] * dimless_to_Hz


class GyreProfile:
    def __init__(self, path):
        """
        Gyre profile created by MESA.

        Args:
            path: Path to gyre profile file.
        """
        self.path = path
        if os.path.isfile(path):
            self.fname = os.path.basename(self.path)

        num_zones, mass, radius, luminosity, version = np.loadtxt(f'{path}', max_rows=1)
        num_zones = int(num_zones)
        self.version = int(version)

        self.header = {'num_zones': num_zones, 'star_mass': mass, 'star_radius': radius, 'star_luminosity': luminosity,
                       'version': version}
        if self.version == 100:
            self.columns = ['zone', 'radius', 'mass', 'luminosity', 'pressure', 'temperature', 'density', 'grad_T',
                            'brunt_N2', 'gamma1', 'grad_ad', 'nu_T', 'opacity', 'opacity_partial_T',
                            'opacity_partial_rho', 'total_energy_generation', 'nuclear_energy_generation_partial_T',
                            'nuclear_energy_generation_partial_rho', 'rotation']

            self.formats = [int] + 18 * [float]
        elif self.version == 101:
            self.columns = ['zone', 'radius', 'mass', 'luminosity', 'pressure', 'temperature', 'density', 'grad_T',
                            'brunt_N2', 'gamma1', 'grad_ad', 'nu_T', 'opacity', 'opacity_partial_T',
                            'opacity_partial_rho', 'nuclear_energy_generation', 'nuclear_energy_generation_partial_T',
                            'nuclear_energy_generation_partial_rho', 'rotation']

            self.formats = [int] + 18 * [float]
        elif self.version == 120:
            self.columns = ['zone', 'radius', 'mass', 'luminosity', 'pressure', 'temperature', 'density', 'grad_T',
                            'brunt_N2', 'gamma1', 'grad_ad', 'nu_T', 'opacity', 'opacity_partial_T',
                            'opacity_partial_rho', 'nuclear_energy_generation', 'nuclear_energy_generation_partial_T',
                            'nuclear_energy_generation_partial_rho', 'gravothermal_energy_generation', 'rotation']
            self.formats = [int] + 19 * [float]
        else:
            raise NotImplementedError('Only fileversions 100, 101, and 120 are implemented implemented.')

        self.loaded = False

    def __repr__(self):
        return 'GyreProfile data file at {}'.format(self.path) + '\n' + str(self.header)

    def _load_gyre_profile(self):
        data = np.rec.array(
            np.loadtxt(f'{self.path}', skiprows=1, dtype=list(zip(self.columns, self.formats))))
        return data

    @_LazyProperty
    def data(self):
        data = self._load_gyre_profile()
        self.loaded = True
        return data


def load_profs(hist, prefix='profile', suffix='.data', save_dill=False, mask=None, mask_kwargs=None):
    """
    Load profiles associated with `History` hist.

    Args:
        hist (History):
        prefix (str, optional): Part of profile name before the profile number. Defaults to 'profile'.
        suffix (str, optional): Part of profile name after the profile number. Defaults to '.data'.
        save_dill (bool, optional): If True, will write a `.dill` file containing the `History` data.
        mask (array of bool or function): Mask to apply to column(s).
        mask_kwargs (dict, optional): kwargs to pass to mask if it is a function.

    Returns:
        list of Profile: All profiles in hist.index or those which satisfy mask.
    """
    if hist.index is None:
        return []
    pnums = hist.index[:, 2]

    if mask is not None:
        if hasattr(mask, '__call__'):
            if mask_kwargs is None:
                mask_kwargs = {}
            mask = mask(hist, **mask_kwargs)
        valid_mod = hist.get('model_number')[mask]
        pnums = hist.index[:, 2][np.isin(hist.index[:, 0], valid_mod)]

    profs = []
    for i in pnums:

        try:
            if suffix.endswith('.GYRE'):
                prof = GyreProfile(os.path.join(hist.LOGS, '{}{}{}'.format(prefix, i, suffix)))
                profs.append(prof)
            else:
                prof = Profile(os.path.join(hist.LOGS, '{}{}{}'.format(prefix, i, suffix)), save_dill=save_dill)
                profs.append(prof)
        except IndexError:
            print(os.path.join(hist.LOGS, '{}{}{}'.format(prefix, i, suffix)) +
                  ' not found in {}'.format(hist.index_path))
    return profs


def load_gss(hist, gyre_data_dir='gyre_out', gyre_summary_prefix='profile', gyre_summary_suffix='.data.GYRE.sgyre_l',
             return_pnums=False, use_mask=None, keep_columns='all', gyre_version='7', save_dill=False, reload=False, verbose=False, nanval=-1e99,
                         nanclip=None):
    """
    Load `GyreSummary` associated with `History` hist.

    Args:
        hist:
        gyre_data_dir: Directory containing gyre summary files.
        gyre_summary_prefix (str, optional): Part of gyre summary name before the profile number. Defaults to 'profile'.
        gyre_summary_suffix (str, optional): Part of gyre summary name after the profile number. Defaults to '.data.GYRE.sgyre_l'.
        return_pnums (bool, optional): Defaults to False. If True, will also return profile numbers.
        use_mask (bool, np.array, or function, optional): If True, will exclude pre-main sequence.
                If an array of bools will use that as mask. If a function, will call function(hist) and use that as the mask.
        keep_columns (list of str, optional): Which columns of the history data file to keep.
        gyre_version (str): Gyre version.
        save_dill (bool, optional): If True, will write a `.dill` file containing the `GyreSummary` data.
        reload (bool, optional): If True, will ignore a pre-existing `.dill` file and reload from the gyre summary file.
        verbose (bool, optional): Print extra information.
        nanval (float, optional): Set all values equal to this to NaN.
        nanclip (2 floats, optional): Set all values outside this range to NaN.

    Returns:
        list of GyreSummary or list of list of GyreSummary: If return_pnums is False returns only `GyreSummary`.
            If return_pnums is True also return profile numbers.
    """
    dirpath = os.path.abspath(os.path.join(hist.LOGS, '..', gyre_data_dir))

    use_mask = uf.get_mask(hist, use_mask)

    min_mod, max_mod = hist.get('model_number')[use_mask][[0, -1]]

    pnums = []
    for fname in os.listdir(dirpath):
        fname = os.path.split(fname)[-1]
        if fname.startswith(gyre_summary_prefix) and fname.endswith(gyre_summary_suffix):
            try:
                pnum = int(fname[len(gyre_summary_prefix):-len(gyre_summary_suffix)])
            except ValueError:
                continue
            if pnum in hist.index[:, 2]:  # Only load gss which are in the index
                if min_mod <= hist.index[:, 0][hist.index[:, 2] == pnum] <= max_mod:
                    pnums.append(pnum)
    pnums.sort()

    gss = []
    for pnum in pnums:
        fname = '{}{}{}'.format(gyre_summary_prefix, pnum, gyre_summary_suffix)
        path = os.path.join(dirpath, fname)
        gss.append(GyreSummary(path, keep_columns=keep_columns, gyre_version=gyre_version, save_dill=save_dill, reload=reload,
                     verbose=verbose, nanval=nanval, nanclip=nanclip))

    if return_pnums:
        return list(zip(gss, np.array(pnums)))
    return gss


def load_modes_from_profile(prof, gyre_data_dir='gyre_out', mode_prefix='', mode_suffix='.mgyre', keep_columns='all',
                            gyre_version='7', save_dill=False, reload=False, verbose=False, nanval=-1e99, nanclip=None):
    """
    Load all `GyreMode` associated with `Profile` prof.

    Args:
        prof:
        gyre_data_dir: Directory containing gyre summary files.
        mode_prefix (str, optional): First part of the mode filename.
            If an empty string will use the profile name. Defaults to ''.
        mode_suffix (str, optional): Last part of the gyre mode filename. Defaults to '.mgyre'.
        keep_columns (list of str, optional): Which columns of the mode data file to keep.
        gyre_version (str): Gyre version.
        save_dill (bool, optional): If True, will write a `.dill` file containing the `GyreMode` data.
        reload (bool, optional): If True, will ignore a pre-existing `.dill` file and reload from the mode file.
        verbose (bool, optional): Print extra information.
        nanval (float, optional): Set all values equal to this to NaN.
        nanclip (2 floats, optional): Set all values outside this range to NaN.


    Returns:
        list of GyreMode:
    """
    dirpath = os.path.abspath(os.path.join(prof.LOGS, '..', gyre_data_dir))
    fnames = os.listdir(dirpath)

    if mode_prefix == '':
        mode_prefix = prof.fname

    modes = []
    for fname in fnames:
        if fname.startswith(mode_prefix):
            if fname.endswith(mode_suffix):
                path = os.path.join(dirpath, fname)
                modes.append(GyreMode(path, keep_columns=keep_columns, gyre_version=gyre_version, save_dill=save_dill, reload=reload, verbose=verbose, nanval=nanval, nanclip=nanclip))
    return modes


def load_gs_from_profile(prof, gyre_data_dir='gyre_out', gyre_summary_prefix='', gyre_summary_suffix='.data.GYRE.sgyre_l',
                         keep_columns='all', gyre_version='7', save_dill=False, reload=False, verbose=False, nanval=-1e99,
                         nanclip=None):
    """
    Load the `GyreSummary` associated with `Profile` prof.

    Args:
        prof (Profile):
        gyre_data_dir: Directory containing gyre summary files.
        gyre_summary_prefix (str, optional): First part of gyre summary name. First part of the gyre summary filename.
            If an empty string will use the profile name. Defaults to ''.
        gyre_summary_suffix (str, optional): Part of gyre summary name after the profile number. Defaults to '.data.GYRE.sgyre_l'.
        keep_columns (list of str, optional): Which columns of the history data file to keep.
        gyre_version (str): Gyre version.
        save_dill (bool, optional): If True, will write a `.dill` file containing the `History` data.
        reload (bool, optional): If True, will ignore a pre-existing `.dill` file and reload from the history file.
        verbose (bool, optional): Print extra information.
        nanval (float, optional): Set all values equal to this to NaN.
        nanclip (2 floats, optional): Set all values outside this range to NaN.
    Returns:
        GyreSummary:
    """
    dirpath = os.path.abspath(os.path.join(prof.LOGS, '..', gyre_data_dir))
    fnames = os.listdir(dirpath)

    if gyre_summary_prefix == '':
        gyre_summary_prefix = prof.fname

    for fname in fnames:
        if fname.startswith(gyre_summary_prefix) and fname.endswith(gyre_summary_suffix):
            break
        else:
            fname = ''

    path = os.path.join(dirpath, fname)
    gs = GyreSummary(path, keep_columns=keep_columns, gyre_version=gyre_version, save_dill=save_dill, reload=reload,
                     verbose=verbose, nanval=nanval, nanclip=nanclip)
    return gs


# def load_modes(hist, gyre_summary_dir='gyre_out', gyre_summary_prefix='profile', gyre_summary_suffix='.data.GYRE.sgyre_l',
#                mode_dir='gyre_out/detail', mode_prefix='profile{}.', mode_suffix='.mgyre', keep_columns='all',
#                gyre_version='7', save_dill=False, reload=False, verbose=False, nanval=-1e99, nanclip=None):
#
#     gss, pnums = load_gss(hist, gyre_summary_dir, gyre_summary_prefix, gyre_summary_suffix, return_pnums=True,
#                           keep_columns=keep_columns, gyre_version=gyre_version, save_dill=save_dill, reload=reload,
#                           verbose=verbose, nanval=nanval, nanclip=nanclip)
#
#     dirpath = os.path.abspath(os.path.join(hist.LOGS, '..', mode_dir))
#     fnames = os.listdir(dirpath)
#
#     mode_info = []
#     modes = []
#
#     i = 0
#     for fname in fnames:
#         fname = os.path.split(fname)[-1]
#         prefix_parts = mode_prefix.split('{}')
#         pnum = int(fname[len(prefix_parts[0]):fname.index(prefix_parts[1])])
#         mode_prefix_pnum = mode_prefix.format(pnum)
#         if fname.startswith(mode_prefix_pnum) and fname.endswith(mode_suffix):
#             path = os.path.join(dirpath, fname)
#             md = GyreMode(path, keep_columns=keep_columns, gyre_version=gyre_version, save_dill=save_dill, reload=reload,
#                           verbose=verbose, nanval=nanval, nanclip=nanclip)
#             nu = uf.get_freq(md, kind='mode')
#             l = md.header['l']
#             order_names = ['n_p', 'n_g', 'n_pg']
#             if sum(np.isin(order_names, list(md.header.keys()))) > 1:
#
#                 try:
#                     n_p = md.header['n_p']
#                 except KeyError:
#                     n_p = md.header['n_pg'] + md.header['n_g']
#                     md.header['n_p'] = n_p
#
#                 try:
#                     n_g = md.header['n_g']
#                 except KeyError:
#                     n_g = md.header['n_p'] - md.header['n_pg']
#                     md.header['n_g'] = n_g
#
#                 try:
#                     n_pg = md.header['n_pg']
#                 except KeyError:
#                     n_pg = md.header['n_p'] - md.header['n_g']
#                     md.header['n_pg'] = n_pg
#
#             else:
#                 raise ValueError('Not enough information in header to determine n_p, n_g, and n_pg.')
#
#             md.header['profile_number'] = pnum
#
#             for key in ['n_p', 'n_g', 'n_pg']:
#                 if key in md.header and key in locals().keys():
#                     if md.header[key] != locals()[key]:
#                         raise ValueError(f'Mismatch between mode {key} and calculated {key}!')
#
#                 else:
#                     md.header[key] = locals()[key]
#
#             mode_info.append([pnum, l, nu, n_pg, n_p, n_g, i])
#             modes.append(md)
#             i += 1
#
#     mode_info = np.rec.array(mode_info, names=['pnum', 'l', 'nu', 'n_pg', 'n_p', 'n_g', 'i'])
#     mode_info = np.sort(mode_info, order=['pnum', 'l', 'nu', 'n_pg'])
#     modes = np.asarray([modes[i] for i in mode_info.i], dtype=object)
#
#     mode_info.i = np.arange(len(mode_info))
#
#     return modes


def naive_merge_hists(base_hist, hists):
    """
    Merge two `History` objects. This function simply stacks the history data onto a new copy of base_hist.

    Args:
        base_hist (History):
        hists (list of History): Histories to stack.

    Returns:
        History
    """
    new_hist = copy.copy(base_hist)
    new_hist.data = np.lib.recfunctions.stack_arrays([h.data for h in hists], asrecarray=True, usemask=False)
    return new_hist


def load_gss_to_hist(hist, gyre_data_dir='gyre_out', gyre_summary_prefix='profile',
                     gyre_summary_suffix='.data.GYRE.sgyre_l', use_mask=None, keep_columns='all',
                     gyre_version='7', save_dill=False, reload=False, verbose=False, nanval=-1e99, nanclip=None):
    """
    Load ``GyreSummary`` and profile numbers associated with ``History`` hist and place in the attribute ``History.gsspnum``.
    This is equivalent to doing ``hist.gsspnum = load_gss(..., return_pnums=True, ...)``.

    Args:
        hist:
        gyre_data_dir: Directory containing gyre summary files.
        gyre_summary_prefix (str, optional): Part of gyre summary name before the profile number. Defaults to 'profile'.
        gyre_summary_suffix (str, optional): Part of gyre summary name after the profile number. Defaults to '.data.GYRE.sgyre_l'.
        use_mask (bool, np.array, or function, optional): If True, will exclude pre-main sequence.
                If an array of bools will use that as mask. If a function, will call function(hist) and use that as the mask.
        keep_columns (list of str, optional): Which columns of the history data file to keep.
        gyre_version (str): Gyre version.
        save_dill (bool, optional): If True, will write a `.dill` file containing the `GyreSummary` data.
        reload (bool, optional): If True, will ignore a pre-existing `.dill` file and reload from the gyre summary file.
        verbose (bool, optional): Print extra information.
        nanval (float, optional): Set all values equal to this to NaN.
        nanclip (2 floats, optional): Set all values outside this range to NaN.

    Returns:
        list of GyreSummary or list of list of GyreSummary: If return_pnums is False returns only `GyreSummary`.
            If return_pnums is True also return profile numbers.
    """
    hist.gsspnum = load_gss(hist, gyre_data_dir=gyre_data_dir, gyre_summary_prefix=gyre_summary_prefix,
                            gyre_summary_suffix=gyre_summary_suffix, return_pnums=True, use_mask=use_mask,
                            keep_columns=keep_columns, gyre_version=gyre_version, save_dill=save_dill, reload=reload,
                            verbose=verbose, nanval=nanval, nanclip=nanclip)
    return hist
