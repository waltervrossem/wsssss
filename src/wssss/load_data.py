#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import os
import shutil

import dill
import numpy as np

from . import functions as uf
from .constants import post15140
from .constants import pre15140
import ast


def _read_data_file_header_columns(path):
    """Read a mesa history or profile file and return the header and columns."""
    lines = []
    with open(path, 'r') as handle:
        lines.extend(handle.readline() for _ in range(7))

    if lines[0].strip() == '':
        lines[:3] = lines[1:4]
        lines[3] = '\n'

    header_columns = lines[1].split()
    header_data = [ast.literal_eval(_) for _ in lines[2].split()]
    header = {k: v for k, v in zip(header_columns, header_data)}

    columns = lines[5].split()
    first_row = lines[6].split()

    formats = [np.array(ast.literal_eval(_)).dtype if _ != 'NaN' else np.float64 for _ in first_row]
    first_line = np.rec.array(first_row, dtype={'names': columns, 'formats': formats})

    return header, columns, first_line


def _fix_history(path):
    with open(f'{path}', 'rb') as handle:
        lines = handle.readlines()
    
    expected_len = len(lines[6])
    
    bad_lines = []
    good_lines = []
    for i, line in enumerate(lines):
        if line.startswith(b'\x00'):
            bad_lines.append(i)
            bad_lines.append(i + 1)  # Line after line with \x00\x00... is garbled
        else:
            if len(line) != expected_len:
                bad_lines.append(i)
            else:
                good_lines.append(line)
    bad_lines = [i for i in bad_lines if i >= 6]  # skip header for bad lines
    bad_lines = np.unique(bad_lines)
    
    # for bad_line in bad_lines[::-1]:  # work back to front
    #     _ = lines.pop(bad_line)
    print(f'Removed {len(bad_lines)} lines:\n{bad_lines}')
    return good_lines

# noinspection PyTypeChecker
def _read_data_file(path):
    """Read a mesa history or profile file and return the header, columns, and data."""
    lines = []
    with open(path, 'r') as handle:
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
        data = np.rec.array(np.loadtxt(path, skiprows=6, dtype={'names': columns, 'formats': formats}))
    except ValueError as exc:
        print(f"File {path} gave ValueError when reading:\n{exc.args[0]}\nTrying to fix.")
        
        shutil.copy2(path, f'{path}_original')
        
        lines = _fix_history(path)
        
        with open(f'{path}_test', 'wb') as handle:#
            handle.writelines(lines)
        
        data = np.rec.array(np.loadtxt(f'{path}_test', skiprows=6, dtype={'names': columns, 'formats': formats}))

    return header, columns, data


def _discard_columns_rec_array(rec_array, to_keep):
    """Recreate a rec array from `rec_array` keeping only columns `to_keep`, and discarding other columns."""
    names, formats = np.array(rec_array.dtype.descr).T
    mask = (names != '') & np.in1d(names, to_keep)
    names = names[mask]
    formats = formats[mask]
    return np.rec.array(rec_array[names].tolist(), dtype=dict(names=names, formats=formats))


def _discard_rows_rec_array(rec_array, mask):
    """Recreate a rec array from `rec_array` keeping only rows masked in `mask`, and discarding other rows."""
    names, formats = np.array(rec_array.dtype.descr).T
    return np.rec.array(rec_array[mask].tolist(), dtype=dict(names=names, formats=formats))


class LazyProperty(object):
    def __init__(self, func):
        self._func = func
        self.__name__ = func.__name__
        self.__doc__ = func.__doc__

    def __get__(self, obj, klass=None):
        if obj is None:
            return None
        result = obj.__dict__[self.__name__] = self._func(obj)
        return result


class _Data:
    """Common methods and attributes for History, Profile, and GyreSummary."""
    def __init__(self, path, keep_columns='all', save_dill=False, reload=False, verbose=False,
                 nanval=-1e99, nanclip=None):
        self.path = os.path.abspath(path)
        if os.path.isfile(self.path):
            self.fname = os.path.basename(self.path)
        else:
            raise FileNotFoundError(self.path)
        self.directory = os.path.dirname(self.path)

        self.keep_columns = keep_columns
        self.save_dill = save_dill
        if self.path.endswith('.dill'):
            self.dill_path = path
        else:
            self.dill_path = os.path.join(self.directory, self.fname+'.dill')

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
                    if os.path.getmtime(self.dill_path) < os.path.getmtime(self.path):
                        if self.verbose:
                            print('.dill file is older than loaded file! Reloading.')
                        self.save_dill = True
                        raise ValueError()
                    tmp = dill.load(handle)
                    self.header = tmp.header
                    self.columns = tmp.columns
                    self.data = tmp.data
                    self._first_row = tmp.first_row
                    if self.keep_columns != 'all':
                        self.columns = self.keep_columns
                        self.data = _discard_columns_rec_array(self.data, self.columns)
                    self.loaded = True
                except Exception:
                    if self.verbose:
                        print("Failed to load dill from: \n{}".format(self.dill_path))
                    header, columns, first_row = _read_data_file_header_columns(self.path)
                    self.header = header
                    self.columns = columns
                    self._first_row = first_row

        else:
            header, columns, first_row = _read_data_file_header_columns(self.path)
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
        # new_self.data = self.data[mask]
        new_self.data = _discard_rows_rec_array(self.data, mask)
        mnum0, mnum1 = self.data.model_number[[0,-1]]
        if self.index is not None:
            idx_mask = (self.index[:,0] >= mnum0) & (self.index[:,0] <= mnum1)
            new_self.index = self.index[idx_mask]
        return new_self


    @LazyProperty
    def data(self):
        header, columns, data = _read_data_file(self.path)

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
            with open(self.dill_path, 'wb') as handle:
                dill.dump(self, handle)

        if self.keep_columns != 'all':
            self.columns = self.keep_columns
            self.data = _discard_columns_rec_array(self.data, self.columns)
            data = self.data

        return data


    def dump(self, path_to_dump):
        with open(path_to_dump, 'wb') as handle:
            dill.dump(self, handle)

    def get(self, *args, **kwargs):
        if 'mask' not in kwargs:
            mask = ...
        else:
            if kwargs['mask'] is None:
                mask = ...
            else:
                mask = kwargs['mask']

        if len(args) == 1:
            return self.data[args[0]][mask]
        else:
            return [self.data[cname][mask] for cname in args]


class _Mesa(_Data):
    def __init__(self, *args, index_name='profiles.index', **kwargs):
        super().__init__(*args, **kwargs)
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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __repr__(self):
        initial_model = self._first_row['model_number']
        initial_mass = self._first_row['star_mass']
        initial_age = self._first_row['star_age']
        star_info = f'Initial model={initial_model}, mass={initial_mass:.2f}, age={initial_age}'
        return 'MESA history data file at {}'.format(self.path) + '\n' + star_info

    def get_profile_num(self, model_num, method='closest', earlier=True):
        """Returns the `closest` (by default) or `previous` or `next` profile number for a given model number.
        If earlier is True, and there are two closest profiles, return the one with a lower model number. """

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

    def get_model_num(self, profile_nums):
        """Returns the profile number, model number, and model index"""
        if hasattr(profile_nums, '__len__'):
            if isinstance(profile_nums, Profile):
                profile_nums = [profile_nums]
        else:
            profile_nums = [profile_nums]
        if isinstance(profile_nums[0], Profile):
            profile_nums = [p.profile_num for p in profile_nums]
        idxs = np.where(np.in1d(self.index[:, 2], profile_nums))
        model_nums = self.index[:, 0][idxs]
        data_idx = np.where(np.in1d(self.data['model_number'], model_nums))[0]
        return np.array(list(zip(profile_nums, model_nums, data_idx)), dtype=int)


    def _scrub_hist(self):
        """Scrub history data for backups and retries."""
        max_model = self.data['model_number'][-1]
        scrubbed = self.data[self.data['model_number'] <= max_model]

        u, i = np.unique(scrubbed['model_number'][::-1], return_index=True)
        scrubbed = scrubbed[::-1][i]

        return scrubbed

class Profile(_Mesa):
    def __init__(self, *args, load_GyreProfile=False, suffix_GyreProfile='.GYRE', **kwargs):
        super().__init__(*args, **kwargs)
        self.LOGS = self.directory

        if self.index is not None:
            self.profile_num = self.index[np.where(self.index == self.header['model_number'])[0][0]][2]
        else:
            self.profile_num = None
        if load_GyreProfile:
            self.GyreProfile = GyreProfile(f'{self.path}{suffix_GyreProfile}')

    def __repr__(self):
        try_to_get = ['model_number', 'num_zones', 'star_mass', 'star_age', 'Teff', 'photosphere_L',
                      'center_h1', 'center_he4', 'date']
        return 'MESA profile data file at {}'.format(self.path) + '\n' + str(
            {key: self.header[key] for key in try_to_get if key in self.header.keys()})


class _Gyre(_Data):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class GyreSummary(_Gyre):
    def __init__(self, *args, gyre_version='6', **kwargs):
        super().__init__(*args, **kwargs)
        self.gyre_version = gyre_version


    def __repr__(self):
        return f'GyreSummary loaded from {self.path}'


    def _dimless_to_Hz(self):
        if 'M_star' in self.header.keys():
            M_star = self.header['M_star']
            R_star = self.header['R_star']
            G = pre15140.standard_cgrav
        else:
            M_star = self.get('M_star')[0]
            R_star = self.get('R_star')[0]
            G = post15140.standard_cgrav  # This changed in version 6.
        return 1.0 / (2 * np.pi) * ((G * M_star / (R_star) ** 3))


    def get_frequencies(self, freq_units):
        if 'Re(omega)' in self.columns:
            dimless_to_Hz = self.calc_dimless_to_Hz() * {'uHz': 1e6, 'mHz': 1e3, 'Hz': 1e0}[freq_units]
            freq_name = 'Re(omega)'
            freq_unit = {'uHz': r'\mu', 'mHz': 'm', 'Hz': ''}[freq_units]
        elif 'Re(freq)' in self.columns:  # Assumes freq already in uHz.
            dimless_to_Hz = {'uHz': 1e0, 'mHz': 1e-3, 'Hz': 1e-6}[freq_units]
            freq_name = 'Re(freq)'
            freq_unit = {'uHz': r'\mu', 'mHz': 'm', 'Hz': ''}[freq_units]
        return self.data[freq_name] * dimless_to_Hz

class GyreMode(_Gyre):
    def __init__(self, *args, gyre_version, **kwargs):
        super().__init__(*args, **kwargs)
        self.gyre_version = gyre_version


    def __repr__(self):
        return f'GyreMode loaded from {self.path}'

    def _dimless_to_Hz(self):
        M_star = self.header['M_star']
        R_star = self.header['R_star']
        if self.gyre_version < '6':
            G = pre15140.standard_cgrav
        else:
            G = post15140.standard_cgrav
        return 1.0 / (2 * np.pi) * ((G * M_star / (R_star) ** 3))


    def get_frequencies(self, freq_units):
        if 'Re(omega)' in self.header:
            dimless_to_Hz = self.calc_dimless_to_Hz() * {'uHz': 1e6, 'mHz': 1e3, 'Hz': 1e0}[freq_units]
            freq_name = 'Re(omega)'
            freq_unit = {'uHz': r'\mu', 'mHz': 'm', 'Hz': ''}[freq_units]
        elif 'Re(freq)' in self.header:  # Assumes freq already in uHz.
            dimless_to_Hz = {'uHz': 1e0, 'mHz': 1e-3, 'Hz': 1e-6}[freq_units]
            freq_name = 'Re(freq)'
            freq_unit = {'uHz': r'\mu', 'mHz': 'm', 'Hz': ''}[freq_units]
        return self.data[freq_name] * dimless_to_Hz


def load_profs(hist, prefix='profile', suffix='.data', only_RC=False, save_dill=False, mask=None, mask_kwargs={}):
    if hist.index is None:
        return []
    pnums = hist.index[:, 2]
    if only_RC:
        mask_RC = uf.get_rc_mask(hist)
        min_RC = min(hist.get('model_number')[mask_RC])
        max_RC = max(hist.get('model_number')[mask_RC])
        pnums = pnums[np.logical_and(hist.index[:, 0] >= min_RC, hist.index[:, 0] <= max_RC)]

    if mask is not None:
        if only_RC:
            raise ValueError('only_RC and mask cannot be defined at the same time.')
        if hasattr(mask, '__call__'):
            mask = mask(hist, **mask_kwargs)
        valid_mod = hist.get('model_number')[mask]
        pnums = hist.index[:,2][np.in1d(hist.index[:, 0], valid_mod)]

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
             return_pnums=False, only_RC=False, use_mask=None):
    dirpath = os.path.abspath(os.path.join(hist.LOGS, '..', gyre_data_dir))
    
    if only_RC:
        use_mask = uf.get_rc_mask
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
        gss.append(GyreSummary(os.path.join(dirpath, fname)))
    
    if return_pnums:
        return list(zip(gss, np.array(pnums)))
    return gss


def load_modes_from_profile(prof, gyre_version, gyre_data_dir='gyre_out', mode_prefix='', mode_suffix='.mgyre'):
    dirpath = os.path.abspath(os.path.join(prof.LOGS, '..', gyre_data_dir))
    fnames = os.listdir(dirpath)

    if mode_prefix == '':
        mode_prefix = prof.fname

    modes = []
    for fname in fnames:
        if fname.startswith(mode_prefix):
            if fname.endswith(mode_suffix):
                modes.append(GyreMode(os.path.join(dirpath, fname), gyre_version))
    return modes


def load_gs_from_profile(prof, gyre_data_dir='gyre_out', gyre_summary_prefix='',
                         gyre_summary_suffix='.data.GYRE.sgyre_l'):
    dirpath = os.path.abspath(os.path.join(prof.LOGS, '..', gyre_data_dir))
    fnames = os.listdir(dirpath)

    if gyre_summary_prefix == '':
        gyre_summary_prefix = prof.fname

    for fname in fnames:
        if fname.startswith(gyre_summary_prefix) and fname.endswith(gyre_summary_suffix):
            break
        else:
            fname = ''

    gs = GyreSummary(os.path.join(dirpath, fname))
    return gs


def load_modes(hist, gyre_version, gyre_summary_dir='gyre_out', gyre_summary_prefix='profile',
               gyre_summary_suffix='.data.GYRE.sgyre_l', mode_dir='gyre_out/detail',
               mode_prefix='profile{}.', mode_suffix='.mgyre'):
    gss, pnums = load_gss(hist, gyre_summary_dir, gyre_summary_prefix, gyre_summary_suffix, return_pnums=True)
    
    dirpath = os.path.abspath(os.path.join(hist.LOGS, '..', mode_dir))
    fnames = os.listdir(dirpath)
    
    mode_info = []
    modes = []
    
    i = 0
    for fname in fnames:
        fname = os.path.split(fname)[-1]
        prefix_parts = mode_prefix.split('{}')
        pnum = int(fname[len(prefix_parts[0]):fname.index(prefix_parts[1])])
        mode_prefix_pnum = mode_prefix.format(pnum)
        if fname.startswith(mode_prefix_pnum) and fname.endswith(mode_suffix):
            md = GyreMode(os.path.join(dirpath, fname), gyre_version)
            nu = uf.get_freq(md, kind='mode')
            l = md.header['l']
            order_names = ['n_p', 'n_g', 'n_pg']
            if sum(np.in1d(order_names, list(md.header.keys()))) > 1:
                
                try:
                    n_p = md.header['n_p']
                except KeyError:
                    n_p = md.header['n_pg'] + md.header['n_g']
                    md.header['n_p'] = n_p
                    
                try:
                    n_g = md.header['n_g']
                except KeyError:
                    n_g = md.header['n_p'] - md.header['n_pg']
                    md.header['n_g'] = n_g
                    
                try:
                    n_pg = md.header['n_pg']
                except KeyError:
                    n_pg = md.header['n_p'] - md.header['n_g']
                    md.header['n_pg'] = n_pg
                
            else:
                raise ValueError('Not enough information in header to determine n_p, n_g, and n_pg.')
            
            md.header['profile_number'] = pnum
            
            for key in ['n_p', 'n_g', 'n_pg']:
                if key in md.header and key in locals().keys():
                    if md.header[key] != locals()[key]:
                        raise ValueError(f'Mismatch between mode {key} and calculated {key}!')
                    
                else:
                    md.header[key] = locals()[key]
            
            mode_info.append([pnum, l, nu, n_pg, n_p, n_g, i])
            modes.append(md)
            i += 1
    
    mode_info = np.rec.array(mode_info, names=['pnum', 'l', 'nu', 'n_pg', 'n_p', 'n_g', 'i'])
    mode_info = np.sort(mode_info, order=['pnum', 'l', 'nu', 'n_pg'])
    modes = np.asarray([modes[i] for i in mode_info.i], dtype=object)
    
    mode_info.i = np.arange(len(mode_info))
    
    return modes


class GyreProfile:
    def __init__(self, path):
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
                       'brunt_N2', 'gamma1', 'grad_ad', 'nu_T', 'opacity', 'opacity_partial_T', 'opacity_partial_rho',
                       'total_energy_generation', 'nuclear_energy_generation_partial_T',
                       'nuclear_energy_generation_partial_rho', 'rotation']

            self.formats = [int] + 18 * [float]
        elif self.version == 101:
            self.columns = ['zone', 'radius', 'mass', 'luminosity', 'pressure', 'temperature', 'density', 'grad_T',
                       'brunt_N2', 'gamma1', 'grad_ad', 'nu_T', 'opacity', 'opacity_partial_T', 'opacity_partial_rho',
                       'nuclear_energy_generation', 'nuclear_energy_generation_partial_T',
                       'nuclear_energy_generation_partial_rho', 'rotation']

            self.formats = [int] + 18 * [float]
        elif self.version == 120:
            self.columns = ['zone', 'radius', 'mass', 'luminosity', 'pressure', 'temperature', 'density', 'grad_T',
                       'brunt_N2', 'gamma1', 'grad_ad', 'nu_T', 'opacity', 'opacity_partial_T', 'opacity_partial_rho',
                       'nuclear_energy_generation', 'nuclear_energy_generation_partial_T',
                       'nuclear_energy_generation_partial_rho', 'gravothermal_energy_generation', 'rotation']
            self.formats = [int] + 19 * [float]
        else:
            raise NotImplementedError('Only fileversions 100, 101, and 120 are implemented implemented.')

        self.loaded = False

    def _load_gyre_profile(self):
        data = np.rec.array(np.loadtxt(f'{self.path}', skiprows=1, dtype={'names': self.columns, 'formats': self.formats}))
        return data

    @LazyProperty
    def data(self):
        self.data = self._load_gyre_profile()
        self.loaded = True
        return data


def naive_merge_hists(base_hist, hists):
    """Naive hist merging"""
    new_hist = copy.copy(base_hist)
    new_hist.data = np.lib.recfunctions.stack_arrays([h.data for h in hists], asrecarray=True, usemask=False)
    return new_hist
    

def load_gss_to_hist(hist, gyre_data_dir='gyre_out', gyre_summary_prefix='profile', gyre_summary_suffix='.data.GYRE.sgyre_l', only_RC=False, use_mask=None):
    return_pnums = True
    hist.gsspnum = load_gss(hist, gyre_data_dir, gyre_summary_prefix, gyre_summary_suffix, return_pnums, only_RC, use_mask)
    return hist
