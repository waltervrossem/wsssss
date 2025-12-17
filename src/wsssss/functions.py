#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import functools
import multiprocessing as mp
import os
import re

import dill
import numpy as np
from scipy import integrate as ig
from scipy import interpolate as ip
from scipy import stats
from scipy.optimize import newton
from .constants import post15140
from .constants import pre15140


def get_mesa_version(mesa_dir):
    """
    Get the version of MESA in mesa_dir

    Args:
        mesa_dir (str): Path to MESA directory.

    Returns:
        str: Version string

    Examples:
        >>> get_mesa_version(os.environ['MESA_DIR'])
        '24.03.1'
    """
    with open(f'{mesa_dir}/data/version_number', 'r') as handle:
        version = handle.read().strip()
    return version


def compare_version(version1, version2, operator):
    """
    Compare two MESA versions.

    Args:
        version1 (str): LHS of comparison.
        version2 (str): RHS of comparison.
        operator (str): Comparison operator string. Can be one of '<', '>', '<=', '>=', '==', '!='.

    Returns:
        bool: Result of performing the comparison.

    Examples:
        >>> compare_version('11701', '8118', '<')  # '11701' < '8118' would return True.
        False
    """
    allowed_ops = ['<', '>', '<=', '>=', '==', '!=']
    if not operator in allowed_ops:
        raise ValueError(f'`operator` must be one of: {", ".join(allowed_ops)}.')
    i = allowed_ops.index(operator)

    r_version1 = False
    r_version2 = False
    if version1.startswith('r'):
        r_version1 = True
    if version2.startswith('r'):
        r_version2 = True

    lt = None
    gt = None
    lte = None
    gte = None
    eq = None
    neq = None

    eq = version1 == version2
    neq = not eq

    # if version number starts with an r it is after 15140
    if r_version1 and r_version2:
        lt = version1 < version2
    elif r_version1 and not r_version2:
        lt = False
    elif not r_version1 and r_version2:
        lt = True
    else:
        lt = int(version1) < int(version2)

    gte = not lt
    gt = gte and neq
    lte = not gt

    return [lt, gt, lte, gte, eq, neq][i]


def get_constants(p_or_hist):
    """
    Get the correct MESA constants used for p_or_hist.

    Args:
        p_or_hist (Profile or History): Profile or History for which to get constants

    Returns:
        module: Module containing the constants for p_or_hist.
    """
    version = str(p_or_hist.header['version_number'])

    if compare_version(version, '15140', '<'):
        return pre15140
    else:
        return post15140


# Mixing type codes for pre and post 15140
mix_dict = {'pre15140': {-1:'no_region',
                         0: 'no_mixing',
                         1: 'convective_mixing',
                         2: 'softened_convective_mixing',
                         3: 'overshoot_mixing',
                         4: 'semiconvective_mixing',
                         5: 'thermohaline_mixing',
                         6: 'rotation_mixing',
                         7: 'rayleigh_taylor_mixing',
                         8: 'minimum_mixing',
                         9: 'anonymous_mixing'},
            'post15140': {-1:'no_region',
                          0: 'no_mixing',
                          1: 'convective_mixing',
                          2: 'overshoot_mixing',
                          3: 'semiconvective_mixing',
                          4: 'thermohaline_mixing',
                          5: 'rotation_mixing',
                          6: 'rayleigh_taylor_mixing',
                          7: 'minimum_mixing',
                          8: 'anonymous_mixing',
                          9: 'leftover_convective_mixing'},
            # Need a merged when combining histories from different versions
            'merged': {'no_region': -1,
                       'no_mixing': 100,
                       'convective_mixing': 101,
                       'overshoot_mixing': 103,
                       'semiconvective_mixing': 104,
                       'thermohaline_mixing': 105,
                       'rotation_mixing': 106,
                       'minimum_mixing': 107,
                       'anonymous_mixing': 109},
            'names':  {'no_region': '',
                       'no_mixing': 'No mixing',
                       'convective_mixing': 'Convection',
                       'overshoot_mixing': 'Overshooting',
                       'semiconvective_mixing': 'Semiconvection',
                       'thermohaline_mixing': 'Thermohaline',
                       'rotation_mixing': 'Rotational',
                       'minimum_mixing': 'Minimum',
                       'anonymous_mixing': 'Anonymous'},
            }
mix_dict['merged_r'] = {v:k for k,v in mix_dict['merged'].items()}  # Reversed merged.
def convert_mixing_type(mix_type, version, unknown_mixing=100, other_mixing=[(None, None)]):
    """
    Convert the mixing type codes to a merged version compatible with pre- and post-15140 MESA.

    Args:
        mix_type (np.array):
        version (str): MESA version
        unknown_mixing (int, optional): Mixing type code to use for unreckognized mixing type code.
            Defaults to 100, which is the merged code for no_mixing.
        other_mixing (list of tuple(str, int), optional):  Information on custom mixing types. (name, mix_type_code)

    Returns:
        np.array: New mixing type codes associated with mix_type.
    """
    if compare_version(version, '15140', '>='):
        pre_post = 'post'
    else:
        pre_post = 'pre'
    key = f'{pre_post}15140'

    merged = mix_dict['merged']
    for custom_name, custom_mix_type in other_mixing:
        if custom_name is not None and custom_mix_type is not None:
            mix_dict[key][custom_mix_type] = custom_name
            merged[custom_name] = custom_mix_type
            mix_dict['merged_r'] = {v: k for k, v in mix_dict['merged'].items()}

    mix_names = np.vectorize(mix_dict[key].get)(mix_type, unknown_mixing)
    return np.vectorize(mix_dict['merged'].get)(mix_names, unknown_mixing)


def cell2face(val, dm, dm_is_m=False, m_center=0):
    """
    Convert a cell average value (e.g. density) to the value at the cell face (e.g. radius).

    Args:
        val (np.array): Value to convert.
        dm (np.array): Cell mass or mass coordinate if dm_is_m is True.
        dm_is_m (bool, optional): If True, treat dm as mass coordinate instead of cell mass.
        m_center (float, optional: If dm_is_m is True, set this as the interior mass.

    Returns:
        np.array: Value at cell face of val.
    """
    if dm_is_m:
        dm = np.diff(dm, append=m_center)
    face = np.zeros_like(val)
    face[0] = val[0]
    face[1:] = (val[:-1] * dm[1:] + val[1:] * dm[:-1]) / (dm[1:] + dm[:-1])
    return face


def get_logTeffL(hist, mask=None, linear=False):
    if 'log_Teff' in hist.columns:
        logTeff = hist.get('log_Teff')
    elif 'effective_T' in hist.columns:
        logTeff = np.log10(hist.get('effective_T'))
    else:
        raise ValueError('log_Teff or effective_T not in history file.')

    if 'photosphere_L' in hist.columns:
        logL = np.log10(hist.get('photosphere_L'))
    elif 'luminosity' in hist.columns:
        logL = np.log10(hist.get('luminosity'))
    elif 'log_L' in hist.columns:
        logL = hist.get('log_L')
    else:
        if 'photosphere_r' in hist.columns:
            print('Calculating logL from Teff and R_photosphere.')
            R = hist.get('photosphere_r')
            logL = 2 * np.log10(R) + 4 * logTeff - 4 * np.log10(5777)
        else:
            raise ValueError('log_L, luminosity, or photosphere_L not in history file.')
    mask = get_mask(hist, mask)
    if linear:
        logTeff = 10 ** logTeff
        logL = 10 ** logL
    return logTeff[mask], logL[mask]


def get_radius(p, unit='cm'):
    """
    Get the radius from Profile p and perform a unit conversion with the correct constants.

    Args:
        p (Profile):
        unit (str):

    Returns:
        array-like: Radial coordinate in unit.
    """
    c = get_constants(p)

    if 'radius' in p.columns:
        radius = p.get('radius') * c.rsun
    elif 'logR' in p.columns:
        radius = 10 ** p.get('logR') * c.rsun
    elif 'radius_cm' in p.columns:
        radius = p.get('radius_cm')
    else:
        raise ValueError("No radius found in profile.")

    if unit == 'log':
        radius = np.log10(radius/c.rsun)
    elif unit.lower().replace('_', '') in ['rsun', 'rsol']:
        radius = radius / c.rsun
    elif unit.lower() == 'cm':
        radius = radius
    else:
        raise ValueError(f'Unknown unit {unit}. Must be one of cm, rsun, rsol, or log.')
    return radius


def get_m_bot_CZ(hist, mask=None, max_q_bot=0.999):
    """
    Get the mass coordinate of the bottom of the convective envelope.

    Args:
        hist (History):
        mask (bool, np.array, or function, optional): If True, will exclude pre-main sequence.
                If an array of bools will use that as mask. If a function, will call function(hist) and use that as the mask.
        max_q_bot (float, optional): Maximum dimensionless mass coordinate below which to look for the bottom of the convective envelope.

    Returns:
        array-like: Bottom of the convective envelope.
    """
    if 'm_botCZ' in hist.columns:
        return hist.get('m_botCZ', mask=mask)
    m_bot_CZ = np.zeros_like(hist.get('star_mass', mask=mask))
    n_mix = 0
    for col in hist.columns:
        if col.startswith('mix_qtop'):
            n_mix += 1
    for i in range(n_mix):
        i += 1
        mix_type = convert_mixing_type(hist.get(f'mix_type_{i}', mask=mask), hist.header['version_number'])
        m_bot_CZ = np.maximum(m_bot_CZ,
                              hist.get('star_mass', mask=mask) * hist.get(f'mix_qtop_{i}', mask=mask) *
                              (mix_type == mix_dict['merged']['no_mixing']) * (hist.get(f'mix_qtop_{i}', mask=mask) <= max_q_bot))
    return m_bot_CZ


def get_cheb_mask(hist, min_Yc=1e-6, max_Yc=1.0):
    """
    Get the Core Helium Burning mask for hist.

    Defined as:
    * Have helium in core
    * Have convective core
    * Hydrogen depleted core

    Args:
        hist (History):
        min_Yc (float, optional): Minimum central helium mass fraction. Defaults to 1e-6.
        max_Yc (float, optional): Maximum central helium mass fraction. Defaults to 1.0.

    Returns:
        array of bool:
    """

    mask = hist.data.center_he4 >= min_Yc
    mask = np.logical_and(mask, hist.data.center_he4 <= max_Yc)
    mask = np.logical_and(mask, hist.data.mass_conv_core > 0)
    mask = np.logical_and(mask, hist.data.center_h1 < 1e-6)

    return mask


def get_rc_mask(hist, min_Yc=0.1, max_fYc=0.95, first_chunk=False):
    """
    Get the Red Clump mask for hist.

    Defined as:
    Have helium in core
    Have convective core
    Hydrogen depleted core

    Args:
        hist (History):
        min_Yc (float, optional): Minimum central helium mass fraction. Defaults to 0.1.
        max_fYc (float, optional): Maximum central helium mass fraction factor. This factor is multiplied by the
            maximum Yc achieved during evolution to get the value which is compared against. Defaults to 0.95.
        first_chunk (bool, optional): If True, only returns the mask for the first contiguous block of True in the mask.

    Returns:
        array of bool:
    """

    mask = hist.data.center_he4 >= min_Yc
    mask = np.logical_and(mask, hist.data.center_he4 <= max_fYc * max(hist.data.center_he4))
    mask = np.logical_and(mask, hist.data.mass_conv_core > 0)
    mask = np.logical_and(mask, hist.data.center_h1 < 1e-6)

    if first_chunk and np.any(mask):
        end_first_block = np.where(np.diff(mask, prepend=False))[0][1]
        mask[end_first_block + 1:] = False
    return mask


def get_pms_mask(hist, invert=False, ZAMS_method='Xc', fXc=0.99):
    """
    Get the pre-main sequence mask for hist.

    Args:
        hist (History):
        invert (bool, optional): If True, invert the PMS mask. Defaults to False.
        ZAMS_method (str or callable, optional): Can be 'Xc', or 'LH'. If 'Xc', will use when X_c reaches ``fXc`` of
            the initial central hydrogen mass fraction. If 'LH', will end when the hydrogen burning luminosity is equal
            to total luminosity. If ZAMS_method is callable will call ``ZAMS_method(hist)``, which should return a
            mask with the same length as hist. Defaults to 'Xc'.
        fXc (float, optional): Define the end of PMS as when central hydrogen mass fraction reaches ``fXc`` of the
            initial value. Defaults to 0.99

    Returns:
        array of bool:

    """

    if ZAMS_method == 'LH':
        _, logL = get_logTeffL(hist)
        if 'log_LH' in hist.columns:
            maskL = ((hist.data.log_LH - logL) < 0) & (hist.data.center_h1 > 0.6)
        else:
            have_pp = 'pp' in hist.columns
            have_cno = 'cno' in hist.columns
            if have_pp and have_cno:
                log_LH = np.log10(10**hist.get('pp') + 10**hist.get('cno'))
                maskL = ((log_LH - logL) < 0) & (hist.data.center_h1 > 0.6)
            else:
                raise KeyError('Must have either log_LH or pp and cno as columns if using hydrogen luminosity as ZAMS start.')
        starts_pms = maskL[0]

        mask = np.zeros_like(maskL)
        if starts_pms:
            end = np.where(np.diff(maskL))[0][0]
            mask[:end + 1] = True
    elif ZAMS_method == 'Xc':
        mask = hist.data.center_h1 >= hist.data.center_h1[0] * 0.99
    elif callable(ZAMS_method):
        mask = ZAMS_method(hist)
    else:
        raise ValueError(f'Unknown ZAMS_method: {ZAMS_method}.')


    if invert:
        mask = ~mask

    return mask


def get_ms_mask(hist, min_Xc=1e-3, ZAMS_method='Xc'):
    """
    Get the main sequence mask for hist.

    Args:
        hist (History):
        min_Xc (float): Minimum central hydrogen mass fraction for the end of the MS.
        ZAMS_method (str, optional): See get_pms_mask for details. Defaults to 'Xc'.

    Returns:
        array of bool:

    """

    mask = hist.data.center_h1 > min_Xc
    mask = np.logical_and(mask, get_pms_mask(hist, invert=True, ZAMS_method=ZAMS_method))

    return mask


def get_sgb_mask(hist, min_Xc=1e-3, fCZ=0.35):
    """
    Get the sub-giant branch mask.

    Args:
        hist (History):
        min_Xc (float): Minimum central hydrogen mass fraction for the end of the MS.
        fCZ (float): Convective envelope mass fraction limit for end of SGB.

    Returns:
        array of bool:

    """
    ms_mask = get_ms_mask(hist, min_Xc)
    if np.any(ms_mask):
        min_mod = hist.data.model_number[ms_mask][-1]
    else:
        if 'center_Rho' in hist.columns:
            center_Rho = hist.get('center_Rho')
        elif 'log_center_Rho' in hist.columns:
            center_Rho = 10 ** hist.get('log_center_Rho')
        if np.log10(center_Rho[0]) >= 3.5:  # starts during or after SGB
            return np.zeros_like(ms_mask, dtype=bool)
        min_mod = hist.data.model_number[0]

    mask = get_m_bot_CZ(hist) / hist.data.star_mass <= (1 - fCZ)
    mask = mask & (hist.data.model_number > min_mod)
    max_mod = hist.data.model_number[mask][0]

    mask = np.logical_and(hist.data.model_number > min_mod, hist.data.model_number < max_mod)
    return mask


def get_bump_mask(hist, max_logT_lim=3.8, min_logT_lim=3.6, min_logL=0.5):
    """
    Get the RGB bump mask.

    Defined as having an increasing effective temperature on the RGB within the Teff and L limits.

    Args:
        hist (History):
        max_logT_lim (float, optional): Maximum logT below which to look for the RGBb. Defaults to 3.8.
        min_logT_lim (float, optional): Minimum logT below which to look for the RGBb. Defaults to 3.6.
        min_logL (float, optional): Minimum logL  above which to look for the RGBb. Defaults to 0.5.

    Returns:
        array of bool:

    """

    logTeff, logL = get_logTeffL(hist)

    mask = hist.data.center_h1 < 1e-9
    mask = mask & (hist.data.center_he4 > 0.95)
    mask = mask & get_rgb_mask(hist)

    dlogT = np.diff(logTeff, prepend=99)
    dlogL = np.diff(logL, prepend=99)

    angle = np.arctan2(dlogL, dlogT)
    mask = mask & (angle < -0.2 * np.pi) & (angle > -0.6 * np.pi)

    # Cooler than (3.5, 1) -> (3.6, 3)
    y = 20 * (logTeff - 3.5) + 1
    mask = mask & (y > logL)

    # Bump is defined as when L decreases and Teff increases on RGB
    # submask_bump = np.logical_and(np.diff(logL[mask], prepend=-99) < 0, np.diff(logTeff[mask], prepend=99) > 0)
    # Bump is defined as when Teff increases on RGB
    # submask_bump = np.diff(logTeff[mask], prepend=99) > 0
    # mask[mask] = submask_bump

    return mask


def get_tip_mask(hist, logT_lim=3.8, min_logL=2.0):
    """
    Get the RGB tip mask for hist.

    Args:
        hist (History):
        logT_lim (float, optional): Maximum logT below which to look for the RGB tip.

    Returns:
        array of bool:

    """
    logTeff, logL = get_logTeffL(hist)

    mask = hist.data.center_h1 < 1e-9
    mask = np.logical_and(mask, hist.data.center_he4 > 0.95)
    mask = np.logical_and(mask, logTeff < logT_lim)
    mask = np.logical_and(mask, logL > min_logL)

    if sum(mask) == 0:
        return np.zeros_like(mask, dtype=bool)

    minT = min(logTeff[mask])
    maxL = max(logL[mask])

    mask = np.logical_and(mask, logTeff < minT + 0.01)
    mask = np.logical_and(mask, logL > maxL - 0.1)

    return mask


def get_rgb_mask(hist, min_Xc=1e-3, logT_lim=3.8):
    """
    Get the RGB mask for hist.

    Args:
        hist (History):
        min_Xc (float, optional): Minimum central hydrogen mass fraction for the end of the MS, passed to get_sgb_mask.
            Defaults to 1e-3.
        logT_lim (float, optional): Maximum logT below which to look for the end of the RGB. Defaults to 3.8.

    Returns:
        array of bool:

    """
    try:
        sgb_mask = get_sgb_mask(hist, min_Xc)
        if np.any(sgb_mask):
            first_mod = max(hist.data.model_number[sgb_mask])
        else:
            first_mod = hist.data.model_number[0]
    except IndexError:
        first_mod = hist.data.model_number[0]

    has_CHeB = np.any(get_cheb_mask(hist))

    if has_CHeB:
        tip_mask = get_tip_mask(hist, logT_lim)
        logTeff, logL = get_logTeffL(hist)
        masked_i_min_T = np.argmin(logTeff[tip_mask])
        last_mod = hist.data.model_number[tip_mask][masked_i_min_T]
    else:
        if hist.data.center_he4[-1] > 0.9:  # stopped on RGB
            last_mod = hist.data.model_number[-1]

    mask = np.logical_and(hist.data.model_number > first_mod, hist.data.model_number <= last_mod)
    return mask


def get_flashes_mask(hist, logT_lim=3.8):
    """
    Get the helium flashes mask.
    Starts after the RGB tip and ends when CHeB starts.

    Args:
        hist (History):
        logT_lim (float, optional): Passed to `get_tip_mask` to get the start of He-flashes.

    Returns:
        array of bool:

    """
    rc_mask = get_cheb_mask(hist)
    if np.any(rc_mask):
        last_mod = min(hist.data.model_number[rc_mask])
    else:
        last_mod = hist.data.model_number[-1]

    tip_mask = get_tip_mask(hist, logT_lim)
    logTeff, logL = get_logTeffL(hist)
    if np.any(tip_mask):
        masked_i_min_T = np.argmin(logTeff[tip_mask])
        first_mod = hist.data.model_number[tip_mask][masked_i_min_T]
        mask = np.logical_and(hist.data.model_number > first_mod, hist.data.model_number < last_mod)
    else:
        mask = np.zeros_like(rc_mask)

    return mask


# def get_agb_mask(hist):
#     """
#     Get the AGB mask.
#
#
#     Args:
#         hist (History):
#
#     Returns:
#         array of bool:
#     TODO: Define an ending condition.
#     """
#     rc_mask = get_cheb_mask(hist)
#     if np.any(rc_mask):
#         mod_agb_start = max(hist.data.model_number[rc_mask])
#         mask = hist.data.model_number > mod_agb_start
#     else:
#         mask = np.zeros_like(rc_mask, dtype=bool)
#     return mask


mask_functions = [get_pms_mask, get_ms_mask, get_sgb_mask, get_rgb_mask, get_bump_mask, get_tip_mask, get_flashes_mask,
                  get_cheb_mask, get_rc_mask]#, get_agb_mask]
mask_names = ['PMS', 'MS', 'SGB', 'RGB', 'RGBb', 'RGB tip', 'He flashes', 'CHeB', 'RC']#, 'post-CHeB']


def get_gridnum(hist):
    """

    Args:
        hist (History):

    Returns:

    """
    return int(os.path.split(hist.LOGS)[0][-4:])


def get_mask(hist, use_mask):
    """
    Evaluate a mask.

    Args:
        hist (History):
        use_mask (bool, np.array, or function, optional): If True, will exclude pre-main sequence.
                If an array of bools will use that as mask. If a function, will call function(hist) and use that as the mask.

    Returns:
        array of bool:

    """
    if hasattr(use_mask, '__len__'):
        if len(use_mask) == len(hist.get('model_number')):
            mask = use_mask
        else:
            # Try to get numpy to raise the index error first
            _ = hist.get('model_number')[use_mask]
            raise IndexError(
                f'Length of mask not the same as length of data ({len(use_mask)} vs {len(hist.get("model_number"))}')
    elif callable(use_mask):
        mask = use_mask(hist)
    else:
        if use_mask:
            mask = get_pms_mask(hist, invert=True)
        else:
            mask = np.ones_like(hist.get('model_number'), dtype=bool)
    return mask


def get_mean(hist, name, use_mask=None, domain='star_age', filter=None, get_std=False, verbose=True):
    """
    Calculate the average of a quantity defined by `name` over `domain`.

    Args:
        hist (History):
        name (str, array, function): Quantity in hist to average. Can be a column name in hist, an array with the same
                length as hist, an array with the mask specified by use_mask already applied, or a function with
                signature function(hist, mask).
        use_mask (bool, np.array, or function, optional): If True, will exclude pre-main sequence.
                If an array of bools will use that as mask. If a function, will call function(hist) and use that as the mask.
        domain (str, array, optional): Domain over which to average. Must be monotonically increasing or
                decreasing. Can be a column name in hist, or an array with the same length as hist. Defaults to 'star_age'.
        filter (str, optional): Quantities to mask out. Can be one of 'negative', '-', '-ve', 'positive', '+', '+ve', 'finite', 'inf', 'nan', None. Defaults to None.
        get_std (bool, optional): If True, will also return the standard deviation.

    Returns:
        float or Tuple of floats: The average
    """
    if use_mask is not None:
        mask = get_mask(hist, use_mask=use_mask)
    else:
        mask = ...

    if isinstance(domain, str):
        xdat = hist.get(domain)[mask]
    else:
        xdat = domain[mask]
    x_monotonic = False
    if np.all(np.diff(xdat) > 0):
        x_monotonic = True
    elif np.all(np.diff(xdat) < 0):
        x_monotonic = True
    else:
        raise ValueError(f"domain `{domain}` must be monotonic.")
    if isinstance(name, str):
        ydat = hist.get(name)[mask]
    elif callable(name):
        ydat = name(hist, mask)
    else:
        if not isinstance(mask, Ellipsis.__class__):
            if len(name) == sum(mask):  # mask already applied
                ydat = name
            else:
                ydat = name[mask]
        else:
            ydat = name

    if filter is not None:
        filters = filter.split()
        mask = np.ones_like(ydat, dtype=bool)
        for filter in filters:
            if filter.lower() in ['negative', '-', '-ve']:
                mask = mask & (ydat >= 0)
            elif filter.lower() in ['positive', '+', '+ve']:
                mask = mask & (ydat <= 0)
            elif filter.lower() in ['finite', 'inf', 'nan']:
                mask = mask & (np.isfinite(ydat))

        xdat = xdat[mask]
        ydat = ydat[mask]

    if len(xdat) == 0:
        if verbose:
            print(f"Warning: 0-length data when calculating mean of {domain}")
        mean = np.nan
        std = np.nan
    else:
        mean = ig.trapz(ydat, xdat) / (xdat[-1] - xdat[0])
        std = (ig.trapz((ydat - mean) ** 2, xdat) / (xdat[-1] - xdat[0])) ** 0.5
        std = np.mean(np.diff(
            np.quantile(ip.interp1d(xdat, ydat)(np.linspace(min(xdat), max(xdat), 201)), [0.15865, 0.50, 0.84135])))
    if get_std:
        return mean, std
    else:
        return mean


def get_weighted_quantile(x, w, q=(0.15865, 0.50, 0.84135)):
    """
    Calculate the weighted quantile of `x` using weights `w`.

    Args:
        x:
        w (array): Weights of `x`.
        q (tuple of floats): Quantiles at which to calculate.

    Returns:

    """
    if not np.all(x.shape == w.shape):
        raise ValueError('x and w must have the same shape.')

    if len(x) == 0:
        return np.nan * np.ones_like(q)

    argsort = np.argsort(x)
    x = x[argsort]
    w = w[argsort]
    cumsumw = np.cumsum(w)
    cumsumw /= cumsumw[-1]

    return ip.interp1d(cumsumw, x)(q)


# def get_instability_strip(Z, logL_min=3, logL_max=4.5):
#     """Bono2005"""
#     Z = min(max(Z, 0.004), 0.02)
#
#     blue_coefs = np.array([[0.004, 3.617, 0.108, -0.018, 0.001],
#                           [.008, 3.557, 0.153, -0.026, 0.001],
#                           [.02, 3.729, 0.064, -0.017, 0.001]])
#     red_coefs = np.array([[0.004, 3.678, 0.090, -0.026, 0.001],
#                          [0.008, 3.709, 0.077, -0.022, 0.001],
#                          [0.02, 3.763, 0.053, -0.021, 0.001]])
#     logL = np.linspace(logL_min, logL_max, 101)
#     edges = []
#     for coefs in [blue_coefs, red_coefs]:
#         alpha, beta, gamma, _ = ip.interp1d(coefs[:, 0], coefs[:, 1:].T)(Z)
#         logT = alpha + beta*logL + gamma*logL**2
#         edges.append(logT)
#     return logL, edges

def get_instability_strip(Z, Y, logL_min=3, logL_max=4.5, kind='RRLyrae'):
    if kind == 'RRLyrae':
        # https://ui.adsabs.harvard.edu/abs/2015ApJ...808...50M/abstract
        logL_min = 1.5
        logL_max = 1.9

        logL = np.linspace(logL_min, logL_max, 101)
        logTr = -0.084 * logL - 0.012 * np.log10(Z) + 3.879
        logTb = -0.080 * logL - 0.012 * np.log10(Z) + 3.957

        return logL, (logTb, logTr)

    elif kind == 'Cepheid':
        # https://arxiv.org/pdf/astro-ph/9801242.pdf

        (Xref, Zref) = (0.7, 0.004)
        Yref = 1 - Xref - Zref
        deltaY = Y - Yref
        deltaZ = Z - Zref

        logL = np.linspace(logL_min, logL_max, 101)
        logTeff = -0.036 * logL + 3.925
        dlogTeff = 0.04 * deltaY - 0.49 * deltaZ
        logTeff += dlogTeff

    return logL, (logTeff, logTeff - 0.06)


def get_evo_phase(hist, phase_funcs):
    """

    Args:
        hist:
        phase_funcs:

    Returns:

    """
    evo_phase = np.zeros_like(hist.get('model_number'))
    max_phase = 1

    # TODO: Be able to deal with runs that end early
    phase_pass = {}
    for phase_func in phase_funcs:
        try:
            mask = phase_func(hist)
            if np.any(mask):
                evo_phase[mask] = max_phase
                max_phase += 1
                phase_pass[phase_func] = True
            else:
                phase_pass[phase_func] = False
        except (ValueError, IndexError):
            phase_pass[phase_func] = False
            print(phase_func.__name__, 'failed.')
    return evo_phase, max_phase, phase_pass


def get_evo_stretch_func(hist, xaxis='star_age', phase_funcs=None):
    if phase_funcs is None:
        phase_funcs = [get_pms_mask, get_ms_mask, get_sgb_mask, get_rgb_mask, get_flashes_mask, get_cheb_mask,
                       get_agb_mask]
    evo_phase, max_phase, phase_pass = get_evo_phase(hist, phase_funcs)

    xdata = hist.get(xaxis)

    if np.all(np.diff(xdata) < 0):  # monotonically decreasing
        factor = -1 * xdata
    elif np.all(np.diff(xdata) > 0):  # monotonically increasing
        pass
    else:
        raise ValueError(f'Quantity specified in `xaxis` ({xaxis}) must be monotonic.')

    stretched = np.zeros_like(xdata) + evo_phase
    for phase in range(1, max_phase + 1):
        mask = evo_phase == phase
        if sum(mask) == 0:
            continue
        phase_xdata = xdata[mask]
        xdata_min = min(phase_xdata)
        xdata_max = max(phase_xdata)
        func = ip.interp1d([xdata_min, xdata_max], [0.0, 1.0])

        stretched[mask] += func(phase_xdata)

    func = ip.interp1d(hist.data.model_number, stretched)
    return func, phase_pass, max_phase


def get_bottom_envelope(p, indeces_only=False):
    """

    Args:
        p:
        indeces_only:

    Returns:

    """
    version = str(p.header['version_number'])
    radius = get_radius(p, 'Rsol')
    mass = p.get('mass')
    temperature = p.get('temperature')
    mix_type = p.get('mixing_type')

    if compare_version(version, '15140', '>='):
        prefix = 'post'
    else:
        prefix = 'pre'
    mix_type = np.array(list(map(mix_dict['merged'].get, map(mix_dict[f'{prefix}15140'].get, mix_type))))
    mix_OS = mix_type == mix_dict['merged']['overshoot_mixing']  # OS
    mix_CV = mix_type == mix_dict['merged']['convective_mixing']  # convective
    temp_mask = (temperature[1:] > p.header['Teff'] * 2)
    radiative = False
    try:
        bottom_of_CZ = np.where(mix_CV[:-1] & mix_OS[1:] & temp_mask)[0][0]
        bottom_of_US = np.where(mix_OS[:-1] & (mix_type[1:] != mix_dict['merged']['convective_mixing']) & (mix_type[1:] != mix_dict['merged']['overshoot_mixing']) & temp_mask)[0][0]
    except IndexError:
        radiative = True
        bottom_of_CZ = 0
        bottom_of_US = 0
    if indeces_only:
        return bottom_of_CZ, bottom_of_US
    else:
        if not radiative:
            # grada = p.get('grada')
            # gradr = p.get('gradr')
            # grads = grada[bottom_of_CZ - 10:bottom_of_CZ + 10] - gradr[bottom_of_CZ - 10:bottom_of_CZ + 10]
            # r_bCZ = float(ip.interp1d(grads, radius[bottom_of_CZ - 10:bottom_of_CZ + 10])(0))
            # m_bCZ = float(ip.interp1d(grads, mass[bottom_of_CZ - 10:bottom_of_CZ + 10])(0))
            r_bCZ = radius[bottom_of_CZ]
            m_bCZ = mass[bottom_of_CZ]
        else:
            r_bCZ = radius[bottom_of_CZ]
            m_bCZ = mass[bottom_of_CZ]
    return p.header['model_number'], m_bCZ, r_bCZ, mass[bottom_of_US], radius[bottom_of_US]


def get_lamb2(p, l=1):
    if 'lamb_S2' in p.columns:
        lamb2 = p.data.lamb_S2 * (l * (l + 1) / 2)
    elif 'lamb_Sl1' in p.columns:
        lamb2 = (p.data.lamb_Sl1 / (1e6 / (2 * np.pi))) ** 2 * (l * (l + 1) / 2)  # l part to convert from l=1 to l=l
    else:
        radius = get_radius(p, unit='cm')
        lamb2 = l * (l + 1) * (p.data.csound / radius) ** 2
    return lamb2


def calc_MH(hist, ZX_sol=0.0178, use_mask=None):
    """
    Calculate [M/H] from hist.

    Args:
        hist:
        ZX_sol (float, optional): Solar value for Z/X. Defaults to 0.0178.
        use_mask (bool, np.array, or function, optional): If True, will exclude pre-main sequence.
                If an array of bools will use that as mask. If a function, will call function(hist) and use that as the mask.

    Returns:

    """
    mask = get_mask(hist, use_mask)
    surf_X = hist.data.surface_h1[mask] + hist.data.surface_h2[mask]
    surf_Y = hist.data.surface_he3[mask] + hist.data.surface_he4[mask]
    surf_Z = 1 - surf_X - surf_Y
    FeH = np.log10((surf_Z / surf_X) / ZX_sol)
    return FeH


def calc_deltanu(gs, hist, prefix='profile', suffix='.data.GYRE.sgyre_l', freq_units='uHz'):
    """
    Calculate the large frequency separation Delta nu.

    Args:
        gs (GyreSummary):
        hist (History):
        prefix (str, optional): Part of gyre summary name before the profile number. Defaults to 'profile'.
        suffix (str, optional): Part of gyre summary name after the profile number. Defaults to '.data.GYRE.sgyre_l'.
        freq_units (str): Unit to convert to, must be one of 'uHz', 'mHz', or 'Hz'.

    Returns:
        float: Large frequency separation \\Delta\\nu.
    """
    pnum = int(gs.path.split(prefix)[-1].replace(suffix, ''))
    i_hist = hist.get_profile_index(pnum)[0]
    nu_all = gs.get_frequencies(freq_units)

    mask = gs.data.l == 0
    mask = np.logical_and(mask, gs.data.n_pg > 0)
    nu_max = hist.get('nu_max')[i_hist]
    fsig = (0.66 * nu_max ** 0.88) / 2 / np.sqrt(2 * np.log(2.))
    w = np.exp(-((nu_all[mask][:-1] - nu_max) / fsig) ** 2)
    delta_nus = np.diff(nu_all[mask])
    delta_nu = np.sum(w * delta_nus) / np.sum(w)
    return delta_nu


def calc_deltaPg(gs, hist, l, prefix='profile', suffix='.data.GYRE.sgyre_l'):
    """
    Calculate the period spacing Delta P weighted by the power spectrum envelope.
    Args:
        gs (GyreSummary):
        hist (History):
        l (int): Degree of modes.
        prefix (str, optional): Part of gyre summary name before the profile number. Defaults to 'profile'.
        suffix (str, optional): Part of gyre summary name after the profile number. Defaults to '.data.GYRE.sgyre_l'.

    Returns:
        float: Period spacing Delta P.
    """
    pnum = int(gs.path.split(prefix)[-1].replace(suffix, ''))
    i_hist = hist.get_profile_index(pnum)[0]
    nu_max = hist.get('nu_max')[i_hist]

    if l == 0:
        raise ValueError('Cannot use l=0 for period spacing.')

    mask = gs.data.l == l
    nu = gs.get_frequencies('Hz')[mask]
    dPi = -np.diff(nu ** -1)

    fsig = (0.66 * nu_max ** 0.88) / 2 / np.sqrt(2 * np.log(2.))
    w = np.exp(-((nu[:-1] - nu_max) / fsig) ** 2)
    dPi = np.sum(dPi * w / sum(w))
    return dPi


def correct_seismo(hist, gsspnum, mask, xname='center_he4', do_deltanu=True, do_deltaP=True,
                   prefix='profile', suffix='.data.GYRE.sgyre_l', get_poly=False, weight=True):
    """
    Calculate corrected delta_nu and delta_Pg using GyreSummary instances in gsspnum. The correction is done by
    fitting a 2nd order polynomial to the ratio of delta_nu from the History and from the GyreSummary instances and
    likewise for delta_Pg.

    Args:
        hist (History):
        gsspnum (Tuple of GyreSummary, int): Calculate using ``wsssss.load_data.load_gss(..., return_pnums=True, ...)``.
        mask (bool, np.array, or function, optional): If True, will exclude pre-main sequence.
                If an array of bools will use that as mask. If a function, will call function(hist) and use that as the mask.
        xname (str, optional): Domain to use when fitting.
        do_deltanu (bool, optional): If True, calculate the delta_nu correction. Defaults to True.
        do_deltaP (bool, optional): If True, calculate the delta_Pg correction. Defaults to True.
        prefix (str, optional): Part of gyre summary name before the profile number. Defaults to 'profile'.
        suffix (str, optional): Part of gyre summary name after the profile number. Defaults to '.data.GYRE.sgyre_l'.
        get_poly (bool, optional): If True, returns the correction polynomials instead of corrected values.
        weight (bool, optional): If True, weight points by distance in domain, giving sparse points more weight.

    Returns:
        If do_deltanu is True, returns the corrected delta_nus. If do_deltaP is True, also returns corrected delta_Pg.
        If get_poly is True, returns the the correction polynomials instead of corrected values.
    """
    mask = get_mask(hist, use_mask=mask)

    mnum_min, mnum_max = hist.data.model_number[mask][[0, -1]]

    x = []
    y_nu = []
    y_P = []
    deltanus = []
    deltaPs = []
    for gs, pnum in gsspnum:
        i_hist = hist.get_profile_index(pnum)
        if len(i_hist) == 0:  # pnum not in history
            continue
        i_hist = i_hist[0]
        mnum = hist.data.model_number[i_hist]
        if (mnum < mnum_min) or (mnum > mnum_max):
            continue
        if do_deltanu:
            deltanu = calc_deltanu(gs, hist, prefix=prefix, suffix=suffix)
            deltanus.append(deltanu)
            h_deltanu = hist.data.delta_nu[i_hist]
            y_nu.append(deltanu / h_deltanu)

        if do_deltaP:
            deltaP = calc_deltaPg(gs, hist, 1)
            deltaPs.append(deltaP)
            h_deltaP = hist.data.delta_Pg[i_hist]
            y_P.append(deltaP / h_deltaP)

        x.append(hist.get(xname)[i_hist])
    x = np.array(x)
    y_nu = np.array(y_nu)
    y_P = np.array(y_P)

    i_sort = np.argsort(x)
    x = x[i_sort]
    if weight:
        mid = (x[1:] + x[:-1]) / 2
        w = np.zeros_like(x)
        w[1:-1] = np.diff(mid)
        w[1:-1] = (x[2:] - x[:-2])
        w[0] = 2 * (x[1] - x[0])
        w[-1] = 2 * (x[-1] - x[-2])
        w = np.abs(w)
    else:
        w = np.ones_like(x)

    if do_deltanu:
        y_nu = y_nu[i_sort]
        try:
            nu_poly = np.poly1d(np.polyfit(x, y_nu, 2, w=w))
            nu_f = nu_poly(hist.get(xname)[mask])
            new_deltanu = hist.data.delta_nu[mask] * nu_f
        except TypeError:
            print(hist.path)
            raise
    if do_deltaP:
        y_P = y_P[i_sort]
        try:
            P_poly = np.polyfit(x, y_P, 2, w=w)
        except Exception:
            P_poly = np.array([np.nan] * 3)
        P_poly = np.poly1d(P_poly)
        P_f = P_poly(hist.get(xname)[mask])
        new_deltaP = hist.data.delta_Pg[mask] * P_f

    if do_deltanu and do_deltaP:
        if get_poly:
            return nu_poly, P_poly
        return new_deltanu, new_deltaP
    elif do_deltanu and not do_deltaP:
        if get_poly:
            return nu_poly
        return new_deltanu
    elif not do_deltanu and do_deltaP:
        if get_poly:
            return P_poly
        return new_deltaP


def calc_ModDens(numax, deltanu, deltap):
    """
    Calculate mode density.

    Args:
        numax:
        deltanu:
        deltap:

    Returns:
        Mode density
    """
    return deltanu / (numax ** 2 * 1e-6 * deltap)

def calc_abundance(hist, use_mask=None):
    """
    Calculate surface abundances. Returns abundances (by element, not isotope) and element names.
    Args:
        hist:
        use_mask (bool, np.array, or function, optional): If True, will exclude pre-main sequence.
            If an array of bools will use that as mask. If a function, will call function(hist) and use that as the mask.

    Returns:
        array: Array containing surface abundances.

    """
    mask = get_mask(hist, use_mask)
    cols = [col for col in hist.columns if col.startswith('surface')]
    elems = [re.sub('[0-9]', '', iso.split('_')[1].capitalize()) for iso in cols]
    A = np.zeros((sum(mask), len(np.unique(elems))))
    j = 0
    e0 = elems[0]
    for i, col in enumerate(cols):
        if col == 'surface_neut':
            m = 1
        else:
            e = re.sub('[0-9]', '', col.split('_')[1].capitalize())
            if e != e0:
                j += 1
                e0 = e
            m = float(re.sub('[A-z]', '', col.split('_')[1]))
        A[:, j] += hist.get(col)[mask] / m

    _, idx = np.unique(elems, return_index=True)
    elems = np.array(elems)[np.sort(idx)]

    A = np.log10(A)
    A = A - A[:, np.where(elems == 'H')[0][0]][:, np.newaxis] + 12
    return A, elems
