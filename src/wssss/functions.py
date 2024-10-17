#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import functools
import os
import dill
import numpy as np
import multiprocessing as mp
from scipy import integrate as ig
from scipy import interpolate as ip
from scipy import stats
from scipy.optimize import newton

from wssss.constants import post15140
from wssss.constants import pre15140


def get_mesa_version(mesa_dir):
    with open(f'{mesa_dir}/data/version_number', 'r') as handle:
        version = handle.read().strip()
    return version


def get_constants(p_or_hist):
    if 'version' in p_or_hist.header.keys():
        version = str(p_or_hist.header['version'])

    if version < '15140':
        return pre15140
    else:
        return post15140

# Mixing type codes for pre and post 15140
mix_dict = {'pre15140': {0: 'no_mixing',
                         1: 'convective_mixing',
                         2: 'softened_convective_mixing',
                         3: 'overshoot_mixing',
                         4: 'semiconvective_mixing',
                         5: 'thermohaline_mixing',
                         6: 'rotation_mixing',
                         7: 'rayleigh_taylor_mixing',
                         8: 'minimum_mixing',
                         9: 'anonymous_mixing'},
            'post15140': {0: 'no_mixing',
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
            'merged': {'no_mixing': 100,
                       'convective_mixing': 101,
                       'overshoot_mixing': 103,
                       'semiconvective_mixing': 104,
                       'thermohaline_mixing': 105,
                       'rotation_mixing': 106,
                       'minimum_mixing': 107,
                       'anonymous_mixing': 109},
            }


def stackdiff(x1, x2):
    return np.diff(np.hstack((x1, x2)))


def cell2face(val, dm, dm_is_m=False, m_center=0):
    if dm_is_m:
        dm = np.diff(dm, append=m_center)
    face = np.zeros_like(val)
    face[0] = val[0]
    face[1:] = (val[:-1]*dm[1:] + val[1:]*dm[:-1]) / (dm[1:] + dm[:-1])
    return face


def dlog10y_dlog10x(y, x):
    return np.gradient(np.log10(y)) / np.gradient(np.log10(x))


def dy_dx(y, x):
    return np.gradient(y) / np.gradient(x)


def dlog10y_dx(y, x):
    return np.gradient(np.log10(y)) / np.gradient(x)


def dy_dlog10x(y, x):
    return np.gradient(y) / np.gradient(np.log10(x))


def signed_log10(q):
    return np.sign(q)*np.log10(np.maximum(np.abs(q), 1))


def prof2i_hist(prof, hist):
    mod = prof.header['model_number']
    return np.argwhere(hist.get('model_number') == mod)


def calc_logQ(prof):
    return prof.get('logRho') - 2*prof.get('logT') + 12


def get_logTeffL(hist, mask=None):
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
            logL = 2*np.log10(R) + 4 * logTeff - 4 * np.log10(5777)
        else:
            raise ValueError('log_L, luminosity, or photosphere_L not in history file.')
    mask = get_mask(hist, mask)
    return logTeff[mask], logL[mask]


def get_radius(p, unit='cm'):
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
        radius = np.log10(radius)
    elif unit.lower().replace('_', '') in ['rsun', 'rsol']:
        radius = radius/c.rsun
    return radius


def get_m_bot_CZ(hist, mask=None):
    if 'm_botCZ' in hist.cols:
        return hist.get('m_botCZ', mask=mask)
    m_bot_CZ = np.zeros_like(hist.get('star_mass', mask=mask))
    n_mix = 0
    for col in hist.cols:
        if col.startswith('mix_qtop'):
            n_mix += 1
    for i in range(n_mix):
        i += 1
        m_bot_CZ = np.maximum(m_bot_CZ, hist.get(f'mix_qtop_{i}', mask=mask) * (hist.get(f'mix_type_{i}', mask=mask) == 3))
    return m_bot_CZ


def get_cheb_mask(hist, min_Yc=1e-6, max_Yc=1.0):
    """
    Defined as:
    have helium in core
    have convective core
    hydrogen depleted core


    :param hist:
    :param min_Yc:
    :param max_Yc:
    :return:
    """
    
    mask = hist.data.center_he4 >= min_Yc
    mask = np.logical_and(mask, hist.data.center_he4 <= max_Yc)
    mask = np.logical_and(mask, hist.data.mass_conv_core > 0)
    mask = np.logical_and(mask, hist.data.center_h1 < 1e-6)
    
    return mask


def get_rc_mask(hist, min_Yc=0.1, max_Yc=0.95, first_chunk=False):
    """
    Defined as:
    have helium in core
    have convective core
    hydrogen depleted core


    :param hist:
    :param min_Yc:
    :param max_Yc: Fraction of maximum Yc to start RC at (let structure settle before calling it RC)
    :param: first_chunk: Only take first contiguous block of True.
    :return:
    """

    mask = hist.data.center_he4 >= min_Yc
    mask = np.logical_and(mask, hist.data.center_he4 <= max_Yc*max(hist.data.center_he4))
    mask = np.logical_and(mask, hist.data.mass_conv_core > 0)
    mask = np.logical_and(mask, hist.data.center_h1 < 1e-6)

    if first_chunk and np.any(mask):
        end_first_block = np.where(np.diff(mask, prepend=False))[0][1]
        mask[end_first_block+1:] = False
    return mask


def get_pms_mask(hist, invert=False, use_LH=True):
    """
    Defined as X_c >= X_c_init*0.99
    if use_LH is True, then pms stops when LH == L
    :param hist:
    :param invert: Get post pms mask.
    :param use_LH: Use LH condition for end of PMS.
    :return:
    """

    if use_LH:
        _, logL = get_logTeffL(hist)
        maskL = ((hist.data.log_LH - logL) < 0) & (hist.data.center_h1 > 0.6)
        starts_pms =  maskL[0]

        mask = np.zeros_like(maskL)
        if starts_pms:
            end = np.where(np.diff(maskL))[0][0]
            mask[:end+1] = True
    else:
        mask = hist.data.center_h1 >= hist.data.center_h1[0] * 0.99

    if invert:
        mask = ~mask

    return mask


def get_ms_mask(hist, min_Xc=1e-3, use_LH=True):
    """
    Defined as:
    X_c > 1e-3 and after end of pms

    :param hist:
    :param min_Xc: minimum Xc for end of MS
    :param use_LH: which condition to use for start of MS, passed to get_pms_mask.
    :return:
    """
    
    mask = hist.data.center_h1 > min_Xc
    mask = np.logical_and(mask, get_pms_mask(hist, invert=True, use_LH=use_LH))
    
    return mask


def get_sgb_mask_old(hist, min_dmhecore_dlnt=0.05, min_Xc=1e-3, max_logT_lim=3.8, min_logT_lim=3.6):
    ms_mask = get_ms_mask(hist, min_Xc)
    if np.any(ms_mask):
        min_mod = hist.data.model_number[ms_mask][-1]
    else:
        if 'center_Rho' in hist.cols:
            center_Rho = hist.get('center_Rho')
        elif 'log_center_Rho' in hist.cols:
            center_Rho = 10 ** hist.get('log_center_Rho')
        if np.log10(center_Rho[0]) >= 3.5:  # starts during or after SGB
            return np.zeros_like(ms_mask, dtype=bool)
        min_mod = hist.data.model_number[0]

    bump_mask = get_bump_mask(hist, max_logT_lim, min_logT_lim)
    if not np.any(bump_mask):
        tip_mask = get_tip_mask(hist)
        max_mod = hist.data.model_number[tip_mask][0]
    else:
        max_mod = hist.data.model_number[bump_mask][0]

    mask = np.logical_and(hist.data.model_number > min_mod, hist.data.model_number < max_mod)

    age = hist.data.star_age
    dmhecore_dlnt = age*np.gradient(hist.data.he_core_mass, age)
    dmhecore_dlnt[dmhecore_dlnt <= 0] = 0
    smoothed = np.roll(dmhecore_dlnt, -3) * np.roll(dmhecore_dlnt, -2) * np.roll(dmhecore_dlnt, -1) * dmhecore_dlnt * np.roll(dmhecore_dlnt, 1)
    smoothed[smoothed <= 0] = 0

    mask = np.logical_and(mask, smoothed >= min_dmhecore_dlnt)

    conv_mix_q = np.zeros_like(age[mask])
    has_q = np.zeros_like(age[mask], dtype=bool)
    for i in range(4):
        mix_type = hist.get(f'mix_type_{i+2}')[mask]
        mix_mask = mix_type == 1  # Convective mixing
        needs_data = np.logical_and(mix_mask, ~has_q)
        conv_mix_q[needs_data] = hist.get(f'mix_qtop_{i+1}')[mask][needs_data]
        has_q[needs_data] = True
        if i > 1 and np.all(has_q):
            break
    
    if not np.any(mask):
        return np.zeros_like(ms_mask, dtype=bool)
    
    i_lowest_CZ = np.where(mask)[0][np.argmin(conv_mix_q)]
    mask = hist.data.model_number >= min_mod
    mask[i_lowest_CZ+1:] = False

    return mask

def get_sgb_mask(hist, min_Xc=1e-3, fCZ=0.35):
    ms_mask = get_ms_mask(hist, min_Xc)
    if np.any(ms_mask):
        min_mod = hist.data.model_number[ms_mask][-1]
    else:
        if 'center_Rho' in hist.cols:
            center_Rho = hist.get('center_Rho')
        elif 'log_center_Rho' in hist.cols:
            center_Rho = 10 ** hist.get('log_center_Rho')
        if np.log10(center_Rho[0]) >= 3.5:  # starts during or after SGB
            return np.zeros_like(ms_mask, dtype=bool)
        min_mod = hist.data.model_number[0]

    try:
        mask = hist.data.m_botCZ/hist.data.star_mass <= (1-fCZ)
    except AttributeError:
        m_bot_CZ = get_m_bot_CZ(hist)
        mask = m_bot_CZ / hist.data.star_mass <= (1 - fCZ)
    mask = mask & (hist.data.model_number > min_mod)
    max_mod = hist.data.model_number[mask][0]

    mask = np.logical_and(hist.data.model_number > min_mod, hist.data.model_number < max_mod)
    return mask

def get_bump_mask(hist, max_logT_lim=3.8, min_logT_lim=3.6, min_logL=0.5):
    """

    :param hist:
    :param max_logT_lim
    :param min_logT_lim
    :return:
    """

    logTeff, logL = get_logTeffL(hist)
    
    mask = hist.data.center_h1 < 1e-9
    mask = np.logical_and(mask, hist.data.center_he4 > 0.95)
    mask = np.logical_and(mask, logTeff < max_logT_lim)
    mask = np.logical_and(mask, logTeff > min_logT_lim)
    mask = np.logical_and(mask, logL > min_logL)
    
    if sum(mask) == 0:
        return np.zeros_like(mask, dtype=bool)
    
    mod_min = min(hist.get('model_number')[mask])
    try:
        mod_max = min(hist.get('model_number')[get_tip_mask(hist, max_logT_lim)])  # before RGB tip
    except (IndexError, ValueError):  # no flashes, use cheb instead
        mod_max = hist.data.model_number[get_cheb_mask(hist)][0]
    
    # Bump is somewhere in the following mask:
    mask = np.logical_and(mask, hist.get('model_number') >= mod_min)
    mask = np.logical_and(mask, hist.get('model_number') <= mod_max)
    
    # Bump is defined as when L decreases and Teff increases on RGB
    # submask_bump = np.logical_and(np.diff(logL[mask], prepend=-99) < 0, np.diff(logTeff[mask], prepend=99) > 0)
    # Bump is defined as when Teff increases on RGB
    submask_bump = np.diff(logTeff[mask], prepend=99) > 0
    mask[mask] = submask_bump
    
    return mask


def get_tip_mask(hist, logT_lim=3.8):
    
    logTeff, logL = get_logTeffL(hist)

    mask = hist.data.center_h1 < 1e-9
    mask = np.logical_and(mask, hist.data.center_he4 > 0.95)
    mask = np.logical_and(mask, logTeff < logT_lim)
    
    if sum(mask) == 0:
        return np.zeros_like(mask, dtype=bool)
    
    minT = min(logTeff[mask])
    maxL = max(logL[mask])
    
    mask = np.logical_and(mask, logTeff < minT + 0.01)
    mask = np.logical_and(mask, logL > maxL - 0.1)
    
    return mask


def get_rgb_mask(hist, min_Xc=1e-3, logT_lim=3.8, old_sgb=False):
    try:
        if old_sgb:
            sgb_mask = get_sgb_mask_old(hist, min_Xc)
        else:
            sgb_mask = get_sgb_mask(hist, min_Xc)
        if np.any(sgb_mask):
            first_mod = max(hist.data.model_number[sgb_mask])
        else:
            first_mod = hist.data.model_number[0]
    except IndexError:
        first_mod = hist.data.model_number[0]
    
    tip_mask = get_tip_mask(hist, logT_lim)
    if np.any(tip_mask):
        logTeff, logL = get_logTeffL(hist)
        masked_i_min_T = np.argmin(logTeff[tip_mask])
        last_mod = hist.data.model_number[tip_mask][masked_i_min_T]
        mask = np.logical_and(hist.data.model_number > first_mod, hist.data.model_number <= last_mod)
    else:
        mask = np.zeros_like(tip_mask, dtype=bool)

    return mask


def get_flashes_mask(hist, logT_lim=3.8):
    rc_mask = get_cheb_mask(hist)
    if np.any(rc_mask):
        last_mod = min(hist.data.model_number[rc_mask])
    else:
        print('No CHeB model, using last.')
        last_mod = hist.data.model_number[-1]
    
    tip_mask = get_tip_mask(hist, logT_lim)
    logTeff, logL = get_logTeffL(hist)
    masked_i_min_T = np.argmin(logTeff[tip_mask])
    first_mod = hist.data.model_number[tip_mask][masked_i_min_T]
    
    mask = np.logical_and(hist.data.model_number > first_mod, hist.data.model_number < last_mod)
    return mask


def get_agb_mask(hist):
    rc_mask = get_cheb_mask(hist)
    if np.any(rc_mask):
        mod_agb_start = max(hist.data.model_number[rc_mask])
        mask = hist.data.model_number > mod_agb_start
    else:
        mask = np.zeros_like(rc_mask, dtype=bool)
    return mask


mask_functions = [get_pms_mask, get_ms_mask, get_sgb_mask, get_rgb_mask, get_bump_mask, get_tip_mask, get_flashes_mask, get_cheb_mask,
                  get_rc_mask, get_agb_mask]
mask_names = ['PMS', 'MS', 'SGB', 'RGB', 'RGBb', 'RGB tip', 'He flashes', 'CHeB', 'RC', 'post-CHeB']

def calc_dimless_to_Hz(gs):
    if gyre_version < '6':
        G = pre15140.standard_cgrav
    else:
        G = post15140.standard_cgrav
    if 'M_star' in gs.header.keys():
        M_star = gs.header['M_star']
        R_star = gs.header['R_star']
    else:
        M_star = gs.get('M_star')[0]
        R_star = gs.get('R_star')[0]
    return 1.0 / (2 * np.pi) * ((G * M_star / (R_star) ** 3))


def get_freq(gs, freq_units='uHz', kind='summary'):
    if kind == 'summary':
        if 'Re(omega)' in gs.columns:
            dimless_to_Hz = calc_dimless_to_Hz(gs) * {'uHz': 1e6, 'mHz': 1e3, 'Hz': 1e0}[freq_units]
            freq_name = 'Re(omega)'
            freq_unit = {'uHz': r'\mu', 'mHz': 'm', 'Hz': ''}[freq_units]
        elif 'Re(freq)' in gs.columns:  # Assumes freq already in uHz.
            dimless_to_Hz = {'uHz': 1e0, 'mHz': 1e-3, 'Hz': 1e-6}[freq_units]
            freq_name = 'Re(freq)'
            freq_unit = {'uHz': r'\mu', 'mHz': 'm', 'Hz': ''}[freq_units]
        else:
            raise ValueError(f"Can't find frequency column in {gs}.")
        return gs.data[freq_name] * dimless_to_Hz
    elif kind == 'mode':
        if 'Re(freq)' in gs.header:  # Assumes freq already in uHz.
            dimless_to_Hz = {'uHz': 1, 'mHz': 1e-3, 'Hz': 1e-6}[freq_units]
            freq_name = 'Re(freq)'
            freq_unit = {'uHz': r'\mu', 'mHz': 'm', 'Hz': ''}[freq_units]
        elif 'Re(omega)' in gs.header:
            dimless_to_Hz = calc_dimless_to_Hz(gs) * {'uHz': 1e6, 'mHz': 1e3, 'Hz': 1e0}[freq_units]
            freq_name = 'Re(omega)'
            freq_unit = {'uHz': r'\mu', 'mHz': 'm', 'Hz': ''}[freq_units]
        else:
            raise ValueError(f"Can't find frequency column in {gs}.")
        return gs.header[freq_name] * dimless_to_Hz
    else:
        raise ValueError(f'{kind} is not summary or mode.')
    
    
def get_gridnum(hist):
    return int(os.path.split(hist.LOGS)[0][-4:])


def get_mask(hist, use_mask):
    if hasattr(use_mask, '__len__'):
        if len(use_mask) == len(hist.get('model_number')):
            mask = use_mask
        else:
            # Try to get numpy to raise the index error first
            _ = hist.get('model_number')[use_mask]
            raise IndexError(f'Length of mask not the same as length of data ({len(use_mask)} vs {len(hist.get("model_number"))}')
    elif callable(use_mask):
        mask = use_mask(hist)
    else:
        if use_mask:
            mask = get_pms_mask(hist, invert=True)
        else:
            mask = np.ones_like(hist.get('model_number'), dtype=bool)
    return mask


def get_mean(hist, name, use_mask=None, domain='star_age', filter=None, get_std=False):
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
        print(f"Warning: 0-length data when calculating mean of {domain}")
        mean = np.nan
        std = np.nan
    else:
        mean = ig.trapz(ydat, xdat) / (xdat[-1] - xdat[0])
        std = (ig.trapz((ydat-mean)**2, xdat) / (xdat[-1] - xdat[0]))**0.5
        std = np.mean(np.diff(np.quantile(ip.interp1d(xdat, ydat)(np.linspace(min(xdat), max(xdat), 201)),[0.15865, 0.50, 0.84135])))
    if get_std:
        return mean, std
    else:
        return mean


def get_weighted_quantile(x, w, q=(0.15865, 0.50, 0.84135)):
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
        logTr = -0.084 * logL -0.012 * np.log10(Z) + 3.879
        logTb = -0.080 * logL - 0.012 * np.log10(Z) + 3.957

        return logL, (logTb, logTr)

    elif kind == 'Cepheid':
        # https://arxiv.org/pdf/astro-ph/9801242.pdf

        (Xref, Zref) = (0.7, 0.004)
        Yref = 1 - Xref - Zref
        deltaY = Y - Yref
        deltaZ = Z - Zref

        logL = np.linspace(logL_min, logL_max, 101)
        logTeff = -0.036 * logL+ 3.925
        dlogTeff = 0.04 * deltaY - 0.49*deltaZ
        logTeff += dlogTeff

    return logL, (logTeff, logTeff - 0.06)


def get_evo_phase(hist, phase_funcs):
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
        factor = -1*xdata
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
    version = str(p.header['version_number'])
    radius = get_radius(p, 'Rsol')
    mass = p.get('mass')
    temperature = p.get('temperature')
    mix_type = p.get('mixing_type')
    
    if version >= '15140':
        prefix = 'post'
    else:
        prefix = 'pre'
    mix_type = np.array(list(map(mix_dict['merged'].get, map(mix_dict[f'{prefix}15140'].get, mix_type))))
    mix_OS = mix_type == 103  # OS
    mix_CV = mix_type == 101  # convective
    temp_mask = (temperature[1:] > p.header['Teff'] * 2)
    radiative = False
    try:
        bottom_of_CZ = np.where(mix_CV[:-1] & mix_OS[1:] & temp_mask)[0][0]
        bottom_of_US = np.where(mix_OS[:-1] & (mix_type[1:] != 101) & (mix_type[1:] != 103) & temp_mask)[0][0]
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
        lamb2 = p.data.lamb_S2
    elif 'lamb_Sl1' in p.columns:
        lamb2 = (p.data.lamb_Sl1/(1e6/(2*np.pi)))**2 * l*(l+1)/2  # l part to convert from l=1 to l=l
    else:
        radius = get_radius(p)
        lamb2 = l*(l+1) * (p.data.csound/radius)**2
    return lamb2


def calc_r0line_fCZ(freq, r, Nred, Sred, r_bCZ):  # assumes well behaved Nred, Sred
    fNred = ip.interp1d(Nred, r, bounds_error=False)
    fSred = ip.interp1d(Sred, r, bounds_error=False)

    r1 = fSred(freq)
    r2 = fNred(freq)
    r0 = np.sqrt(r1*r2)
    s0 = 0.5*(np.log(r1) - np.log(r2))
    f_CZ = (s0 - np.log(r_bCZ/r0))/(2*s0)
    f_CZ = np.minimum(f_CZ, 1)
    f_CZ = np.maximum(f_CZ, 0)
    return r0, f_CZ


def get_X_from_q(q, kind='strong'):
    if kind == 'strong':
        return np.log(1 - ((1-q)/(1+q))**2)/(-2*np.pi)
    elif kind == 'weak':
        if np.any(q <= 0.25):
            return np.log(4*q)/(-2*np.pi)
        else:
            return ValueError('q must be less than 0.25 for kind weak.')
    else:
        raise ValueError('`kind` must be `strong` or `weak`.')


def calc_q(X):
    return (1 - np.sqrt(1 - np.exp(-2 * np.pi * X))) / (1 + np.sqrt(1 - np.exp(-2 * np.pi * X)))

#
# def calc_qP(hist, integral='MESA'):
#     if integral == 'MESA':
#         Xi = hist.get('X_integral_part')
#     else:
#         Xi = calc_PQ_int_frm_beta(hist, 'parallel')/np.pi
#     dlnc_ds_s0_part_Sl = hist.get('dlnc_ds_s0_part_Sl')
#     NuAJ_s0 = hist.get('NuAJ_s0')
#     Xg = (dlnc_ds_s0_part_Sl - 0.5 * NuAJ_s0)**2 / (2*hist.get('kappa_s0'))
#     X = Xi + Xg
#     return calc_q(X)

def calc_J0(hist):
    # Mean J in EZ
    rho_EZ = (hist.data.m_2 - hist.data.m_1) / (4/3*np.pi * (hist.data.r_2**3 - hist.data.r_1**3))
    r0 = calc_r0(hist)
    m0 = np.sqrt(hist.data.m_2*hist.data.m_1)  # best we can do, <1% error for f_CZ<0.2
    rho_mean = m0 / (4/3 * np.pi * r0**3)
    J0 = 1 - rho_EZ/rho_mean
    return J0
def get_betaN_plus_betaS(hist):
    J0 = calc_J0(hist)
    P0 = hist.data.a_0_int_P
    Q0 = hist.data.a_0_int_Q
    # return -0.5 * np.log((J0 - Q0) / (J0 - P0 / 2)) / calc_s0(hist)
    return -0.5 * np.log((1 - Q0/J0)*(1 - P0/(2*J0))) / calc_s0(hist)

def get_beta_from_dlnc(hist, which='S'):
    r_1 = hist.get('r_1')
    r_2 = hist.get('r_2')

    # if which in ['S', 'N']:
    #     J0 = calc_J0(hist)
    #     P0 = hist.data.a_0_int_P
    #     Q0 = hist.data.a_0_int_Q
    #     if which == 'S':
    #         return np.log(1 - P0 / (2 * J0)) / np.log(r_2 / r_1)
    #     elif which == 'N':
    #         return np.log(1 - Q0 / J0) / np.log(r_2 / r_1)
    if which == 'S':
        dlncds = hist.data.dlnc_ds_s0_part_Sl
    elif which == 'N':
        dlncds = hist.data.dlnc_ds_s0_part_N
    elif which == 'q': # Get equivalent beta for full strong q
        s0 = calc_s0(hist)
        dlncds = 0.25 * (hist.data.dlnPds_s0 - hist.data.dlnQds_s0)  # 2/s0 is already in dlnPQ parts

    def func(beta, r_1, r_2, dlncds):
        alfa = (r_2/r_1) ** beta
        return abs(dlncds - (-beta * (
                    alfa / (1 - alfa) + 1 / np.log(alfa))))

    beta = newton(func, x0=1.5*np.ones_like(r_1), args=(r_1, r_2, dlncds))  # gives betaS
    # if which == 'N':
    #     beta = get_betaN_plus_betaS(hist) - beta  # behaves better than doing newton for betaN directly
    return beta

def calc_PQ_int_frm_beta(hist, kind='parallel', num=101):
    r1 = hist.data.r_1
    r2 = hist.data.r_2
    r0 = np.sqrt(r1*r2)
    J0 = calc_J0(hist)
    betaS = get_beta_from_dlnc(hist)
    if kind == 'parallel':
        betaN = betaS
    elif kind.lower().startswith('non'):
        betaN = get_beta_from_dlnc(hist, 'N')
    r = np.logspace(np.log(np.minimum(r1, r2)), np.log(np.maximum(r1, r2)), num, base=np.e)
    sqrtPQ_div_r = np.sqrt(2 * J0**2 * (1 - (r1/r)**(-2*betaS)) * (1 - (r2/r)**(2*betaN)))/r
    sqrtPQ_div_r[0, :] = 0  # Force to equal 0 at s = pm s_0
    sqrtPQ_div_r[-1, :] = 0
    integ = ig.simps(sqrtPQ_div_r, r, axis=0)
    return integ


def calc_q_app(hist, integral='MESA', which='non-parallel', output='q', fix_bugged_betaN=False):
    if integral == 'MESA':
        Xi = hist.get('X_integral_part')
    elif integral.startswith('approx'):
        Xi = calc_PQ_int_frm_beta(hist, which)/np.pi
    else:
        raise ValueError('integral must be MESA or approx.')

    r1, r2 = get_r1r2(hist)

    betaS = get_beta_from_dlnc(hist, which='S')
    if which.lower().startswith('non'):
        betaN = get_beta_from_dlnc(hist, which='N')
        if fix_bugged_betaN:
            betaN = betaN/2  # fix a missing sqrt in rse betaN calculation
    else:
        betaN = betaS
    gamma = betaN / betaS
    alphaP = (r2 / r1) ** (betaS)
    dlnc_ds_s0_part_noP = -betaS * ((gamma * alphaP ** gamma - alphaP * (alphaP ** gamma * (1 + gamma) - 1)) / (
                2 * (alphaP - 1) * (alphaP ** gamma - 1)) + 1 / np.log(alphaP))

    Xg = (dlnc_ds_s0_part_noP - 0.5 * hist.get('NuAJ_s0')) ** 2 / (2 * hist.get('kappa_s0'))
    X = Xi + Xg

    if output == 'full':
        return calc_q(X), Xi, Xg, betaS, betaN,  alphaP
    else:
        return calc_q(X)

def get_r1r2(hist, mask=None):
    try:
        r1 = hist.get('r_1', mask=mask)
        r2 = hist.get('r_2', mask=mask)
    except ValueError:
        r1 = hist.get('r_1_strong', mask=mask)
        r2 = hist.get('r_2_strong', mask=mask)
    return  r1, r2

def get_m1m2(hist, mask=None):
    try:
        m1 = hist.get('m_1', mask=mask)
        m2 = hist.get('m_2', mask=mask)
    except ValueError:
        m1 = hist.get('m_1_strong', mask=mask)
        m2 = hist.get('m_2_strong', mask=mask)
    return  m1, m2

def calc_s0(hist, mask=None):
    r1, r2 = get_r1r2(hist, mask)
    return np.log(r1/r2)/2


def calc_r0(hist, mask=None):
    r1, r2 = get_r1r2(hist, mask)
    return np.sqrt(r1*r2)

def calc_int_PQ_quad(hist, mask=None, n=21):
    mask = get_mask(hist, mask)
    int_P0 = hist.data.a_0_int_P[mask][:, np.newaxis]
    int_P1 = hist.data.a_1_int_P[mask][:, np.newaxis]
    int_P2 = hist.data.a_2_int_P[mask][:, np.newaxis]
    int_Q0 = hist.data.a_0_int_Q[mask][:, np.newaxis]
    int_Q1 = hist.data.a_1_int_Q[mask][:, np.newaxis]
    int_Q2 = hist.data.a_2_int_Q[mask][:, np.newaxis]

    PQ_quad = np.zeros((sum(mask), n))
    x = np.linspace(-1, 1, n)
    xx = np.ones((sum(mask), n))
    s0 = calc_s0(hist, mask)
    ss = np.abs(s0[:, np.newaxis]) * x[:, np.newaxis].T * xx

    PQ = (int_P0 + int_P1 * ss + int_P2 * ss ** 2) * (int_Q0 + int_Q1 * ss + int_Q2 * ss ** 2)
    PQ[:, 0] = 0
    PQ[:, -1] = 0
    PQ = np.sqrt(PQ)

    I_quad = ig.simpson(PQ, ss, axis=1)
    return I_quad/np.pi
def calc_q_grad(hist, mask=None):
    Xi = hist.get('X_integral_part', mask=mask)
    NuAJ_s0 = hist.get('NuAJ_s0', mask=mask)
    
    s_0 = calc_s0(hist, mask=mask)
    
    P_s0 = hist.get('a_0_int_P', mask=mask)
    Q_s0 = hist.get('a_0_int_Q', mask=mask)
    dlnPds_s0 = hist.get('a_0_grd_P', mask=mask) / (P_s0 / s_0)
    dlnQds_s0 = hist.get('a_0_grd_Q', mask=mask) / (Q_s0 / s_0)

    kappa_s0 = np.sqrt(P_s0 * Q_s0) / abs(s_0)
    dlnc_ds_s0_part = 0.25 * (dlnPds_s0 - dlnQds_s0 + 0)
    
    Xg = (dlnc_ds_s0_part - 0.5 * NuAJ_s0)**2 / (2*kappa_s0)
    X = Xi + Xg
    return calc_q(X)

def calc_q_num(hist, mask=None):
    Xi = hist.get('X_integral_part', mask=mask)
    NuAJ_s0 = hist.get('NuAJ_s0', mask=mask)
    
    s_0 = calc_s0(hist, mask=mask)

    P_s0 = hist.get('a_0_int_P', mask=mask)
    Q_s0 = hist.get('a_0_int_Q', mask=mask)
    kappa_s0 = np.sqrt(P_s0 * Q_s0) / abs(s_0)
    dlnc_ds_s0_part = hist.get('dlnc_ds_s0_part_ip', mask=mask)
    
    Xg = (dlnc_ds_s0_part - 0.5 * NuAJ_s0) ** 2 / (2 * kappa_s0)
    X = Xi + Xg
    return calc_q(X)


def calc_qw(X):
    return 0.25*np.exp(-2*np.pi*X)


def load_meshdat(base_dir):
    if hasattr(base_dir, 'LOGS'):
        base_dir = f'{base_dir.LOGS}/'
    try:
        unsmoothed = np.loadtxt(f'{base_dir}/unsmoothed.dat')
        smoothed = np.loadtxt(f'{base_dir}/smoothed.dat')
        meshed = np.loadtxt(f'{base_dir}/smoothed_meshed.dat')
    except Exception:
        base_dir = f'{base_dir}/../'
        unsmoothed = np.loadtxt(f'{base_dir}/unsmoothed.dat')
        smoothed = np.loadtxt(f'{base_dir}/smoothed.dat')
        meshed = np.loadtxt(f'{base_dir}/smoothed_meshed.dat')
    
    from collections import Counter
    
    uns = []
    smo = []
    msh = []
    
    for i_type, dat in enumerate([unsmoothed, smoothed, meshed]):
        cnt = Counter(dat[:, 0])
        min_npoints = min(cnt.values())
        cum_pts = 0
        if i_type == 0:
            new_dat = uns
        elif i_type == 1:
            new_dat = smo
        elif i_type == 2:
            new_dat = msh
        for j, (mnum, npoints) in enumerate(cnt.items()):
            sub_dat = dat[cum_pts:cum_pts + npoints]
            mod = sub_dat[0][0]
            
            x1 = sub_dat[:min_npoints // 2, 2]
            y1 = sub_dat[:min_npoints // 2, 3]
            x2 = sub_dat[min_npoints // 2:min_npoints, 2]
            y2 = sub_dat[min_npoints // 2:min_npoints, 3]
            new_dat.append([mnum, x1, y1, x2, y2])
            cum_pts += npoints
    uns = np.array(uns, dtype=object)
    smo = np.array(smo, dtype=object)
    msh = np.array(msh, dtype=object)
    return uns, smo, msh


def load_meshdat2(base_dir, oldstyle=False):
    if hasattr(base_dir, 'LOGS'):
        max_mnum = base_dir.data.model_number[-1]
        base_dir = f'{base_dir.LOGS}/'
    else:
        max_mnum = 1e99
    try:
        unsmoothed = np.loadtxt(f'{base_dir}/unsmoothed.dat')
        if oldstyle:
            smoothed = np.loadtxt(f'{base_dir}/smoothed.dat')
        else:
            smoothed = [None]
        try:
            meshed = np.loadtxt(f'{base_dir}/meshed.dat')
        except OSError:
            meshed = np.loadtxt(f'{base_dir}/smoothed_meshed.dat')
    except Exception:
        base_dir = f'{base_dir}/../'
        unsmoothed = np.loadtxt(f'{base_dir}/unsmoothed.dat')
        if oldstyle:
            smoothed = np.loadtxt(f'{base_dir}/smoothed.dat')
        else:
            smoothed = [None]
        try:
            meshed = np.loadtxt(f'{base_dir}/meshed.dat')
        except OSError:
            meshed = np.loadtxt(f'{base_dir}/smoothed_meshed.dat')
    
    uns = []
    smo = []
    msh = []
    
    for i_type, dat in enumerate([unsmoothed, smoothed, meshed]):
        if dat[0] is None:
            continue
        npoints = int((len(dat[0]) - 2) / 2)
        if i_type == 0:
            new_dat = uns
        elif i_type == 1:
            new_dat = smo
        elif i_type == 2:
            new_dat = msh
        
        mnum = dat[::2, 0]
        xP = dat[::2, 2:2+npoints]
        yP = dat[::2, 2+npoints:]
        xQ = dat[1::2, 2:2+npoints]
        yQ = dat[1::2, 2+npoints:]
        
        new_dat.extend([[mnum[i], xP[i], yP[i], xQ[i], yQ[i]] for i in range(len(mnum)) if (mnum[i] != mnum[i-2]) and (mnum[i] <= max_mnum)])
    uns = np.array(uns, dtype=object)
    smo = np.array(smo, dtype=object)
    msh = np.array(msh, dtype=object)
    return uns, smo, msh


def calc_f_CZ_from_hist(hist, mask=None):
    r1, r2 = get_r1r2(hist, mask)
    try:
        r_bCZ = hist.get('r_botCZ', mask=mask)
        f_CZ = calc_fCZ(r1, r2, r_bCZ)
    except ValueError:
        f_CZ = ~(1 + np.sign((r1 - r2) * hist.get('dlnQds_s0', mask=mask))).astype(bool)
    return f_CZ


def calc_fCZ(r1, r2, r_bCZ):
    r0 = np.sqrt(r1 * r2)
    s_bCZ = np.log(r_bCZ / r0)
    s0 = 0.5 * (np.log(r1) - np.log(r2))
    f_CZ = (abs(s0) - s_bCZ) / (2 * abs(s0))
    f_CZ = np.maximum(0, f_CZ)
    f_CZ = np.minimum(f_CZ, 1)
    return f_CZ

def get_coupling(hist, f=0.2):
    f_CZ = calc_fCZ_from_hist(hist)
    
    mask_f = f_CZ < f
    mask_1mf = f_CZ > 1 - f
    mask_inter = (f_CZ >= f) & (f_CZ <= 1 - f)
    
    kind = np.zeros_like(hist.get('star_age'), dtype=int)
    kind[mask_f] = 1  # strong
    kind[mask_1mf] = 2  # weak
    kind[mask_inter] = 3  # inter/two evn
    
    try:
        mask_sgbrgb = get_sgb_mask(hist) | get_rgb_mask(hist)
    except ValueError:
        mask_sgbrgb = np.zeros_like(mask_f, dtype=bool)
    if 'center_Rho' in hist.cols:
        center_Rho = hist.get('center_Rho')
    elif 'log_center_Rho' in hist.cols:
        center_Rho = 10 ** hist.get('log_center_Rho')
    mask = mask_sgbrgb & (hist.get('X_integral_part') >= 0) & (hist.get('X_integral_part_b') >= 0) & (
                hist.get('k_u2b') * hist.get('k_l2b') > 0) & (center_Rho > 1e4)
    if np.any(mask):
        i0 = np.where(mask_sgbrgb & (kind == 1) &
                      (hist.get('X_integral_part_b') > 0) &
                      (center_Rho > 1e4))[0][0]
        i0 -= 1
        
        # mask = mask & (hist.data.star_age <= (hist.data.star_age[i0] + 5e8))
        i1 = np.where(mask_sgbrgb & (kind == 2))[0][0]
        i1 += 1
        
        mask = np.zeros_like(mask)
        mask[i0:i1 + 1] = True
        kind[i0:i1] = 3
        
        print(hist.path, (hist.data.star_age[i1] - hist.data.star_age[i0]) / 1e6)
        
        q0 = hist.get('coupling_strong')[i0]
        # q1 = hist.get('coupling_strong')[i1]
        q1 = 0.25 * np.exp(-2 * np.pi * (hist.get('X_integral_part')[i1]))
        
        x = hist.get('star_age')
        x0 = x[i0]
        x1 = x[i1]
        
        ip_func = ip.interp1d([x0, x1], [q0, q1])
        
        new_q = hist.get('coupling_strong')[:]
        new_q[mask] = ip_func(x[mask])
    else:
        new_q = hist.get('coupling_strong')[:]
    
    # true q blends qs and qw across transition to conv ez
    true_q = np.nan * np.zeros_like(new_q)
    true_q[mask_f] = hist.get('coupling_strong')[mask_f]
    true_q[mask_1mf] = 0.25 * np.exp(-2 * np.pi * hist.get('X_integral_part')[mask_1mf])
    
    ip_func = ip.interp1d(hist.get('star_age')[~(kind == 3)], true_q[~(kind == 3)], bounds_error=False)
    true_q = ip_func(hist.get('star_age'))
    
    return new_q, mask, true_q, f_CZ, kind


def get_coupling2(hist, f=0.2, remove_extra=5, do_blend=False, rgb_type=None, interp_window=5, skip_rgbsgb=False, old_sgb=True):
    mask_cheb = get_cheb_mask(hist)
    if skip_rgbsgb:
        mask_sgbrgb = np.zeros_like(mask_cheb, dtype=bool)
    else:
        if old_sgb:
            mask_sgb = get_sgb_mask_old(hist)
        else:
            mask_sgb = get_sgb_mask(hist)
        mask_rgb = get_rgb_mask(hist)
        mask_sgbrgb = mask_sgb | mask_rgb

    if (rgb_type is None) or (rgb_type == ''):
        interp_rgb = False
        do_sum = False
    elif rgb_type == 'sum':
        do_sum = True
        interp_rgb = True
    elif rgb_type == 'interp':
        do_sum = False
        interp_rgb = True

    if 'center_Rho' in hist.cols:
        center_Rho = hist.get('center_Rho')
    elif 'log_center_Rho' in hist.cols:
        center_Rho = 10**hist.get('log_center_Rho')
    else:
        raise ValueError('No center_Rho-like columns found,')
    bad_vals = (hist.get('X_integral_part_b') > 0) & (center_Rho > 1e4) & mask_sgbrgb
    if np.any(bad_vals):
        i0, i1 = np.where(bad_vals)[0][[0, -1]]
        if hasattr(remove_extra, '__len__'):
            if len(remove_extra) == 2:
                i0 -= remove_extra[0]
                i1 += remove_extra[1]
            else:
                raise ValueError('remove_extra must be integer or of length-2.')
        else:
            i0 -= remove_extra
            i1 += remove_extra

        mask = np.zeros_like(mask_sgbrgb, dtype=bool)
        mask[i0:i1+1] = True
    else:
        mask = bad_vals
    f_CZ = calc_f_CZ_from_hist(hist)
    mask_f = f_CZ < f
    mask_1mf = f_CZ > 1 - f
    mask_inter = (f_CZ >= f) & (f_CZ <= 1 - f)

    kind = np.zeros_like(hist.get('star_age'), dtype=int)
    kind[mask_f] = 1  # strong
    kind[mask_1mf] = 2  # weak
    kind[mask_inter] = 3  # inter/two evn
    # Crossing spike region can cause bad fCZ
    r2_jump = np.diff((hist.data.r_2[mask] - hist.data.r_botCZ[mask]) / hist.data.r_1[mask], prepend=np.nan)
    bad_r2_mask = r2_jump < -0.1
    if np.any(bad_r2_mask):
        i_bad = np.where(bad_r2_mask)[0][0]
        f_CZ[i0 + i_bad] = (f_CZ[i0 + i_bad - 1] + f_CZ[i0 + i_bad + 1]) / 2
        i0 = i0 + i_bad - min(i_bad, 40)  # use as much of q_strong
    
    if np.any(bad_vals):
        mask = np.zeros_like(mask_sgbrgb, dtype=bool)
        mask[i0:i1+1] = True
        x = hist.get('star_age')
        x0 = x[i0]
        x1 = x[i1]
        
        #strong
        y0 = hist.get('coupling_strong')[i0]
        y1 = hist.get('coupling_strong')[i1]
        ip_func = ip.interp1d([x0, x1], [y0, y1])
        new_q = np.copy(hist.get('coupling_strong'))
        new_q[mask] = ip_func(x[mask])
        
        #weak
        new_qw = calc_qw(hist.get('X_integral_part'))
        if do_sum:
            new_qw[mask] = calc_qw(np.nansum([hist.get('X_integral_part')[mask], hist.get('X_integral_part_b')[mask]], axis=0))
        else:
            interp_window += 1
            dydx0, y0 = np.polyfit(x[i0-interp_window:i0], hist.get('X_integral_part')[i0-interp_window:i0]**0.5, 1)
            dydx1, y1 = np.polyfit(x[i1:i1+interp_window], hist.get('X_integral_part')[i1:i1+interp_window]**0.5, 1)
            f1 = x[mask]*dydx0 + y0
            f2 = x[mask]*dydx1 + y1
            new_qw = calc_qw(hist.get('X_integral_part'))
            new_qw[mask] = calc_qw(np.maximum(f1, f2)**2)
        
        kind[mask] = 3
    else:
        new_q = np.copy(hist.get('coupling_strong'))
        new_qw = calc_qw(hist.get('X_integral_part'))

    true_q = np.nan * np.zeros_like(new_q)
    true_q[mask_f] = new_q[mask_f]
    true_q[mask_1mf] = new_qw[mask_1mf]

    ip_func = ip.interp1d(hist.get('star_age')[~(kind == 3)], true_q[~(kind == 3)], bounds_error=False)
    true_q[mask_sgbrgb] = ip_func(hist.get('star_age')[mask_sgbrgb])

    mask_cheb_inter = mask_inter & (~mask_sgbrgb)
    alfa = (f_CZ[mask_cheb_inter] - f) / (1 - 2*f)
    beta = 1 - alfa
    true_q[mask_cheb_inter] = alfa * new_qw[mask_cheb_inter] + beta * new_q[mask_cheb_inter]
    
    if not do_blend:
        new_q[kind == 3] = np.nan
        if not interp_rgb:
            new_qw[kind == 3] = np.nan
        true_q[kind == 3] = np.nan
    
    return new_q, new_qw, mask, true_q, f_CZ, kind
    

def calc_FeH(hist, ZX_sol=0.0178, use_mask=None):
    mask = get_mask(hist, use_mask)
    surf_X = hist.data.surface_h1[mask] + hist.data.surface_h2[mask]
    surf_Y = hist.data.surface_he3[mask] + hist.data.surface_he4[mask]
    surf_Z = 1 - surf_X - surf_Y
    FeH = np.log10((surf_Z / surf_X) / ZX_sol)
    return FeH


def calc_deltanu(gs, hist, prefix='profile', suffix='.data.GYRE.sgyre_l', freq_units='uHz'):
    pnum = int(gs.path.split(prefix)[-1].replace(suffix, ''))
    hist_i = hist.get_model_num(pnum)[0][2]
    nu_all = get_freq(gs, freq_units)

    mask = gs.data.l == 0
    mask = np.logical_and(mask, gs.data.n_pg > 0)
    nu_max = hist.get('nu_max')[hist_i]
    fsig = (0.66 * nu_max ** 0.88) / 2 / np.sqrt(2 * np.log(2.))
    w = np.exp(-((nu_all[mask][:-1] - nu_max) / fsig) ** 2)
    delta_nus = np.diff(nu_all[mask])
    delta_nu = np.sum(w * delta_nus) / np.sum(w)
    return delta_nu


def calc_deltaPg(gs, hist, l, prefix='profile', suffix='.data.GYRE.sgyre_l'):
    pnum = int(gs.path.split(prefix)[-1].replace(suffix, ''))
    hist_i = hist.get_model_num(pnum)[0][2]
    nu_max = hist.get('nu_max')[hist_i]
    
    if l == 0:
        raise ValueError('Cannot use l=0 for period spacing.')
    
    mask = gs.data.l == l
    nu = get_freq(gs, 'Hz')[mask]
    dPi = -np.diff(nu ** -1)

    fsig = (0.66 * nu_max ** 0.88) / 2 / np.sqrt(2 * np.log(2.))
    w = np.exp(-((nu[:-1] - nu_max) / fsig) ** 2)
    dPi = np.sum(dPi*w/sum(w))
    return dPi


def correct_seismo(hist, gsspnum, mask, xname='center_he4', do_deltanu=True, do_deltaP=True,
                   prefix='profile', suffix='.data.GYRE.sgyre_l', get_poly=False, weight=True):
    mask = get_mask(hist, use_mask=mask)

    mnum_min, mnum_max = hist.data.model_number[mask][[0, -1]]

    x = []
    y_nu = []
    y_P = []
    deltanus = []
    deltaPs = []
    for gs, pnum in gsspnum:
        out = hist.get_model_num(pnum)
        if len(out) == 0:  # pnum not in history
            continue
        _, mnum, hist_i = out[0]
        if (mnum <= mnum_min) or (mnum >= mnum_max):
            continue
        if do_deltanu:
            deltanu = calc_deltanu(gs, hist, prefix=prefix, suffix=suffix)
            deltanus.append(deltanu)
            h_deltanu = hist.data.delta_nu[hist_i]
            y_nu.append(deltanu / h_deltanu)
        
        if do_deltaP:
            deltaP = calc_deltaPg(gs, hist, 1)
            deltaPs.append(deltaP)
            h_deltaP = hist.data.delta_Pg[hist_i]
            y_P.append(deltaP / h_deltaP)
        
        x.append(hist.get(xname)[hist_i])
    x = np.array(x)
    y_nu = np.array(y_nu)
    y_P = np.array(y_P)

    i_sort = np.argsort(x)
    x = x[i_sort]
    if weight:
        mid = (x[1:] + x[:-1])/2
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
            P_poly = np.array([np.nan]*3)
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
    return deltanu/(numax**2 * 1e-6*deltap)


def strided_app(a, L, S ):  # Window len = L, Stride len/stepsize = S
    # https://stackoverflow.com/questions/40084931/taking-subarrays-from-numpy-array-with-given-stride-stepsize/40085052#40085052
    nrows = ((a.size-L)//S)+1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a, shape=(nrows,L), strides=(S*n,n))


def weighted_percentile(data, weights, percentiles):
    if len(data) == 0:
        return np.nan
    if weights is None:
        weights = np.ones_like(data)
    sort_idx = np.argsort(data)
    data = data[sort_idx]
    weights = weights[sort_idx]
    cdf = (np.cumsum(weights) - 0.5 * weights) / np.sum(weights)
    return np.interp(percentiles, cdf, data)


def bootstrap_rolling_percentiles(xdat, ydat, window, percentiles, n=1000, size=None, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    
    x_sample_stats = []
    y_sample_stats = []
    
    if size is None:
        size = len(xdat)
    num_pts = len(xdat)
    for i in range(n):
        choice = rng.choice(num_pts, size=num_pts, replace=True)
        x = xdat[choice]
        y = ydat[choice]
        x_sort_idx = np.argsort(x)
        x_sample_stats.append(np.median(strided_app(x[x_sort_idx], window, 1), axis=-1))
        y_sample_stats.append(np.percentile(strided_app(y[x_sort_idx], window, 1), percentiles, axis=-1))
    
    x_sample_stats = np.array(x_sample_stats)
    y_sample_stats = np.array(y_sample_stats)
    
    x_med = np.mean(x_sample_stats, axis=0)
    y_med = np.mean(y_sample_stats, axis=0)
    y_std = np.std(y_sample_stats, axis=0)
    return x_med, y_med, y_std


def bootstrap_binned(xdat, ydat, percentiles, bins=20, min_frac_cts=0.02, n=1000, size=None, rng=None, max_loops=64):

    if isinstance(bins, int):
        nbins = bins

        idx = np.argsort(xdat)
        xsort = xdat[idx]
        usort = np.arange(len(xdat))[idx]

        cts, bins = np.histogram(xsort, bins=nbins)
        ndat = len(xdat)
        min_ct = int(np.ceil(ndat * min_frac_cts))

        pivot = np.where(np.cumsum(cts) > ndat/2)[0][0]
        for i in np.arange(pivot):
            ct = cts[i]
            if ct < min_ct:
                bins[i + 1:pivot+1] = np.linspace(0.5 * (xsort[sum(cts[:i]) + min_ct - 1] + xsort[sum(cts[:i]) + min_ct + 1]), bins[pivot], pivot-i)
                cts, _ = np.histogram(xsort, bins=bins)

        bins = -bins[::-1]
        xsort = -xsort[::-1]
        cts = cts[::-1]
        rev_pivot = nbins - pivot
        for i in np.arange(rev_pivot):
            ct = cts[i]
            if ct < min_ct:
                bins[i + 1:rev_pivot+1] = np.linspace(0.5 * (xsort[sum(cts[:i]) + min_ct - 1] + xsort[sum(cts[:i]) + min_ct + 1]), bins[rev_pivot], rev_pivot-i)
                cts, _ = np.histogram(xsort, bins=bins)

        bins = -bins[::-1]
        xsort = -xsort[::-1]
        cts = cts[::-1]

        nonmon = np.where(np.diff(bins) < 0)[0]
        if len(nonmon) == 2:
            bins[nonmon[0]:nonmon[1] + 2] = np.linspace(bins[nonmon[0]], bins[nonmon[1] + 1], nonmon[1] - nonmon[0] + 2)
        elif len(nonmon) == 1:
            rebin = np.where(cts > min_ct)[0]
            i0 = min(rebin)
            i1 = max(rebin)+1
            bins[i0:i1+1] = np.linspace(bins[i0], bins[i1], i1 - i0 + 1)
        cts, bins = np.histogram(xsort, bins=bins)

    else:
        nbins = len(bins) - 1
    x_sample_stats = []
    y_sample_stats = []
    
    percentiles = np.array(percentiles, dtype=float)
    if min(percentiles) > 1:
        percentiles /= 100
    stat_funcs = [functools.partial(weighted_percentile, weights=None, percentiles=percentile) for percentile in percentiles]
    
    if size is None:
        size = len(xdat)

    def work_func(args):
        n, rng, size, bins, nbins, stat_funcs, xdat, ydat = args
        if rng is None:
            rng = np.random.default_rng()

        x = np.zeros(size)
        y = np.zeros(size)
        y_sample_stats = np.zeros((n, nbins, len(stat_funcs)))
        y_sample_counts = np.zeros((n, nbins))
        for i in range(n):
            choice = rng.choice(len(xdat), size=size, replace=True)
            x[:] = xdat[choice]
            y[:] = ydat[choice]
            ix = np.argsort(x)
            x = x[ix]
            y = y[ix]
            counts, edges, binnumber = stats.binned_statistic(x, x, 'count', bins=bins)
            y_sample_counts[i] = counts
            edge_ix = np.argwhere(np.diff(binnumber, prepend=1)).flatten()

            k_left = 0
            for k_right in edge_ix:
                k = binnumber[k_left] - 1
                y_sample_stats[i, k] = weighted_percentile(y[k_left:k_right], None, percentiles)
                k_left = k_right
            y_sample_stats[i, k + 1] = weighted_percentile(y[k_left:], None, percentiles)
            y_sample_stats[i] *= (counts / counts)[:, None]
        return y_sample_stats

    if rng is not None:
        do_mp = False
    else:
        do_mp = True

    if do_mp:
        nproc = mp.cpu_count()
        if n % nproc:
            n_mp = n // nproc + 1
        else:
            n_mp = n // nproc
        args = n_mp, rng, size, bins, nbins, stat_funcs, xdat, ydat
        payload = dill.dumps((work_func, args))

        with mp.Pool(nproc) as pool:
            out_y_sample_stats = pool.map(run_dill_encoded, [payload]*nproc)
        y_sample_stats = np.concatenate(out_y_sample_stats)
    else:
        args = n, rng, size, bins, nbins, stat_funcs, xdat, ydat
        y_sample_stats = work_func(args)

    y_meds = np.nanmean(y_sample_stats, axis=0)
    y_stds = np.nanstd(y_sample_stats, axis=0)
    return bins, y_meds, y_stds

def run_dill_encoded(payload):
    fun, args = dill.loads(payload)
    return fun(args)


def two_dim_gaussian(x, y, A, mu_x, mu_y, sig_x, sig_y, rho):
    one_m_rho2 = 1 - rho ** 2
    return A / (2 * np.pi * sig_x * sig_y * np.sqrt(one_m_rho2)) * np.exp(-1 / (2 * one_m_rho2) * (
            ((x - mu_x) / sig_x) ** 2 -
            2 * rho * (x - mu_x) * (y - mu_y) / (sig_x * sig_y) +
            ((y - mu_y) / sig_y) ** 2))
def make_kde(xdat, ydat, e_xdat, e_ydat, xmin, xmax, xnum, ymin, ymax, ynum, w=1, rho=0):
    if np.isscalar(w):
        w = w * np.ones_like(xdat)
    if np.isscalar(rho):
        rho = rho*np.ones_like(xdat)

    x_grid = np.linspace(xmin, xmax, xnum)
    y_grid = np.linspace(ymin, ymax, ynum)
    X, Y = np.meshgrid(x_grid, y_grid)

    kde = np.zeros_like(X)

    for i in range(len(xdat)):
        kde += two_dim_gaussian(X, Y, w[i], xdat[i], ydat[i], e_xdat[i], e_ydat[i], rho[i])
    # te = time.time()
    # print(te - ts)
    dx = x_grid[1] - x_grid[0]
    dy = y_grid[1] - y_grid[0]
    integral = np.sum(kde * dx * dy)

    return kde, integral, X, Y
