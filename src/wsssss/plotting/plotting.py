"""
This module contains various functions to create plots using MESA and/or GYRE data.

Examples:
    Create an HRD from a history file from the MS onwards using a simple mask to skip the PMS.

    >>> import wsssss.load_data as ld
    >>> import wsssss.plotting.plotting as pl
    >>> hist = ld.History('path/to/LOGS/history.data')
    >>> pl.make_hrd(hist, use_mask=hist.data.center_h1<hist.data.center_h1[0] * 0.99)

"""

import copy
import os

import matplotlib as mpl
import numpy as np
from matplotlib import animation
from matplotlib import colors
from matplotlib import pyplot as plt
from matplotlib import transforms
from matplotlib import cm
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from scipy.interpolate import interp1d

from . import utils as pu
from .. import functions as uf
from . import kipper

np.seterr(invalid='ignore')


def calc_vhrd(hist, norm):
    xdat, ydat = uf.get_logTeffL(hist)

    tdat = np.diff(hist.get('star_age'), prepend=0.0)
    # tdat = 10 ** hist.get('log_dt')
    dHRD = np.sqrt(np.diff(np.append(0, xdat)) ** 2 + np.diff(np.append(0, ydat)) ** 2)
    vHRD = dHRD / tdat / norm
    return vHRD


def plot_vhrd(f, ax, hist, vHRD_norm, use_mask, norm, add_cbar, add_label):
    ini_mass = hist.get('star_mass')[0]
    xdat, ydat = uf.get_logTeffL(hist)

    mask = uf.get_mask(hist, use_mask)

    xdat = xdat[mask]
    ydat = ydat[mask]
    vHRD = calc_vhrd(hist, vHRD_norm)[mask]

    cax = ax.scatter(xdat, ydat, c=np.log10(vHRD), lw=0, cmap='viridis', norm=norm)
    ax.plot(xdat, ydat, c='k', lw=1)
    cbar = None
    if add_cbar:
        cbar = f.colorbar(cax, ax=ax)

    if add_label:
        ax.text(xdat[0], ydat[0], '${:.2f}\\mathrm{{M}}_\\odot$'.format(ini_mass), ha='right', va='top', fontsize=10)

    return f, ax, cbar, cax


def make_vhrd(hist, add_cbar=True, vHRD_norm=1.0, use_mask=True, ax=None, bounds=None, add_label=False):
    if bounds is None:
        bounds = np.arange(-9, 0, 1)
    norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)

    f, ax = pu.get_figure(ax)

    if type(hist) in [list, tuple, dict]:
        for h in hist:
            if isinstance(hist, dict):
                h = hist[h]
            f, ax, cbar, cax = plot_vhrd(f, ax, h, vHRD_norm, use_mask, norm, add_cbar, add_label)
            add_cbar = False
    else:
        f, ax, cbar, cax = plot_vhrd(f, ax, hist, vHRD_norm, use_mask, norm, add_cbar, add_label)

    if add_cbar:
        cbax = f.axes[1]
        cbax.set_ylabel(r'$\log_{10} \; v_\mathrm{HRD}$')

    ax.set_xlabel(r'$\log_{10} \; (T_\mathrm{eff} / K)$')
    ax.set_ylabel(r'$\log_{10} \; (L / L_\odot)$')

    ax.invert_xaxis()

    pu.hrd_const_rad(ax, fontsize=10)

    return f, ax


def plot_hrd_data(f, ax, hist, zdata, znorm, use_mask, norm, add_cbar, cmap, sdat=None, small_cbar=False,
                  label_func=None, vmin=None, vmax=None, linecolor='k', linear=False, linestyle='-'):
    xdat, ydat = uf.get_logTeffL(hist)
    if linear:
        xdat = 10 ** xdat
        ydat = 10 ** ydat

    mask = uf.get_mask(hist, use_mask)
    if not np.any(mask):
        return f, ax, None, None

    if type(zdata) == str:
        zdat = hist.get(zdata) / znorm
    elif zdata is None:
        zdat = np.zeros_like(xdat)
    elif np.ndim(zdata) == 0:
        zdat = (zdata / znorm) * np.ones_like(xdat)
    else:
        zdat = zdata / znorm

    xdat = xdat[mask]
    ydat = ydat[mask]
    zdat = zdat[mask]
    if sdat is not None:
        sdat = sdat[mask][::-1]
    else:
        sdat = 36

    # Reverse data to plot early data on top.
    if zdata is not None:
        ax.scatter(xdat[::-1], ydat[::-1], c='k', s=7 ** 2)
        cax = ax.scatter(xdat[::-1], ydat[::-1], c=zdat[::-1], s=sdat, lw=0, cmap=cmap, norm=norm, vmin=vmin, vmax=vmax)
    else:
        cax = None
    ax.plot(xdat, ydat, c=linecolor, lw=1, linestyle=linestyle)
    cbar = None
    if add_cbar and zdata is not None:
        if not small_cbar:
            cbar = f.colorbar(cax, ax=ax)
        else:
            cbar = f.colorbar(cax, ax=ax, aspect=40, pad=0)

    if label_func is None:
        ini_mass = hist.get('star_mass')[0]
        label = '${:.1f}M_\\odot$'.format(ini_mass)
        i_label = 0
        ha = 'right'
        va = 'top'
        angle = 0
    else:
        label, i_label, ha, va, angle = label_func(hist, mask)

    ax.text(xdat[i_label], ydat[i_label], label, ha=ha, va=va, fontsize=10, rotation=angle)

    return f, ax, cbar, cax


def make_hrd(hist, zdata=None, zlabel='', add_cbar=True, znorm=1.0, use_mask=None, ax=None, norm=None, cmap='viridis',
             add_const_rad=True, label_func=None, vmin=None, vmax=None, linecolor='k', linear=False):
    f, ax = pu.get_figure(ax)
    linestyles = ['-', '--', '-.', ':']
    if type(hist) in [list, tuple]:
        pass
    else:
        hist = [hist]
    if (zdata is not None) and not callable(zdata):
        if len(hist) == 1:
            if len(zdata) == len(hist[0]):
                zdata = [zdata]
        if (len(zdata) != len(hist)) and (type(zdata) != str):
            raise ValueError("When passing multiple histories and zdata is not a string"
                             "the length of zdata must the the length of hist.")

    for i, h in enumerate(hist):
        if type(zdata) == str or zdata is None:
            f, ax, cbar, cax = plot_hrd_data(f, ax, h, zdata, znorm, use_mask, norm, add_cbar, cmap, vmin=vmin,
                                             vmax=vmax, label_func=label_func, linecolor=linecolor, linear=linear,
                                             linestyle=linestyles[i % len(linestyles)])
        elif callable(zdata):
            f, ax, cbar, cax = plot_hrd_data(f, ax, h, zdata(h), znorm, use_mask, norm, add_cbar, cmap, vmin=vmin,
                                             vmax=vmax, label_func=label_func, linecolor=linecolor, linear=linear,
                                             linestyle=linestyles[i % len(linestyles)])
        else:
            f, ax, cbar, cax = plot_hrd_data(f, ax, h, zdata[i], znorm, use_mask, norm, add_cbar, cmap, vmin=vmin,
                                             vmax=vmax, label_func=label_func, linecolor=linecolor, linear=linear,
                                             linestyle=linestyles[i % len(linestyles)])
        if add_cbar and cbar is not None:
            cbax = f.axes[-1]
            cbax.set_ylabel('{}'.format(zlabel))
        add_cbar = False

    if linear:
        ax.set_xlabel(r'$T_\mathrm{eff} \; (\mathrm{K})$')
        ax.set_ylabel(r'$L \; (\mathrm{L}_\odot)$')
    else:
        ax.set_xlabel(r'$\log_{10} \; (T_\mathrm{eff} / \mathrm{K})$')
        ax.set_ylabel(r'$\log_{10} \; (L / \mathrm{L}_\odot)$')

    if np.diff(ax.get_xlim()) > 0:
        ax.invert_xaxis()
    ax.margins(0.05)

    if add_const_rad:
        pu.hrd_const_rad(ax, linear=linear)

    return f, ax


def make_propagation(p, hist, xname='logR', l=1, ax=None, only_NS=True, do_reduced=False, only_reduced=False,
                     add_legend=True, n_col_legend=3, fill_cavity=False, show_burn_level=0, do_numax=True):
    c = uf.get_constants(p)

    f, ax = pu.get_figure(ax)
    radius = uf.get_radius(p)

    csound = p.data.csound
    brunt_N2 = p.data.brunt_N2
    brunt_N2[brunt_N2 < 0] = 0

    lamb2 = uf.get_lamb2(p)

    if hist is not None:
        i_hist = p.get_hist_index(hist)

    if 'scale_height' in p.columns:
        H = p.data.scale_height * c.rsun
        cs2H2 = (csound / (4 * np.pi * H)) ** 2
        nu_c2 = cs2H2[1:] * (1 - 2 * np.diff(H) / np.diff(radius))
        x_skip = 1
    else:
        rho = 10 ** p.data.logRho
        dlnrho = -np.diff(np.log(rho))
        H = -(dlnrho / -np.diff(radius)) ** -1
        cs2H2 = (csound[1:] / (4 * np.pi * H)) ** 2
        nu_c2 = cs2H2[1:] * (1 - 2 * np.diff(H) / np.diff(radius)[1:])
        x_skip = 2

    x = pu.get_x_and_set_xlabel(p, xname, ax, hist=hist)

    if not only_reduced:  # Also plot normal N and S
        S = 1E6 / (2 * np.pi) * lamb2 ** 0.5
        N = 1E6 / (2 * np.pi) * brunt_N2 ** 0.5
        ax.plot(x, S, 'C0', label=r'$S_{\ell=1}$')
        ax.plot(x, N, 'C1', label='$N$')

    if do_reduced:  # Add N_red and S_red
        if 'q_J' in p.columns:
            J = p.get('q_J')
        else:
            J = 1 - 1 / 3 * np.gradient(np.log(p.data.mass), np.log(radius))

        S_red = 1E6 / (2 * np.pi) * lamb2 ** 0.5 * J
        N_red = 1E6 / (2 * np.pi) * brunt_N2 ** 0.5 / J

        ax.plot(x, S_red, 'C0--', label='$\\tilde{S}_{\\ell=1}$')
        ax.plot(x, N_red, 'C1--', label='$\\tilde{N}$')

    if not only_NS and not only_reduced:  # Add Acoustic cutouff
        ax.plot(x[x_skip:], 1E6 * nu_c2 ** 0.5, label=r'$\nu_c$')
        ax.plot(x, 1E6 * (csound / (4 * np.pi * H)), ls='--', label=r'$\nu_\mathrm{ac}$ approx')
        if hist is not None:
            ax.hlines(hist.data.acoustic_cutoff[i_hist] / (2 * np.pi), x[0], x[-1], label=r'$\nu_\mathrm{ac}$')

    if do_numax and (hist is not None):
        numax = hist.get('nu_max')[i_hist]
        xmin = np.nanmin(x)
        xmax = np.nanmax(x)

        if do_reduced:
            y1 = S_red
            y2 = N_red
        else:
            y1 = S
            y2 = N

        gmask = numax <= np.minimum(y1, y2)  # g-like
        pmask = numax >= np.maximum(y1, y2)  # p-like
        emask = ~(pmask | gmask)  # not p-like or g-like

        nu1, = ax.plot(x, np.where(~gmask, np.nan, numax * gmask), 'k--')
        nu2, = ax.plot(x, np.where(~emask, np.nan, numax * emask), 'k:')
        nu3, = ax.plot(x, np.where(~pmask, np.nan, numax * pmask), 'k-')

        nu_lines = (nu1, nu2, nu3)
        nu_lines_lc = mpl.collections.LineCollection([[[0, 0]] * 2] * 3, linestyles=['--', ':', '-'],
                                                     colors=['k', 'k', 'k'], label=r'$\nu_\mathrm{max}$')

    if xname != 's':
        ax.set_ylim(1e6 * np.min(lamb2) ** 0.5 / 2, 1e6 * np.max(lamb2) ** 0.5 * 2)
    ax.set_ylabel(r'Frequency ($\mu$Hz)')

    ax.set_yscale('log')
    if xname in ['logR']:
        ax.set_xscale('log')

    if fill_cavity:
        if do_reduced:
            ax.fill_between(x, -1e99 * np.ones_like(S_red), np.minimum(S_red, N_red), label='g-cavity', hatch='//',
                            zorder=-5,
                            facecolor='none', edgecolor='C1')
            ax.fill_between(x, np.maximum(S_red, N_red), 1e99, label='p-cavity', hatch='--', zorder=-5,
                            facecolor='none', edgecolor='C0')
        else:
            ax.fill_between(x, -1e99 * np.ones_like(S), np.minimum(S, N), label='g-cavity', hatch='//',
                            zorder=-5,
                            facecolor='none', edgecolor='C1')
            ax.fill_between(x, np.maximum(S, N), 1e99, label='p-cavity', hatch='--', zorder=-5,
                            facecolor='none', edgecolor='C0')

    if show_burn_level > 0:
        burning = p.data.net_nuclear_energy >= show_burn_level

        h_mask = (np.gradient(p.data.h1, p.data.mass) > 0) & (p.data.h1 > 1e-6) & burning
        he_mask = (np.gradient(p.data.he4, p.data.mass) > 0) & (p.data.he4 > 1e-6) & (p.data.h1 < 1e-6) & burning

        if np.any(h_mask):
            bmin_h = min(x[h_mask])
            bmax_h = max(x[h_mask])
            ax.axvspan(bmin_h, bmax_h, color='tab:cyan', zorder=-4, label='H-burn', alpha=0.5)

        if np.any(he_mask):
            bmin_he = min(x[he_mask])
            bmax_he = max(x[he_mask])
            ax.axvspan(bmin_he, bmax_he, color='tab:olive', zorder=-4, label='He-burn', alpha=0.5)

    handles, labels = ax.get_legend_handles_labels()
    handler_map = None
    if do_numax and (hist is not None):
        labels.append(nu_lines_lc.get_label())
        handles.append(nu_lines_lc)
        handler_map = {LineCollection: pu.HandlerDashedLines()}

    if add_legend:
        n_item = len(labels)
        n_row = int(np.ceil(n_item / n_col_legend))

        order = list(range(n_item))
        if 'p-cavity' in labels and 'He-burn' in labels:
            order[labels.index('He-burn')] = labels.index('p-cavity')
            order[labels.index('p-cavity')] = labels.index('He-burn')
        elif 'p-cavity' in labels:
            order[labels.index('p-cavity')] = labels.index('g-cavity')
            order[labels.index('g-cavity')] = labels.index('p-cavity')

        labels = [f'{labels[i]}' for i in range(n_item)]
        ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order], ncol=n_col_legend,
                  bbox_to_anchor=(0., 1.02, 1., .102), loc=3, mode='expand', borderaxespad=0.,
                  handler_map=handler_map)
    return f, ax


def make_propagation2(p, hist, xname='logR', l=1, ax=None, only_NS=True, do_reduced=True, only_reduced=False,
                      add_legend=True, fill_cavity=False, show_burn_level=0):
    c = uf.get_constants(p)

    f, ax = pu.get_figure(ax)

    hist_i = hist.get_profile_index(p.profile_num)[0]

    radius = uf.get_radius(p)

    csound = p.data.csound
    brunt_N2 = p.data.brunt_N2
    # brunt_N2[brunt_N2 < 0] = 0

    lamb2 = uf.get_lamb2(p)

    if 'scale_height' in p.columns:
        H = p.data.scale_height * c.rsun
        cs2H2 = (csound / (4 * np.pi * H)) ** 2
        nu_c2 = cs2H2[1:] * (1 - 2 * np.diff(H) / np.diff(radius))
        x_skip = 1
    else:
        rho = 10 ** p.data.logRho
        dlnrho = -np.diff(np.log(rho))
        H = -(dlnrho / -np.diff(radius)) ** -1
        cs2H2 = (csound[1:] / (4 * np.pi * H)) ** 2
        nu_c2 = cs2H2[1:] * (1 - 2 * np.diff(H) / np.diff(radius)[1:])
        x_skip = 2

    if xname == 's':
        if hist is None:
            raise ValueError("Need History for xname='s'.")
        else:
            i_hist = p.get_hist_index(hist)
            r1 = hist.data.r_1[i_hist]
            r2 = hist.data.r_2[i_hist]
        r0 = np.sqrt(r1 * r2)
        x = np.log(radius / r0)
        s0 = np.log(r1 / r2) / 2
        ax.set_xlabel('$s$')
        ax.set_xlim(-1.5 * abs(s0), 1.5 * abs(s0))
        smin, smax = ax.get_xlim()
        x[(x < smin) | (x > smax)] = np.nan
    else:
        x = pu.get_x_and_set_xlabel(p, xname, ax, hist=hist)

    if not only_reduced:  # Also plot normal N and S
        S2 = (1E6 / (2 * np.pi)) ** 2 * lamb2
        N2 = (1E6 / (2 * np.pi)) ** 2 * brunt_N2
        ax.plot(x, S2, label=r'$S^2_{\ell=1}$')
        ax.plot(x, N2, label='$N^2$')

    ax.axhline(hist.get('nu_max')[hist_i] ** 2, c='k', zorder=-1, label=r'$\nu^2_\mathrm{max}$')

    if do_reduced:  # Add N_red and S_red
        J = 1 - 1 / 3 * np.gradient(np.log(p.data.mass), np.log(radius))
        S2_red = (1E6 / (2 * np.pi) * J) ** 2 * lamb2
        N2_red = (1E6 / (2 * np.pi) / J) ** 2 * brunt_N2

        ax.plot(x, S2_red, label='$\\tilde{S}^2_{\\ell=1}$')
        ax.plot(x, N2_red, label='$\\tilde{N}^2$')

    if not only_NS and not only_reduced:  # Add Acoustic cutouff
        ax.plot(x[x_skip:], 1e12 * nu_c2, label=r'$\nu^2_\mathrm{ac}$')
        ax.plot(x, 1e12 * (csound / (4 * np.pi * H)) ** 2, ls='--', label=r'$\nu^2_\mathrm{ac}$ approx')
        ax.hlines((hist.data.acoustic_cutoff[hist_i] / (2 * np.pi)) ** 2, x[0], x[-1], label=r'$\nu^2_\mathrm{ac}$')

    ax.set_ylim(1e12 * np.min(lamb2) / 2, 1e12 * np.max(lamb2) * 2)
    ax.set_ylabel(r'Frequency$^2$ ($\mu$Hz$^2$)')

    ax.set_yscale('log')
    if xname in ['logR']:
        ax.set_xscale('log')

    if fill_cavity:
        if do_reduced:
            ax.fill_between(x, -1e99 * np.ones_like(S2_red), np.minimum(S2_red, N2_red), label='g-cavity', hatch='//',
                            zorder=-5,
                            facecolor='none', edgecolor='C5')
            ax.fill_between(x, np.maximum(S2_red, N2_red), 1e99, label='p-cavity', hatch='--', zorder=-5,
                            facecolor='none', edgecolor='C8')

    if show_burn_level > 0:
        burning = p.data.net_nuclear_energy >= show_burn_level

        h_mask = (np.gradient(p.data.h1, p.data.mass) > 0) & (p.data.h1 > 1e-6) & burning
        he_mask = (np.gradient(p.data.he4, p.data.mass) > 0) & (p.data.he4 > 1e-6) & (p.data.h1 < 1e-6) & burning

        if np.any(h_mask):
            bmin_h = min(x[h_mask])
            bmax_h = max(x[h_mask])
            ax.axvspan(bmin_h, bmax_h, color='tab:cyan', zorder=-4, label='H-burn', alpha=0.5)

        if np.any(he_mask):
            bmin_he = min(x[he_mask])
            bmax_he = max(x[he_mask])
            ax.axvspan(bmin_he, bmax_he, color='tab:olive', zorder=-4, label='He-burn', alpha=0.5)

    if add_legend:
        handles, labels = ax.get_legend_handles_labels()
        n_item = len(labels)
        n_row = int(np.ceil(n_item / 3))
        n_col = 3

        order = list(range(n_item))
        if 'p-cavity' in labels and 'He-burn' in labels:
            order[labels.index('He-burn')] = labels.index('p-cavity')
            order[labels.index('p-cavity')] = labels.index('He-burn')
        elif 'p-cavity' in labels:
            order[labels.index('p-cavity')] = labels.index('g-cavity')
            order[labels.index('g-cavity')] = labels.index('p-cavity')

        labels = [f'{labels[i]}' for i in range(n_item)]
        ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order], ncol=n_col,
                  bbox_to_anchor=(0., 1.02, 1., .102), loc=3, mode='expand', borderaxespad=0.)
    return f, ax


def make_echelle(gs, hist, ax=None, l_list=(0, 1, 2), offset='auto', delta_nu='median',
                 prefix='profile', suffix='.data.GYRE.sgyre_l', freq_units='uHz', legend_loc='upper right'):
    f, ax = pu.get_figure(ax)

    pnum = int(gs.path.split(prefix)[-1].replace(suffix, ''))
    hist_i = hist.get_profile_index(pnum)[0]
    nu_all = gs.get_frequencies(freq_units)

    if type(delta_nu) == str:
        delta_nu = delta_nu.lower()

        if delta_nu == 'mesa':
            delta_nu = hist.data.delta_nu[hist_i]

        elif delta_nu in ['mean', 'median', 'weighted', 'envelope', 'gaussian']:
            mask = gs.data.l == 0
            mask = np.logical_and(mask, gs.data.n_pg > 0)

            delta_nus = np.diff(nu_all[mask])
            if delta_nu == 'mean':
                delta_nu = np.mean(delta_nus)
            elif delta_nu == 'median':
                delta_nu = np.median(delta_nus)
            elif delta_nu in ['weighted', 'envelope', 'gaussian']:
                delta_nu = uf.calc_deltanu(gs, hist)

    mask = gs.data.l == 0
    E_l0 = gs.get('E_norm')[mask]
    nu_l0 = nu_all[mask]
    # interpolate over log10 inertia for better behaviour
    try:
        log_f_El0 = interp1d(nu_l0, np.log10(E_l0), kind='cubic', bounds_error=True)
    except ValueError:
        log_f_El0 = interp1d(nu_l0, np.log10(E_l0), kind='linear', bounds_error=True)
    if offset == 'auto_old':
        # Find an offset which results in the minimum amount of wrapping in the echelle diagram.
        def get_length(x):
            return np.sum(np.abs(np.diff(x)))

        lengths = []
        for offset in np.linspace(0, delta_nu, int(delta_nu * 10) + 1):
            length = 0

            for l in l_list:
                mask = gs.data.l == l

                nu = nu_all[mask]
                mod_nu = (nu + offset) % delta_nu

                length += get_length(mod_nu)

            lengths.append([offset, length])
        lengths = np.array(lengths)
        offset = np.where((lengths[:, 1] - lengths[:, 1].min()) < delta_nu * 1e-3)[0][0]

    elif offset == 'auto':
        nu_mod_dnu = nu_all[gs.data.l == 0] % delta_nu
        if (np.max(nu_mod_dnu) - np.min(nu_mod_dnu)) > 0.5 * delta_nu:  # has a fold
            nu_mod_dnu[nu_mod_dnu <= 0.5 * delta_nu] += delta_nu

        offset = 0.2 * delta_nu - np.min(nu_mod_dnu)
        offset = offset % delta_nu

    for i, l in enumerate(l_list):
        mask = gs.data.l == l

        nu = nu_all[mask]
        ms = pu.calc_inertia_marker_size(gs, l=l, freq_units=freq_units)
        ax.scatter((nu + offset) % delta_nu, nu, ms, c=f'C{i}', label=r'$\ell={}$'.format(l),
                   marker=['o', 's', '^', 'v'][i % 4])

    freq_unit = {'uHz': r'\mu', 'mHz': 'm', 'Hz': ''}[freq_units]
    ax.set_xlabel(r'Frequency + {:.2f} mod {:.2f} $\mu$Hz'.format(offset, delta_nu))
    ax.set_ylabel(f'Frequency (${freq_unit}$Hz)')
    if legend_loc != '':
        ax.legend(loc=legend_loc, ncol=len(l_list))

    ax.set_xlim(0, delta_nu)

    nu_max = hist.get('nu_max')[hist_i]
    fmid = nu_max
    fsig = (0.66 * nu_max ** 0.88) / 2 / np.sqrt(2 * np.log(2.))  # Mosser 2012a

    fmin = max(1e-4, fmid - 2 * fsig - 0.5 * delta_nu)
    fmax = fmid + 2 * fsig + 0.5 * delta_nu

    ax.set_ylim(fmin, fmax)

    # ax.text(0, 0, rf'$\Delta\nu = {delta_nu:.2f}')

    return f, ax


#
# def make_period_echelle(gs, hist, ax=None,
#                  prefix='profile', suffix='.data.GYRE.sgyre_l', freq_units='uHz', legend_loc='upper right'):
#     f, ax = pu.get_figure(ax)
#
#     pnum = int(gs.path.split(prefix)[-1].replace(suffix, ''))
#     hist_i = hist.get_profile_index(pnum)[0]
#     nu_all = gs.get_frequencies(freq_units)
#     dPi1 = hist.get('delta_Pg')[hist_i]
#
#     ms = pu.calc_inertia_marker_size(gs, l=1, freq_units=freq_units)
#
#     nu = uf.get_freq(gs, 'Hz')
#
#     # mask = None
#     # ax.scatter(1/nu[mask] % dPi1, nu, ms, color='C1', label=fr'$\ell=1$')
#
#     freq_unit = {'uHz': r'\mu', 'mHz': 'm', 'Hz': ''}[freq_units]
#     ax.set_xlabel(r'$1/\nu \; \mathrm{mod} \; \Delta \Pi_1$')
#     ax.set_ylabel(f'Frequency (${freq_unit}$Hz)')
#
#     if legend_loc != '':
#         ax.legend(loc=legend_loc, ncol=len(l_list))
#     ax.scatter((1 / nu % dPi1) - dPi1, nu, ms, color='C1')
#     ax.axvline(0, 0, 1, ls='--', color='k', zorder=-3)
#     return f, ax


def make_inertia(gs, ax=None, l_list=(0, 1, 2), freq_units='uHz', div=True, legend_loc='upper right',
                 scale_marker=False):
    f, ax = pu.get_figure(ax)
    freqs = gs.get_frequencies(freq_units)

    mask = gs.data.l == 0
    E_l0 = gs.get('E_norm')[mask]
    # interpolate over log10 inertia for better behaviour
    try:
        log_f_El0 = interp1d(freqs[mask], np.log10(E_l0), kind='cubic', bounds_error=True)
    except ValueError:
        log_f_El0 = interp1d(freqs[mask], np.log10(E_l0), kind='linear', bounds_error=True)

    for i, l in enumerate(l_list):
        mask = gs.get('l') == l
        E = gs.get('E_norm')[mask]
        if scale_marker:
            ms = pu.calc_inertia_marker_size(gs, l, freq_units)
        if div:
            E_div_El0 = E / 10 ** log_f_El0(freqs[mask])
            if scale_marker:
                ax.scatter(freqs[mask], E_div_El0, s=ms, color=f'C{l}', label=fr'$\ell={l}$',
                           marker=['o', 's', '^', 'v'][i % 4])
                ax.plot(freqs[mask], E_div_El0, color=f'C{l}', lw=1)
            else:
                ax.plot(freqs[mask], E_div_El0, label=fr'$\ell={l}$', lw=1, color=f'C{l}',
                        marker=['o', 's', '^', 'v'][i % 4])
        else:
            if l == 0:
                x = np.linspace(min(freqs[mask]), max(freqs[mask]), 1000)
                ax.plot(x, 10 ** log_f_El0(x), 'C0-', lw=1)
                if scale_marker:
                    ax.scatter(freqs[mask], E, s=ms, color=f'C{l}', label=fr'$\ell={l}$',
                               marker=['o', 's', '^', 'v'][i % 4])
                else:
                    ax.plot(freqs[mask], E, 'C0.', lw=1)
                    ax.plot([], [], 'C0', lw=1, label=fr'$\ell={l}$', marker=['o', 's', '^', 'v'][i % 4])
            else:
                if scale_marker:
                    ax.scatter(freqs[mask], E, s=ms, color=f'C{l}', label=fr'$\ell={l}$',
                               marker=['o', 's', '^', 'v'][i % 4])
                    ax.plot(freqs[mask], E, color=f'C{l}', lw=1)
                else:
                    ax.plot(freqs[mask], E, label=fr'$\ell={l}$', lw=1, color=f'C{l}',
                            marker=['o', 's', '^', 'v'][i % 4])

    if legend_loc != '':
        ax.legend(loc=legend_loc, ncol=len(l_list))

    freq_unit = {'uHz': r'\mu', 'mHz': 'm', 'Hz': ''}[freq_units]
    ax.set_xlabel(f'Mode frequency  (${freq_unit}$Hz)')
    if div:
        ax.set_ylabel(f'Normalized mode inertia $E_{{\\ell}} / E_{{\\ell=0}}$')
    else:
        ax.set_ylabel(f'Mode inertia $E_{{\\ell}}$')
    ax.set_yscale('log')

    return f, ax


def make_resolutions(profs, ax=None, yres=100, xname='model_number', yname='mass', zorder=0):
    f, ax = pu.get_figure(ax)

    n_zones = 0
    for p in profs:
        n_zones += p.header['num_zones']
    dat = np.zeros((n_zones, 2))
    i = 0

    x_edges = [0]
    for p in profs:
        nz = p.header['num_zones']
        dat[i:i + nz, 0] = p.header[xname]
        dat[i:i + nz, 1] = p.get(yname)
        i += nz

        x_edges.append(p.header[xname])
    x_edges = np.array(x_edges)
    x_edges[0] = x_edges[1] - (x_edges[2] - x_edges[1])

    counts, edges_x, edges_y, cax = ax.hist2d(dat[:, 0], dat[:, 1],
                                              bins=((x_edges),
                                                    np.linspace(dat[:, 1].min(), dat[:, 1].max(), yres + 1)),
                                              norm=colors.LogNorm(vmin=1), cmin=1, zorder=zorder)
    f.colorbar(cax, ax=ax)

    return f, ax, (counts, edges_x, edges_y)


def make_hrd_profiles(hist, add_cbar=True, vHRD_norm=1.0, use_mask=True, ax=None, show_prof_num=-1):
    f, ax = make_vhrd(hist, add_cbar, vHRD_norm, use_mask, ax)

    mask = uf.get_mask(hist, use_mask)
    masked_modnum = hist.get('model_number')[mask]

    model_num, _, pnum = hist.index.T
    model_num = np.intersect1d(model_num, masked_modnum)
    hist_i = np.where(np.isin(hist.data.model_number, model_num))[0]

    xdat, ydat = uf.get_logTeffL(hist)

    ax.plot(xdat[hist_i], ydat[hist_i], 'k+')

    if show_prof_num > 0:
        for i in range(0, len(hist_i), show_prof_num):
            ax.text(xdat[hist_i[i]], ydat[hist_i[i]], str(pnum[i]), verticalalignment='top')

    return f, ax


def make_hrd_models(hist, add_cbar=True, vHRD_norm=1.0, use_mask=True, ax=None, show_mod_num=1000, bounds=None):
    f, ax = make_vhrd(hist, add_cbar, vHRD_norm, use_mask, ax, bounds)

    mask = uf.get_mask(hist, use_mask)
    masked_modnum = hist.get('model_number')[mask]

    hist_i = np.where(np.logical_and(hist.data.model_number % show_mod_num == 0,
                                     np.isin(hist.data.model_number, masked_modnum)))[0]

    xdat, ydat = uf.get_logTeffL(hist)
    ax.plot(xdat[hist_i], hist.data.log_L[hist_i], 'k+')

    if show_mod_num > 0:
        for i in range(0, len(hist_i)):
            ax.text(xdat[hist_i[i]], hist.data.log_L[hist_i[i]], str((i + 1) * show_mod_num),
                    verticalalignment='top')

    return f, ax


def make_age_nu(hist, gss, l=1, ax=None, gyre_summary_prefix='profile',
                gyre_summary_suffix='.data.GYRE.sgyre_l', age_unit='Myr', marker='k.'):
    f, ax = pu.get_figure(ax)

    pnums = []
    for gs in gss:
        fname = os.path.split(gs.path)[-1]
        if fname.startswith(gyre_summary_prefix) and fname.endswith(gyre_summary_suffix):
            pnum = int(fname[len(gyre_summary_prefix):-len(gyre_summary_suffix)])
        else:
            raise ValueError('Gyre summary file {} does not '
                             'start with `{}` or end with `{}`.'.format(gs.path, gyre_summary_prefix,
                                                                        gyre_summary_suffix))
        pnums.append(pnum)

    asort = np.argsort(pnums)
    gss = [gss[i] for i in asort]
    pnums = [pnums[i] for i in asort]

    age_factor = {'yr': 1e0, 'kyr': 1e3, 'Myr': 1e6, 'Gyr': 1e9}[age_unit]

    hist_i = hist.get_profile_index(pnums)
    ages = hist.get('star_age')[hist_i] / age_factor

    for i, gs in enumerate(gss):
        mask = gs.data.l == l
        mask = np.logical_and(mask, gs.data.n_pg > 0)
        nu = gs.get_frequencies('uHz')[mask]

        age = ages[i] * np.ones_like(nu)
        ax.plot(age, nu, marker)

    ax.set_xlabel('Age ${}$'.format(age_unit))
    ax.set_ylabel(r'Frequency ($\mu$Hz)')

    return f, ax


def make_mesa_gyre_delta_nu(hist, gss, l_list=(0, 1, 2), xaxis='model_number', gyre_summary_prefix='profile',
                            gyre_summary_suffix='.data.GYRE.sgyre_l', ax=None):
    if ax is None:
        f, (ax0, ax1) = plt.subplots(nrows=2, ncols=1, sharex=True, gridspec_kw={'height_ratios': [3, 1]})

    elif not hasattr(ax, '__len__'):
        f = pu.get_figure()
        bbox = ax.get_position()
        width, height = bbox.p1 - bbox.p0

        bbox2_p0 = bbox.p0.copy()
        bbox2_p1 = bbox.p1.copy()
        bbox2_p0[1] += height / 3
        bbox2 = transforms.Bbox([bbox2_p0, bbox2_p1])

        ax.set_position(bbox2)
        ax.xaxis.set_visible(False)
        ax0 = ax

        bbox3_p0 = bbox.p0.copy()
        bbox3_p1 = bbox.p1.copy()
        bbox3_p1[1] = bbox2_p0[1]
        bbox3 = transforms.Bbox([bbox3_p0, bbox3_p1])
        ax1 = f.add_axes(bbox3)

    else:
        ax0, ax1 = ax
        f = pu.get_figure()

    ax0.tick_params(labelbottom=False)

    gyre_delta_nus = []

    for gs in gss:
        pnum = int(gs.fname[len(gyre_summary_prefix):-len(gyre_summary_suffix)])
        delta_nus = []
        for l in l_list:
            mask = gs.data.l == l
            mask = np.logical_and(mask, gs.data.n_pg > 0)

            delta_nus.extend(np.diff(gs.get_frequencies('uHz')[mask]))

        median_delta_nu = np.mean(delta_nus)
        mean_delta_nu = np.median(delta_nus)

        gyre_delta_nus.append([pnum, median_delta_nu, mean_delta_nu])
    gyre_delta_nus = np.array(gyre_delta_nus)
    hist_i = hist.get_profile_index(gyre_delta_nus[:, 0])

    xdat = hist.get(xaxis)
    mesa_delta_nu = hist.get('delta_nu')

    ax0.plot(xdat[hist_i], gyre_delta_nus[:, 1], 'xC1', label='median')
    ax0.plot(xdat[hist_i], gyre_delta_nus[:, 2], '+C2', label='mean')
    ax0.plot(xdat, mesa_delta_nu, label='MESA')
    ax0.set_ylabel(r'$\Delta \nu$ ($\mu$Hz)')
    # ax0.set_ylim([0, ax0.get_ylim()[1]])
    ax0.set_yscale('log')

    ax0.legend()

    ax1.plot(xdat[hist_i], gyre_delta_nus[:, 1] / mesa_delta_nu[hist_i], 'xC1', label='median')
    ax1.plot(xdat[hist_i], gyre_delta_nus[:, 2] / mesa_delta_nu[hist_i], '+C2', label='mean')
    ax1.grid(axis='y')

    ax1.set_ylabel(r'$\Delta \nu_{\mathrm{gyre}} / \Delta\nu_{\mathrm{MESA}}$')
    ax1.set_xlabel(xaxis)

    # ax0.set_xlim([min(xdat), max(xdat)])
    # ax1.set_xlim([min(xdat), max(xdat)])

    return f, (ax0, ax1)


def make_composition(prof, elements='all', xname='mass', ax=None, add_burnmix=True, legend_loc='best', add_cbar=True,
                     add_legend=False, add_label_lines=True, num_line_label=4, normalize_comp=False, grad=False,
                     fontsize=7, hist=None):
    f, ax = pu.get_figure(ax)

    e_pp = ['h1', 'he4']
    e_cno = ['h1', 'he4', 'c12', 'n14', 'o16']
    e_3a = ['he4', 'c12', 'o16']
    e_c = ['he4', 'c12', 'o16', 'ne20', 'mg24']

    do_xyz = False
    if isinstance(elements, str):
        elements = elements.lower()
    if elements == 'all':
        elems = [_ for _ in prof.columns if len(_) <= 4 and not _.startswith('log') and _[-1].isdigit()]
    elif elements == 'pp':
        elems = e_pp
    elif elements == 'cno':
        elems = e_cno
    elif elements == '3a':
        elems = e_3a
    elif elements == 'c':
        elems = e_c
    elif elements == 'xyz':
        do_xyz = True
        elems = ['X', 'Y', 'Z']
        X = prof.get('h1')
        if 'h2' in prof.columns:
            X += prof.get('h2')
        if 'h3' in prof.columns:
            X += prof.get('h3')
        Y = prof.get('he4')
        if 'he3' in prof.columns:
            Y += prof.get('he3')
        Z = 1 - X - Y
    else:
        elems = elements

    xdata = pu.get_x_and_set_xlabel(prof, xname, ax=ax, hist=hist)

    _colors = ['C0', 'C1', 'C2', 'C3', 'C5', 'C7', 'C8', 'C9']
    linestyles = ['-', '--', '-.', ':']
    for i, e in enumerate(elems):
        _c = _colors[i % len(_colors)]
        ls = linestyles[i // len(_colors)]
        if do_xyz:
            if e == 'X':
                comp = X
            elif e == 'Y':
                comp = Y
            elif e == 'Z':
                comp = Z
        else:
            comp = prof.get(e)
        if normalize_comp:
            comp -= min(comp)
            comp /= max(comp)
        if grad:
            if 'pressure' in prof.columns:
                pressure = prof.get('pressure')
            elif 'logP' in prof.columns:
                pressure = 10**prof.get('logP')
            else:
                raise KeyError('No column name with name pressure or logP.')
            comp = uf.cell2face(comp, prof.data.mass, True)
            comp_grad = np.gradient(comp, pressure) / (comp / pressure)
            comp_grad[comp < 1e-9] = np.nan
            comp = np.abs(comp_grad)

        ax.plot(xdata, comp, c=_c, ls=ls, label=e.capitalize())

    if grad:
        ax.set_ylabel('abs dlnX/dlnP')
    else:
        ax.set_ylabel('Mass fraction')
        ax.set_yscale('log')
    if xname in ['logR']:
        ax.set_xscale('log')

    if xname == 'mass':
        ax.set_xlim(0, None)
    if not grad:
        if do_xyz:
            ax.set_ylim(10 ** -3.5, 2)
            ax.set_yticks([1e0, 1e-1, 1e-2, 1e-3])
        else:
            ax.set_ylim(10 ** -6.5, 2)
            ax.set_yticks([1e0, 1e-3, 1e-6])

    if add_burnmix == 'mix':
        pu.add_mixing(ax, prof, xname, add_legend=False)
    elif add_burnmix == 'burn':
        pu.add_burning(ax, prof, xname, add_cbar=add_cbar)
    else:
        if add_burnmix:
            pu.add_burning(ax, prof, xname, add_cbar=add_cbar)
            pu.add_mixing(ax, prof, xname, add_legend=False)

    if add_label_lines:
        pu.line_legend(ax, num_line_label=num_line_label, fontsize=fontsize)

    if add_legend:
        ax.legend(loc=legend_loc)
    return f, ax


def make_gradients(prof, xname='mass', hist=None, ax=None, add_legend=True, n_cols_legend=2):
    f, ax = pu.get_figure(ax)

    x = pu.get_x_and_set_xlabel(prof, xname, ax=ax, hist=hist)
    labels = {'gradr': r'$\nabla_\mathrm{rad}$', 'grada': r'$\nabla_\mathrm{ad}$',
              'actual_gradT': r'actual $\nabla$', 'gradT': r'$\nabla$'}

    for i, grad in enumerate(['gradr', 'grada', 'actual_gradT', 'gradT']):
        try:
            ax.plot(x, prof.get(grad), ['k-', 'k--', 'r', 'r'][i], label=labels[grad], zorder=-(i // 2) + 2,
                    lw=1.5 + 4 * (i // 2))
        except ValueError:
            pass

    if 'logP' in prof.columns:
        logP = prof.get('logP')
        skip_grad_mu = False
    elif 'pressure' in prof.columns:
        logP = np.log10(prof.get('pressure'))
        skip_grad_mu = False
    else:
        skip_grad_mu = True

    if 'mu' in prof.columns:
        pass
    else:
        skip_grad_mu = False

    if not skip_grad_mu:
        grad_mu = np.gradient(np.log10(prof.get('mu'), logP))
        ax.plot(x, grad_mu, 'k-.', label=r'$\nabla_\mu$')

    if add_legend:
        pu.top_legend(ax, n_cols_legend)
    ax.set_ylim(0, 1)
    return f, ax


def make_period_spacing(gs, hist, ax=None, freq_units='uHz', prefix='profile', suffix='.data.GYRE.sgyre_l',
                        l_list=(0, 1, 2), legend_loc='upper right'):
    f, ax = pu.get_figure(ax)
    pnum = int(gs.path.split(prefix)[-1].replace(suffix, ''))
    hist_i = hist.get_profile_index(pnum)[0]
    ax.axvline(hist.get('nu_max')[hist_i], color='k', zorder=-3)
    for i, l in enumerate(l_list):
        if l == 0:  # No radial g-modes
            continue
        mask = gs.data.l == l
        nu = gs.get_frequencies(freq_units)[mask]
        dPi = -np.diff((nu * {'uHz': 1e-6, 'mHz': 1e-3, 'Hz': 1}[freq_units]) ** -1)

        # ax.plot(dPi, nu[:-1] % hist.get('delta_Pg')[hist_i], f'C{l}.', label=fr'$\ell={l}$')

        ms = pu.calc_inertia_marker_size(gs, l=l, freq_units=freq_units)

        # ax.plot(nu[:-1], dPi, f'C{l}.', label=fr'$\ell={l}$')
        ax.scatter(nu[:-1], dPi, s=ms[:-1], color=f'C{l}', marker=['o', 's', '^', 'v'][i % 4])
        ax.plot(nu[:-1], dPi, color=f'C{l}', lw=1)
        line = plt.matplotlib.lines.Line2D([], [], linewidth=1, linestyle='-', marker=['o', 's', '^', 'v'][i % 4],
                                           color=f'C{l}', label=fr'$\ell={l}$')
        ax.add_line(line)

    freq_prefix = {'uHz': r'\mu', 'mHz': 'm', 'Hz': ''}[freq_units]
    ax.set_ylabel(r'$\Delta P$ (s)')
    ax.set_xlabel(f'Mode frequency  (${freq_prefix}$Hz)')

    if legend_loc != '':
        ax.legend(loc=legend_loc)
    # ax.set_xlim(0, None)

    return f, ax


def make_structural(prof, ax=None, xname='mass', hist=None, normalize=False):
    f, ax = pu.get_figure(ax)

    x = pu.get_x_and_set_xlabel(prof, xname, ax, hist=hist)

    for yname in ('logP', 'logRho', 'logT', 'logR'):
        if yname in prof.columns:
            ydata = prof.get(yname)
        else:
            other_yname = {'logP': 'pressure', 'logRho': 'density', 'logT': 'temperature', 'logR': 'radius'}[yname]
            ydata = np.log10(prof.get(other_yname))
        if normalize:
            ydata -= min(ydata)
            ydata /= max(ydata)
        ax.plot(x, ydata, label=yname)

    if xname == 'mass':
        ax.set_xlim(0, None)

    pu.line_legend(ax, edge_space=0.075)
    ax.set_xlabel(xname)

    return f, ax


# def make_eos(prof, ax=None, mesa_version='15140', cbar_min=0, cbar_max=None):
#     f, ax = pu.get_figure(ax)
#
#     eos_path = os.path.join(os.path.dirname(__file__), f'../misc/eos_plotter_{mesa_version}.dat')
#
#     # Based on plotter.py from $MESA_DIR/eos
#
#     def parse(fname):
#         nY, nX = np.loadtxt(fname, max_rows=1, skiprows=3, unpack=True, dtype=int)
#         data = np.loadtxt(fname, skiprows=4)
#         data = np.reshape(data, (nX, nY, -1))
#         Yran = data[0, :, 0]
#         Xran = data[:, 0, 1]
#         data = np.swapaxes(data, 0, 1)
#         return data, Yran, Xran
#
#     with open(eos_path, 'r') as handle:
#         title = handle.readline().strip()
#         xlabel = handle.readline().strip()
#         ylabel = handle.readline().strip()
#
#     eosDT, Yran, Xran = parse(eos_path)
#
#     # set up plot and labels
#     ax.set_title(title)
#     ax.set_xlabel(xlabel)
#     ax.set_ylabel(ylabel)
#     ax.set_xlim(Xran.min(), Xran.max())
#     ax.set_ylim(Yran.min(), Yran.max())
#
#     # set up color map
#     cmap = copy.copy(mpl.cm.get_cmap("viridis"))
#     cmap.set_over('white')
#     cmap.set_under('black')
#
#     pcol = ax.pcolormesh(Xran, Yran, eosDT[..., 2], shading='nearest', cmap=cmap, vmin=cbar_min, vmax=cbar_max)
#     pcol.set_edgecolor('face')
#     cax = f.colorbar(pcol, extend='both')
#     cax.set_label('')
#
#     try:
#         xprof = prof.get('logRho')
#     except ValueError:
#         xprof = np.log10(prof.get('density'))
#
#     try:
#         yprof = prof.get('logT')
#     except ValueError:
#         yprof = np.log10(prof.get('temperature'))
#     ax.plot(xprof, yprof, 'k')
#
#     return f, ax


def make_eigenfunc_compare(gs, gefs, prof, hist, l_list=(1,), prop_y_lims=(3e0, 5e4), eig_y_lims=None,
                           x_lims=(1e-3, 0.5), prop_kwargs={}, inertia_kwargs={}):
    f, axes = plt.subplot_mosaic([list('AAADD'), list('BBBDD'), list('CCC..')])
    axes['A'].sharex(axes['B'])
    axes['B'].sharex(axes['C'])
    i_hist = prof.get_hist_index(hist)
    if eig_y_lims is None:
        eig_y_lims = ((None, None), (None, None))

    if len(eig_y_lims) == 1:
        axes['B'].get_shared_y_axes().join(axes['B'], axes['C'])
    axes['D'].yaxis.set_label_position('right')
    axes['D'].yaxis.tick_right()

    prop_kwargs_use = {'xname': 'x', 'ax': axes['A'], 'add_legend': False}
    prop_kwargs_use.update(prop_kwargs)
    make_propagation(prof, hist, **prop_kwargs_use)

    inertia_kwargs_use = {'ax': axes['D'], 'l_list': l_list, 'div': False, 'legend_loc': ''}
    inertia_kwargs_use.update(inertia_kwargs)
    make_inertia(gs, **inertia_kwargs_use)
    axes['D'].axvline(hist.get('nu_max')[i_hist], color='k', zorder=-5)

    if 'density' in prof.columns:
        rho = prof.get('density')
    elif 'logRho' in prof.columns:
        rho = 10 ** prof.get('logRho')
    scaling_func = interp1d(prof.data.radius / prof.data.radius[0], prof.data.radius * rho ** 0.5, bounds_error=False)

    for i, gef in enumerate(gefs):
        scaling = scaling_func(gef.data.x)
        axes['A'].axhline(gef.header['Re(freq)'], color=f'C{i % 10}')
        axes['B'].plot(gef.data.x, scaling * gef.get('Re(xi_r)'), color=f'C{i % 10}')
        axes['C'].plot(gef.data.x, scaling * gef.get('Re(xi_h)'), color=f'C{i % 10}')

        l = gef.header['l']
        n_pg = gef.header['n_pg']
        n_p = gef.header['n_p']
        mask = (gs.data.l == l) & (gs.data.n_pg == n_pg) & (gs.data.n_p == n_p)
        intertia = gs.data.E_norm[mask]
        axes['D'].plot(gs.get('Re(freq)')[mask], intertia, f'kX', zorder=9)
        axes['D'].plot(gs.get('Re(freq)')[mask], intertia, f'C{i % 10}x', zorder=10)

    axes['A'].set_ylim(prop_y_lims)
    if len(eig_y_lims) == 1:
        axes['B'].set_ylim(eig_y_lims[0])
    else:
        axes['B'].set_ylim(eig_y_lims[0])
        axes['C'].set_ylim(eig_y_lims[1])
    axes['B'].set_xlim(x_lims)

    axes['C'].set_xlabel(axes['A'].get_xlabel())
    axes['A'].set_xlabel('')

    axes['B'].set_ylabel(r'$r \; \rho^{1/2} \; \xi_r$')
    axes['C'].set_ylabel(r'$r \; \rho^{1/2} \; \xi_h$')

    axes['A'].set_xscale('log')
    # axes['A'].xaxis.set_ticklabels([])
    axes['A'].xaxis.set_ticklabels([], minor=False)
    axes['B'].xaxis.set_ticklabels([], minor=False)
    axes['A'].xaxis.set_ticklabels([], minor=True)
    axes['B'].xaxis.set_ticklabels([], minor=True)

    return f, axes

def make_kipp(hist, profs=None, ax=None, xaxis='model_number', yaxis='mass', caxis='eps_net', zone_filename='zones_wsssss.dat',
                 verbose=False, save_zones=True, clobber_zones=False, prof_prefix='profile', prof_suffix='.data', mixing_min_height=0,
                 prof_resolution=200, xlims=None, ylims=None, clims=None, kwargs_mixing=None, norm=None, cmap=None,
                 kwargs_profile_color=None, return_Kipp_data=False, parallel=True, ignore_monotonic=False, legend_loc=None, add_cbar=False):
    """
    Create a Kippenhahn diagram.

    Args:
        hist (History): MESA History with which to create a Kippenhahn diagram.
        profs (list of Profile, optional): MESA Profile used to create a Kippenhahn diagram. Is optional if `caxis='eps_net'` and `burning_regions` in the history columns is used.
        ax (matplotlib axes, optional): Matplotlib figure axis. If set to `None` will create one.
        xaxis (str, optional): x-axis column name. Defaults to 'model_number'.
        yaxis (str, optional): y-axis column name. Must be one of `mass` or `radius`. Defaults to `mass`.
        caxis (str, optional): colour-axis column name. Must be 'eps_net' or a column in `Profile`.
        zone_filename (str, optional): Filename of zonefile.
        verbose (bool, optional): If ``True``, prints extra information.
        save_zones (bool, optional): If ``True``, will cache Kippenhahn regions in ``zone_filename``.
        clobber_zones (bool, optional): If ``True``, will generate Kippenhahn regions from scratch.
        prof_prefix (str, optional): Profile file name prefix.
        prof_suffix (str, optional): Profile file name suffix.
        prof_resolution (int, optional): y-axis resolution for Profiles.
        mixing_min_height (float, optional): Minimum fractional vertical extent of mixing region. If smaller, the region is not plotted.
        xlims (length 2 array, optional): Lower and upper limits of the x-axis.
        ylims (length 2 array, optional): Lower and upper limits of the y-axis.
        clims (length 2 array, optional): Lower and upper limits of the colour-axis. Cannot be used at the same time with norm.
        kwargs_mixing (dict, optional): kwargs used to draw mixing regions, if `None`, defaults to `plotting.utils.pu.get_default_mixing_kwargs()`.
        norm (optional): Matplotlib normalize class.
        cmap (optional): Matplotlib colormap.
        kwargs_profile_color (dict, optional): Passed to tripcolor in ``add_color``.
        return_Kipp_data (bool, optional): If ``True``, also returns Kipp_data.
        parallel (bool, optional): If ``True``, calculate Kippenhahn regions in parallel.
        legend_loc(str, optional): If set, create a legend along that side. Can be 'top' or 'right'.
        add_cbar(bool, optional): If ``True``, will create a colorbar for the color axis.

    Returns:
        tuple of matplotlib.figure.Figure, matplotlib.axes._axes.Axes, and optionally Kipp_data: Figure, axis, and optionally `Kipp_data` of the Kippenhahn diagram.

    """
    if clims is None and caxis == 'eps_net':
        clims = [-3, 8]

    if norm is not None:
        if 'norm' in kwargs_profile_color.keys():
            print(f'Using norm from argument.')
        kwargs_profile_color['norm'] = norm
    else:
        if kwargs_profile_color is None:
            norm = None
        else:
            norm = kwargs_profile_color['norm']

    kd = kipper.Kipp_data(hist, profs, xaxis, yaxis, caxis, norm, zone_filename, verbose, save_zones, clobber_zones, prof_prefix,
                   prof_suffix, prof_resolution, parallel, ignore_monotonic)
    f, ax = kd.make_kipp(ax, xlims, ylims, clims, norm, cmap, mixing_min_height, kwargs_mixing=kwargs_mixing, kwargs_profile_color=kwargs_profile_color)

    ax.set_xlabel(xaxis.replace('_', ' '))
    ax.set_ylabel(yaxis)

    if legend_loc is not None:
        if kwargs_mixing is None:
            kwargs_mixing = pu.get_default_mixing_kwargs()
        labels = []
        handles = []
        for mixtype, val in kd.has_mixtype.items():
            if val:
                hatch = kwargs_mixing[mixtype]['hatch']
                color = kwargs_mixing[mixtype]['color']
                line = kwargs_mixing[mixtype]['line']
                show = kwargs_mixing[mixtype]['show']
                if show:
                    labels.append(uf.mix_dict['names'][uf.mix_dict['merged_r'][mixtype]])
                    handles.append(plt.Rectangle((0,0), 0, 0,
                                                 fill=False, hatch=hatch, edgecolor=color, linewidth=line))
        if len(labels) > 0:
            if legend_loc == 'top':
                legend = pu.top_figure_legend(f, 4, handles=handles, labels=labels)
            elif legend_loc == 'right':
                legend = pu.side_figure_legend(f, 1, handles=handles, labels=labels)
            else:
                pass

    if add_cbar and caxis:
        norm, cmap = kd.color_info

        f = ax.get_figure()
        f.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax)
        cbax = f.axes[-1]
        cbax.set_ylabel(caxis)

    if return_Kipp_data:
        return f, ax, kd
    return f, ax
