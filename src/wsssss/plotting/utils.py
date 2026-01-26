#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib as mpl
import numpy as np
from matplotlib import cm
from matplotlib import colors
from matplotlib import patheffects
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.legend_handler import HandlerLineCollection
from matplotlib.lines import Line2D
from scipy.interpolate import interp1d

from .. import functions as uf
from ..constants import post15140


class HandlerDashedLines(HandlerLineCollection):
    """
    Custom Handler for LineCollection instances.
    """

    # From https://matplotlib.org/stable/gallery/text_labels_and_annotations/legend_demo.html
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        # figure out how many lines there are
        numlines = len(orig_handle.get_segments())
        xdata, xdata_marker = self.get_xdata(legend, xdescent, ydescent,
                                             width, height, fontsize)
        leglines = []
        # divide the vertical space where the lines will go
        # into equal parts based on the number of lines
        ydata = np.full_like(xdata, height / (numlines + 1))
        # for each line, create the line at the proper location
        # and set the dash pattern
        for i in range(numlines):
            legline = Line2D(xdata, ydata * (numlines - i) - ydescent)
            self.update_prop(legline, orig_handle, legend)
            # set color, dash pattern, and linewidth to that
            # of the lines in linecollection
            try:
                color = orig_handle.get_colors()[i]
            except IndexError:
                color = orig_handle.get_colors()[0]
            try:
                dashes = orig_handle.get_dashes()[i]
            except IndexError:
                dashes = orig_handle.get_dashes()[0]
            try:
                lw = orig_handle.get_linewidths()[i]
            except IndexError:
                lw = orig_handle.get_linewidths()[0]
            if dashes[1] is not None:
                legline.set_dashes(dashes[1])
            else:
                legline.set_linestyle(orig_handle.get_linestyle()[i])
            legline.set_color(color)
            legline.set_transform(trans)
            legline.set_linewidth(lw)
            leglines.append(legline)
        return leglines


class MidpointBoundaryNorm(colors.BoundaryNorm):
    """
    Generate a colormap index based on discrete intervals.

    Unlike :class:`Normalize` or :class:`LogNorm`,
    :class:`MidpointBoundaryNorm` maps values to integers instead of to the
    interval 0-1.

    Mapping to the 0-1 interval could have been done via
    piece-wise linear interpolation, but using integers seems
    simpler, and reduces the number of conversions back and forth
    between integer and floating point.
    """

    def __init__(self, boundaries, ncolors, midpoint, clip=False, away_from_midpoint=True, singular_midpoint=True):
        """
        Parameters
        ----------
        boundaries : array-like
            Monotonically increasing sequence of boundaries.
        ncolors : int
            Number of colors in the colormap to be used.
        midpoint : scalar
            Value to assign the middle color to.
        clip : bool, optional
            If clip is ``True``, out of range values are mapped to 0 if they
            are below ``boundaries[0]`` or mapped to ncolors - 1 if they are
            above ``boundaries[-1]``.

            If clip is ``False``, out of range values are mapped to -1 if
            they are below ``boundaries[0]`` or mapped to ncolors if they are
            above ``boundaries[-1]``. These are then converted to valid indices
            by :meth:`Colormap.__call__`.
        away_from_midpoint : bool, optional
            If away_from_midpoint is ``True``, determine colors away from
            the midpoint value instead of all in one direction.
        singular_midpoint : bool, optional
            If singular_midpoint is ``True``, only values which equal midpoint
            are set to that color. Only used when away_from_midpoint is ``True``.

        Notes
        -----
        *boundaries* defines the edges of bins, and data falling within a bin
        is mapped to the color with the same index.

        If the number of bins doesn't equal *ncolors*, the color is chosen
        by linear interpolation of the bin number onto color numbers.
        """
        super().__init__(boundaries, ncolors, clip)
        self.away_from_midpoint = away_from_midpoint
        self.singular_midpoint = singular_midpoint
        self.clip = clip
        self.vmin = boundaries[0]
        self.vmax = boundaries[-1]
        self.midpoint = midpoint
        self.boundaries = np.asarray(boundaries)

        self.N = len(self.boundaries)
        self.Ncmap = ncolors
        if self.N - 1 == self.Ncmap:
            self._interp = False
        else:
            self._interp = True

    def __call__(self, value, clip=None):
        if clip is None:
            clip = self.clip

        xx, is_scalar = self.process_value(value)
        mask = np.ma.getmaskarray(xx)
        xx = np.atleast_1d(xx.filled(self.vmax + 1))
        if clip:
            np.clip(xx, self.vmin, self.vmax, out=xx)
            max_col = self.Ncmap - 1
        else:
            max_col = self.Ncmap
        imid = np.argmin(np.abs(self.boundaries - self.midpoint))
        iret = np.zeros(xx.shape, dtype=np.int16)
        if self.away_from_midpoint:
            if self.singular_midpoint:
                left = xx < self.midpoint
                iret[left] = np.digitize(xx[left], self.boundaries) - 1
                right = xx > self.midpoint
                iret[right] = np.digitize(xx[right], self.boundaries, True)
                iret[~(left | right)] = imid  # Exactly equal to midpoint
            else:
                for i, b in enumerate(self.boundaries[::-1]):
                    if b <= self.midpoint:
                        iret[xx <= b] = self.N - 1 - i
                for i, b in enumerate(self.boundaries):
                    if b >= self.midpoint:
                        iret[xx >= b] = i
        else:
            for i, b in enumerate(self.boundaries):
                iret[xx >= b] = i
        if self._interp:
            if self.away_from_midpoint & self.singular_midpoint:
                scalefac_l = self.Ncmap / 2 / imid
                scalefac_r = self.Ncmap / 2 / (self.N - imid + 1)
                imid = int(self.Ncmap / 2)
                iret[left] = (iret[left] * scalefac_l).astype(np.int16)
                iret[right] = (imid + iret[right] * scalefac_r).astype(np.int16)
                iret[~(left | right)] = imid
            else:
                scalefac = (self.Ncmap - 1) / (self.N - 2)
                iret = (iret * scalefac).astype(np.int16)
                imid = int(imid * scalefac)

        if not (self._interp & self.away_from_midpoint & self.singular_midpoint):
            imask = xx < self.midpoint
            iret[imask] = iret[imask] / imid * 0.5 * (self.Ncmap - 1)
            imask = xx >= self.midpoint
            iret[imask] = (iret[imask] - imid) / (self.Ncmap - imid) * 0.5 * (self.Ncmap - 1) + 0.5 * (self.Ncmap - 1)
            iret = iret.astype(np.int16)

        iret[xx < self.vmin] = -1
        iret[xx >= self.vmax] = max_col
        ret = np.ma.array(iret, mask=mask)
        if is_scalar:
            ret = int(ret[0])  # assume python scalar
        return ret

    def inverse(self, value):
        """
        Raises
        ------
        ValueError
            MidpointBoundaryNorm is not invertible, so calling this method will always
            raise an error
        """
        return ValueError("MidpointBoundaryNorm is not invertible")


def get_figure(ax):
    if ax is None:
        f, ax = plt.subplots()
    else:
        f = ax.get_figure()
    return f, ax


def top_legend(ax, ncol=2, **kwargs):
    return ax.legend(ncol=ncol, bbox_to_anchor=(0., 1.02, 1., .102), loc=3, mode='expand', borderaxespad=0., **kwargs)


def top_figure_legend(f, ncol, top=0.9, **kwargs):
    fig_size = f.bbox.corners()[3]
    f.subplots_adjust(top=top)
    corners = np.array([ax.bbox.corners() for ax in f.axes]) / fig_size

    left = np.min(corners[:, :, 0])
    right = np.max(corners[:, :, 0])
    bot = np.max(corners[:, :, 1])

    width = right - left
    height = 1 - bot - 0.02

    legend = f.legend(loc='upper center', ncol=ncol, mode='expand', borderaxespad=0.,
                      bbox_to_anchor=(left, bot, width, height),
                      **kwargs)
    return legend


def side_figure_legend(f, ncol, right=0.75, **kwargs):
    fig_size = f.bbox.corners()[3]
    f.subplots_adjust(right=right)
    corners = np.array([ax.bbox.corners() for ax in f.axes]) / fig_size

    left = np.max(corners[:, :, 0]) + 0.01
    top = bot = np.max(corners[:, :, 1])
    bot = np.min(corners[:, :, 1])

    width = 1 - left - 0.01
    height = top - bot

    legend = f.legend(loc='upper center', ncol=ncol, mode='expand', borderaxespad=0.,
                      bbox_to_anchor=(left, bot, width, height),
                      **kwargs)
    return legend


def colored_line(f, ax, xdat, ydat, cdat, norm=None, cmap='viridis', lw=2, add_cbar=True, do_lims=False, **kwargs):
    points = np.asarray([xdat, ydat]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap=plt.get_cmap(cmap), zorder=5, lw=1, norm=norm, **kwargs)
    lc.set_array(cdat)
    lc.set_linewidth(lw)
    ax.add_collection(lc)
    if add_cbar:
        f.colorbar(lc, ax=ax)

    if do_lims:
        # Use automatic x and ylimits
        xmin = min(xdat)
        xmax = max(xdat)
        ymin = min(ydat)
        ymax = max(ydat)

        ax.plot([xmin, ymin], [xmax, ymax], lw=0)
        ax.set_prop_cycle(None)
    return lc


def hrd_const_rad(ax, fontsize=8, radii=None, angle=None, linear=False):
    xlim1 = ax.get_xlim()
    ylim1 = ax.get_ylim()
    if linear:
        xlim1 = np.log10(xlim1)
        ylim1 = np.log10(ylim1)
    xw = xlim1[0] - xlim1[1]
    yw = ylim1[1] - ylim1[0]

    xfactor = 0.07 / 4 * yw / xw

    # at this scale doesn't matter which constants are used
    const = 4 * np.pi * post15140.boltz_sigma / post15140.lsun * post15140.rsun ** 2
    log_const = np.log10(const)
    logTs = np.linspace(0, 6, 2)
    if radii is None:
        radii = np.logspace(-1, 6, 8)

    radii = np.unique(radii)
    radii = radii[radii > 0]
    _radii = []
    for rad in radii:
        # Do number string formatting first so calculated and displayed radii are equal
        if max(radii) < 10:
            nr_string = f'{rad:.1f}'
        else:
            if rad < 1:
                nr_string = f'{rad:.1f}'
            else:
                nr_string = f'{rad:.0f}'
        rad = float(nr_string)
        if (rad in _radii) or (rad <= 0):
            continue
        _radii.append(rad)

        logR = np.log10(rad)
        logLs = 2 * logR + 4 * logTs + log_const
        if linear:
            ax.plot(10**logTs, 10**logLs, 'grey', linestyle='--', zorder=-10, lw=1)
        else:
            ax.plot(logTs, logLs, 'grey', linestyle='--', zorder=-10, lw=1)

        xleft = np.interp(ylim1[1], logLs, logTs)
        xright = np.interp(ylim1[0], logLs, logTs)
        xleft = min(xleft, xlim1[0])
        xright = max(xright, xlim1[1])

        yleft = np.interp(xleft, logTs, logLs)
        yright = np.interp(xright, logTs, logLs)

        is_top = (xleft < xlim1[0])
        is_bottom = (xright > xlim1[1])
        textxs = []

        if is_top:
            textxs.append(xleft - xfactor * xw)
            textxs.append(xright + 0.07 * xw)
        elif is_bottom:
            textxs.append(xright + xfactor * xw)
            textxs.append(xleft - 0.07 * xw)
        else:
            textxs.append(xleft - 0.07 * xw)
            textxs.append(xright + 0.07 * xw)
        frac_x_sep = abs(np.diff(textxs)) / xw
        if frac_x_sep < 0.1:
            textxs.pop(0)

        for i, textx in enumerate(textxs):

            if angle is None:
                # Have to reset xylims due to the plotting the lines of constant radius, otherwise get wrong angles.
                if linear:
                    ax.set_ylim(10**ylim1)
                    ax.set_xlim(10**xlim1)
                    screen_dx, screen_dy = ax.transData.transform((10**logTs[0], 10**logLs[0])) - ax.transData.transform(
                        (10**logTs[-1], 10**logLs[-1]))
                else:
                    ax.set_ylim(ylim1)
                    ax.set_xlim(xlim1)
                    screen_dx, screen_dy = ax.transData.transform((logTs[0], logLs[0])) - ax.transData.transform(
                        (logTs[-1], logLs[-1]))
                angle = (np.degrees(np.arctan2(screen_dy, screen_dx)) + 90) % 180 - 90
            texty = np.interp(textx, logTs, logLs)
            if linear:
                textx = 10**textx
                texty = 10**texty
            text = ax.text(textx, texty, r"${}\;\mathrm{{R}}_{{\odot}}$".format(nr_string), color='k', zorder=-9,
                           rotation=angle, size=fontsize, va='center', ha='center', clip_on=True,
                           bbox=dict(facecolor='white', linewidth=0, pad=0))
            text.set_path_effects(
                [patheffects.Stroke(linewidth=3, foreground=ax.get_facecolor()), patheffects.Normal()])
    if linear:
        ax.set_ylim(10**ylim1)
        ax.set_xlim(10**xlim1)
    else:
        ax.set_ylim(ylim1)
        ax.set_xlim(xlim1)


def get_x_and_set_xlabel(p, xname, ax=None, func_on_xaxis=None, hist=None):
    if isinstance(xname, list) or isinstance(xname, tuple):
        x, xname = xname
        label = xname
    elif xname.lower().replace('_', '') in ['radius', 'r', 'rsol', 'rsun']:
        x = uf.get_radius(p, 'Rsol')
        label = r'Radius $(R_\odot)$'
    elif xname in ['radius_cm', 'r_cm']:
        x = uf.get_radius(p)
        label = r'Radius (cm)'
    elif xname in ('x', 'radius_dimless'):
        x = uf.get_radius(p)
        x = x / max(x)
        label = r'Fractional radius $x$ $( r /\mathrm{R}_\star)$'
    elif xname == 'logR':
        x = uf.get_radius(p, 'Rsol')
        label = r'Radius $(r/\mathrm{R}_\odot)$'
    elif xname == 'mass':
        x = p.data.mass
        label = r'Mass coordinate $m/\mathrm{M}_\odot$'
    elif xname in ['q', 'mass_dimless']:
        x = p.data.mass
        x = x / x[0]
        label = r'Fractional mass $q$ $( m /\mathrm{M}_\star)$'
    elif xname in ['zone', 'zone_number']:
        x = np.arange(len(p.data)) + 1
        label = 'Zone Number'
    elif xname == 's':
        if hist is None:
            raise ValueError("Need History for xname='s'.")
        i_hist = p.get_hist_index(hist)
        r1 = hist.data.r_1[i_hist]
        r2 = hist.data.r_2[i_hist]
        r0 = np.sqrt(r1 * r2)
        x = np.log(p.data.radius / r0)
        label = '$s$'
    else:
        x = p.get(xname)
        label = xname.replace('_', ' ')
    if ax is not None:
        ax.set_xlabel(label)

    if func_on_xaxis is not None:
        x = func_on_xaxis(x)
    return x


def get_mix_dict(profile):
    mesa_ver = str(profile.header.get('version_number'))
    if uf.compare_version(mesa_ver, '15140', '>='):
        prefix = 'post'
    else:
        prefix = 'pre'
    mix_dict = uf.mix_dict[f'{prefix}15140']
    return mix_dict


def get_mixing(profile, min_width):
    """

    :param profile:
    :param min_width:
    :return:
    """
    mix_dict = get_mix_dict(profile)

    mixing = profile.get('mixing_type')
    regions = {}
    for i, name in mix_dict.items():
        # if i == 0:
        #    continue
        good_zones = np.where(mixing == i)[0]
        changes = np.where(np.diff(good_zones) != 1)[0]

        starts = []
        ends = []
        if len(good_zones) > 1:
            starts.append(good_zones[0])
            for change in changes:
                ends.append(good_zones[change] + 1)  # Add one to include final cell
                starts.append(good_zones[change + 1])
            ends.append(good_zones[-1])
        pairs = zip(starts, ends)
        regions[name] = [pair for pair in pairs if pair[1] - pair[0] >= min_width]
    return regions

def get_default_mixing_kwargs():
    """
    Get the default coloring scheme for mixing.
    Returns:
        dict of dict: Each dict contains kwargs for PathPatch and `show`, which if `False` mean that it will not be
            drawn.
    """
    default_kwargs_mixing = {
        uf.mix_dict['merged']['convective_mixing']: {'name': 'convective_mixing',
                                                     'color': "Chartreuse",
                                                     'hatch': "//",
                                                     'line': 1,
                                                     'show': True
                                                     },
        uf.mix_dict['merged']['overshoot_mixing']: {'name': 'overshoot_mixing',
                                                    'color': "purple",
                                                    'hatch': "x",
                                                    'line': 1,
                                                    'show': True
                                                    },
        uf.mix_dict['merged']['semiconvective_mixing']: {'name': 'semiconvective_mixing',
                                                         'color': "red",
                                                         'hatch': "\\\\",
                                                         'line': 1,
                                                         'show': True
                                                         },
        uf.mix_dict['merged']['thermohaline_mixing']: {'name': 'thermohaline_mixing',
                                                       'color': "Gold",
                                                       'hatch': "||",
                                                       'line': 1,
                                                       'show': False
                                                       },
        uf.mix_dict['merged']['rotation_mixing']: {'name': 'rotation_mixing',
                                                   'color': "brown",
                                                   'hatch': "*",
                                                   'line': 1,
                                                   'show': True
                                                   },
        uf.mix_dict['merged']['anonymous_mixing']: {'name': 'anonymous_mixing',
                                                    'color': "tab:grey",
                                                    'hatch': "+",
                                                    'line': 1,
                                                    'show': True
                                                    },
        uf.mix_dict['merged']['minimum_mixing']: {'name': 'minimum_mixing',
                                                  'color': "cyan",
                                                  'hatch': "-",
                                                  'line': 1,
                                                  'show': True
                                                  },
        uf.mix_dict['merged']['no_mixing']: {'name': 'no_mixing',
                                             'color': "",
                                             'hatch': "",
                                             'line': 0,
                                             'show': False
                                             },

    }
    return default_kwargs_mixing

def add_mixing(ax, profile, xname='mass', min_width=5, ymin=0, ymax=1, alpha=1, add_legend=True, func_on_xaxis=None,
               kwargs_mixing=None):
    """
    kwargs_mixing (dict, optional): kwargs used to draw mixing regions, if `None`, defaults to `plotting.utils.get_default_mixing_kwargs()`.
    """
    if kwargs_mixing is None:
        kwargs_mixing = get_default_mixing_kwargs()

    x = get_x_and_set_xlabel(profile, xname, func_on_xaxis=func_on_xaxis)

    mix_dict = get_mix_dict(profile)
    all_regions = get_mixing(profile, min_width)
    added_to_legend = []
    for _, name in mix_dict.items():
        regions = all_regions[name]
        if len(regions) == 0:
            continue
        mix_type = uf.mix_dict['merged'][name]
        mix_info = kwargs_mixing[mix_type]
        color = mix_info['color']
        hatch = mix_info['hatch']
        line = mix_info['line']
        show = mix_info['show']

        if not show:
            continue

        for region in regions:
            start, end = region
            xstart = x[start]
            xend = x[end]

            if add_legend and name not in added_to_legend:
                ax.axvspan(xstart, xend, ymin, ymax, hatch=hatch, edgecolor=color, linewidth=line,
                           facecolor='none', label=name, zorder=-1, alpha=alpha)
                added_to_legend.append(name)
            else:
                ax.axvspan(xstart, xend, ymin, ymax, hatch=hatch, edgecolor=color, linewidth=line,
                           facecolor='none', zorder=-1, alpha=alpha)


def add_burning(ax, profile, xname='mass', ymin=0, ymax=1, num_levels=None, vmin=-2, vmax=8, add_cbar=True,
                kind='net_nuclear_energy', norm=None, kipp_scaling=True, func_on_xaxis=None):
    """
    """
    x = get_x_and_set_xlabel(profile, xname, func_on_xaxis=func_on_xaxis)

    if kind == 'eps_nuc':
        log_z = np.log10(profile.get('eps_nuc'))
        cbar_label = r'$\log_{10} \; \epsilon_{\mathrm{nuc}}$'
    elif kind == 'net_nuclear_energy':
        log_z = profile.get('net_nuclear_energy')
        if kipp_scaling:
            log_z = np.sign(log_z) * np.ceil(np.abs(log_z))
        cbar_label = r'$\mathrm{sign}(\epsilon_\mathrm{net}) \; \log_{10} \; \max(1, \lceil|\epsilon_\mathrm{net}|\rceil)/(\mathrm{erg}/g/s)$'
    else:
        raise ValueError('`eps_nuc` or `net_nuclear_energy` must be a column in profile.')
    # log_z[log_z < vmin] = vmin
    if num_levels is None:
        num_levels = int(vmax - vmin + 1)

    if (kind == 'net_nuclear_energy') and (vmin < 0):
        levels = np.linspace(vmin - 0.5, vmax + 0.5, num_levels + 1)
        cmap = plt.cm.RdBu
        if norm is None:
            norm = MidpointBoundaryNorm(levels, cmap.N, midpoint=0, singular_midpoint=True)
    else:
        levels = np.linspace(vmin, vmax, num_levels)
        cmap = plt.cm.Blues
        if norm is None:
            norm = colors.BoundaryNorm(levels, cmap.N)

    log_z_disc = norm(log_z)
    i_max = len(log_z_disc) - 1
    pairs = []
    for level in levels:
        mask = log_z_disc == norm(level)
        diff = np.diff(mask.astype(int))
        transitions = list(np.where(diff != 0)[0])
        if len(transitions) == 0:
            continue

        if mask[0]:
            transitions = [0] + transitions
        if mask[-1]:
            transitions = transitions + [i_max]
        for i in range(len(transitions) // 2):
            i_left = transitions[2 * i]
            i_right = transitions[2 * i + 1]
            pairs.append((level, (i_left, i_right)))

    for level, (i_left, i_right) in pairs:
        ax.axvspan(x[i_left], x[i_right], ymin, ymax, color=cmap(norm(level)), ec=None, zorder=-2)

    if add_cbar:
        f = ax.get_figure()
        f.colorbar(cm.ScalarMappable(cmap=cmap, norm=norm), ticks=np.linspace(vmin, vmax, num_levels)[1::2], ax=ax)
        cbax = f.axes[-1]
        cbax.set_ylabel(cbar_label)

    return cm.ScalarMappable(cmap=cmap, norm=norm), cbar_label


def add_hrd_instabilities(ax, classic=True, sdB=True, ):
    xlims = ax.get_xlim()
    ylims = ax.get_ylim()

    if classic:
        ax.plot((3.95, 3.9, 3.8), (1, 1.8, 4.5), 'k--')
        ax.plot((3.85, 3.8, 3.7), (0.8, 1.6, 4.5), 'k--')
    if sdB:
        sdBVr_logTeff = (4.55, 4.45)
        sdBVr_logL = (1, 1.8)
        patch_sdBVr = mpl.patches.Rectangle((sdBVr_logTeff[0], sdBVr_logL[0]),
                                            width=np.diff(sdBVr_logTeff), height=np.diff(sdBVr_logL),
                                            fill=False, edgecolor='b', hatch='\\')
        ax.add_patch(patch_sdBVr)

        sdBVs_logTeff = (4.5, 4.35)
        sdBVs_logL = (1.4, 1.8)
        patch_sdBVr = mpl.patches.Rectangle((sdBVs_logTeff[0], sdBVs_logL[0]),
                                            width=np.diff(sdBVs_logTeff), height=np.diff(sdBVs_logL),
                                            fill=False, edgecolor='b', hatch='/')
        ax.add_patch(patch_sdBVr)

    ax.set_xlim(xlims)
    ax.set_ylim(ylims)


def calc_inertia_marker_size(gs, l, freq_units='uHz'):
    mask = gs.data.l == 0
    E_l0 = gs.get('E_norm')[mask]
    nu_all = gs.get_frequencies(freq_units)
    nu_l0 = nu_all[mask]
    # interpolate over log10 inertia for better behaviour
    try:
        log_f_El0 = interp1d(nu_l0, np.log10(E_l0), kind='cubic', bounds_error=True)
    except ValueError:
        log_f_El0 = interp1d(nu_l0, np.log10(E_l0), kind='linear', bounds_error=True)
    mask = gs.data.l == l
    nu = gs.get_frequencies(freq_units)[mask]
    # x = np.log10(gs.data.E_norm[mask]) - log_f_El0(nu_all[mask])
    # ms = 2.5 * 10 ** (2 * (1 - x))
    xmin = min(E_l0)
    x = np.log10(gs.data.E_norm[mask] / xmin)
    ms = 25 - 2 * x ** 3
    ms = np.minimum(25, ms)
    ms = np.maximum(1, ms)
    return ms


def line_legend(ax, edge_space=0.05, num_line_label=4, fontsize=7, background=None, background_width=3, ignore_nolabel=False):
    xmin, xmax = ax.get_xlim()
    x_range = xmax - xmin
    line_label_x = np.linspace(xmin + edge_space * x_range, xmax - edge_space * x_range, num=num_line_label)
    if background is None:
        background = ax.get_facecolor()
    # Partially based on https://github.com/cphyc/matplotlib-label-lines
    for line in ax.get_lines():
        label = line.get_label()
        xdata = line.get_xdata()
        ydata = line.get_ydata()

        if ignore_nolabel:
            if label.startswith('_child'):
                continue

        for x in line_label_x:
            idxs = np.where(np.diff(np.sign(xdata - x)) != 0)[0] + 1
            for idx in idxs:
                # Data ordered from surface to core
                x_left = xdata[idx]
                x_right = xdata[idx - 1]
                y_left = ydata[idx]
                y_right = ydata[idx - 1]
                y = np.interp(x, (x_left, x_right), (y_left, y_right))

                screen_dx, screen_dy = ax.transData.transform((x_left, y_left)) - ax.transData.transform(
                    (x_right, y_right))
                rotation = (np.degrees(np.arctan2(screen_dy, screen_dx)) + 90) % 180 - 90

                text = ax.text(x, y, label, rotation=rotation, color=line.get_color(), ha='center', va='center',
                               clip_on=True, size=fontsize, bbox={'alpha': 0})
                text.set_path_effects(
                    [patheffects.Stroke(linewidth=background_width, foreground=background),
                     patheffects.Normal()])



def decimate_RDP(pts, epsilon, return_index=False):
    """
    Ramer–Douglas–Peucker decimation algorithm with some adjustments.
    Args:
        pts (np.array): 2-d array containing points to decimate.
        epsilon: Tolerance for keeping line segments
        return_index (bool, optional): If True, return indeces instead of values. Defaults to False.

    Returns:
        np.array: Decimated version of pts.
    """
    good = np.isfinite(pts[:,1])
    good_loc = np.where(good)[0]
    last_i = good_loc[-1]
    # First and last non-nan point
    i_start = good_loc[0]
    i_end = good_loc[-1]

    if return_index:
        new_pts = np.zeros(len(pts), dtype=int)
        new_pts[0] = i_start
    else:
        new_pts = np.zeros_like(pts)
        new_pts[0] = pts[i_start]
    i_insert = 1

    while i_start <= last_i:
        pt0, pt1 = pts[i_start], pts[i_end]
        if np.isnan(pt0 + pt1).any():  # Keep one NaN as separator for blocks
            if return_index:
                new_pts[i_insert] = i_end
            else:
                new_pts[i_insert] = pts[i_end]
            i_insert += 1
            i_start = good_loc[good_loc > i_end][0]  # First good point after current end
            i_end = last_i
            pt0, pt1 = pts[i_start], pts[i_end]

        if i_end - i_start <= 1:  # If next point has difference larger than epsilon keep it and start from next.
            if return_index:
                new_pts[i_insert] = i_end
            else:
                new_pts[i_insert] = pts[i_end]
            i_insert += 1
            i_start += 1
            i_end = last_i
            pt0, pt1 = pts[i_start], pts[i_end]

        # Perpendicular distance
        delta = np.abs(np.cross(pt1 - pt0, pt0 - pts[i_start + 1:i_end]) / np.linalg.norm(pt1 - pt0))
        try:
            i_dmax = np.nanargmax(delta)
        except ValueError:
            print(i_start, i_end, pt0, pt1)
            print(np.linalg.norm(pt1 - pt0))
            print(delta)
            raise
        dmax = delta[i_dmax]

        if dmax <= epsilon:  # Keep i_end and skip points in between
            if return_index:
                new_pts[i_insert] = i_end
            else:
                new_pts[i_insert] = pts[i_end]
            i_insert += 1
            i_start = i_end
            i_end = last_i
        else:
            i_end = i_start + i_dmax + 1
        if i_start >= last_i:  # Done
            break
    new_pts = new_pts[:i_insert]
    return new_pts
