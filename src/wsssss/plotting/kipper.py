#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import dill
import joblib.externals.loky.process_executor
import numpy as np

from scipy import sparse
from scipy.spatial import Delaunay
from scipy import integrate as ig
from joblib import Parallel, delayed

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.path import Path
from matplotlib.patches import PathPatch

# For debugging creation of connected zones
# Uncomment these imports, the make_path and plot_graph methods, and the associated lines in calc_zones
# import shapely as sh
# import networkx as nwx

from .. import functions as uf
from .. import load_data as ld
from . import utils as pu


class Kipp_data:
    def __init__(self, hist, profs, xaxis='model_number', yaxis='mass', caxis='eps_net', norm=None,
                 zone_filename='zones_wsssss.dat', verbose=False, save_zones=True, clobber_zones=False,
                 prof_prefix='profile', prof_suffix='.data', prof_resolution=500, parallel=True, ignore_monotonic=False):
        self.__version__ = '0.1.0'
        self.parallel = parallel
        self.verbose = verbose
        self.xaxis = xaxis
        self.yaxis = yaxis
        self.caxis = caxis
        self.xaxis_data = np.empty(len(hist)+1)
        self.xaxis_data[:-1] = hist.get(self.xaxis)
        self.xaxis_data[-1] = self.xaxis_data[-2] + (self.xaxis_data[-2] - self.xaxis_data[-3])
        self.has_mixtype = {}
        self.norm = norm
        self.color_info = None

        if os.name == 'nt':
            if parallel:
                print('Cannot use parallel on windows, settings to False.')
                self.parallel = False

        # Check if monotonic
        if not ignore_monotonic and not np.all(np.diff(np.sign(np.diff(self.xaxis_data))) == 0):
            raise ValueError(f'xaxis {xaxis} is not monotinically increasing or decreasing.')

        if zone_filename:
            self.zone_file = f'{hist.LOGS}/{zone_filename}'
        else:
            self.zone_file = ''
        self.load_zones = bool(self.zone_file)
        self.save_zones = save_zones and self.load_zones  # Also check if zone filename not empty
        self.prof_prefix = prof_prefix
        self.prof_suffix = prof_suffix
        self.prof_resolution = prof_resolution

        if yaxis == 'mass':
            self.mix_template_type = 'mix_type_{}'
            self.mix_template_top = 'mix_qtop_{}'
            self.burn_template_type = 'burn_type_{}'
            self.burn_template_top = 'burn_qtop_{}'
        elif yaxis == 'radius':
            self.mix_template_type = 'mix_relr_type_{}'
            self.mix_template_top = 'mix_relr_top_{}'
            self.burn_template_type = 'burn_relr_type_{}'
            self.burn_template_top = 'burn_relr_top_{}'
        else:
            self.mix_template_type = ''
            self.mix_template_top = ''
            self.burn_template_type = ''
            self.burn_template_top = ''

        if clobber_zones:
            self.load_zones = False

        if self.load_zones and os.path.exists(self.zone_file):
            if os.path.getmtime(self.zone_file) < os.path.getmtime(hist.path):
                self.load_zones = False
                if self.verbose:
                    print('zone file is older than loaded history file! Reloading.')
            else:
                self.load_zones = True
        else:
            self.load_zones = False

        burn_zones = None
        color_zones = None
        if self.load_zones:
            if self.verbose:
                print(f'Loading zonefile {self.zone_file}')
            try:
                loaded_version, loaded_yaxis, mixing_zones, loaded_caxis, burn_zones, yminmax = self.read_zones()
                self.ymin = yminmax[0]
                self.ymax = yminmax[1]
                if loaded_version != self.__version__:
                    if self.verbose:
                        print(f'Kipp_data version mismatch: {loaded_version} {self.__version__}')
                    load_success = False
                else:
                    load_success = True
                    if self.verbose:
                        print(f'Loading successfull. {len(mixing_zones)} mixing zones.')
            except Exception as exc:
                if self.verbose:
                    print(f'Loading zonefile failed. Recreating zones.')
                    print(exc)

                load_success = False

        if (not self.load_zones) or (not load_success):
            if self.caxis == 'eps_net':
                burn_zones = self.calc_zones(*self.get_hist_data(hist, 'burn'))
            mixing_zones = self.calc_zones(*self.get_hist_data(hist, 'mix'))
            self.load_zones = False

        if self.load_zones:
            # Check if loaded data is valid with inputs
            if loaded_yaxis != self.yaxis:
                if self.verbose:
                    print(f'yaxis from loaded zones and settings mismatch, recalculating yaxis and caxis: {loaded_yaxis} {yaxis}')
                mixing_zones = self.calc_zones(*self.get_hist_data(hist, 'mix'))
                if caxis == 'eps_net':
                    burn_zones = self.calc_color(hist, profs)
            elif loaded_caxis != self.caxis:
                if self.verbose:
                    print(f'caxis from loaded zones and settings mismatch, recalculating caxis: {loaded_caxis} {caxis}')
                if caxis == 'eps_net':
                    burn_zones = self.calc_color(hist, profs)
            else:  # Loaded correctly
                self.save_zones = False

        if self.parallel:
            joblib.externals.loky.get_reusable_executor().shutdown(wait=True)  # Kill workers

        self.mixing_zones = mixing_zones
        self.burn_zones = burn_zones

        if self.caxis not in ('', None, 'eps_net'):
            self.color_zones = self.calc_color(hist, profs, norm)
        else:
            self.color_zones = None
        if self.save_zones or clobber_zones:
            self.write_zones()

    def write_zones(self):
        if self.verbose:
            print(f'Writing zonefile {self.zone_file}')
        with open(self.zone_file, 'wb') as handle:
            dill.dump((self.__version__, self.yaxis, self.mixing_zones, self.caxis, self.burn_zones, (self.ymin, self.ymax)), handle)

    def read_zones(self):
        if self.verbose:
            print(f'Reading zonefile {self.zone_file}')
        with open(self.zone_file, 'rb') as handle:
            return dill.load(handle)

    def get_profile_xyz_data(self, hist, profs, norm):

        xyz_data = []
        for i, p in enumerate(profs):
            y = p.get(self.yaxis)
            c = p.get(self.caxis)

            #Normalize
            y_min = np.min(y)
            y_max = np.max(y)
            y_norm = (y - y_min)/(y_max - y_min)
            c_norm = norm(c)
            c_norm = c_norm.filled(np.nan)

            idx = pu.decimate_RDP(np.asarray([y_norm, c_norm]).T, epsilon=1/self.prof_resolution, return_index=True)

            # If using norm's clip, remove first and last block of clipped values.
            if norm.clip:
                if (c_norm[idx[0]] <= 0 or (c_norm[idx[0]] >= 1)):
                    idx = idx[1:]
                if (c_norm[idx[-1]] <= 0 or (c_norm[idx[-1]] >= 1)):
                    idx = idx[:-1]
            xyz = np.zeros((3, idx.shape[0]))

            xyz[0] = p.get_hist_index(hist)
            xyz[1] = y[idx]
            xyz[2] = norm.inverse(c_norm[idx])
            xyz_data.append(xyz)

        xyz_data = np.concatenate(xyz_data, axis=1)
        return xyz_data

    def get_hist_data(self, hist, kind):
        if kind == 'mix':
            template_top = self.mix_template_top
            template_type = self.mix_template_type
            bad_value = -1
        elif kind == 'burn':
            template_top = self.burn_template_top
            template_type = self.burn_template_type
            bad_value = -9999
        else:
            raise ValueError("kind must be either 'mix' or 'burn'.")

        constants = uf.get_constants(hist)
        center_column = ''
        if self.yaxis == 'mass':
            y_scale = hist.get('star_mass')

            if 'm_center' in hist.columns:
                center_column = 'm_center'
            elif 'm_center_gm' in hist.columns:
                center_column = 'm_center_gm'
                unit_scale = 1/constants.msun

        elif self.yaxis == 'radius':
            if 'r_center' in hist.columns:
                center_column = 'r_center'
                unit_scale = 1
            elif 'r_center_cm' in hist.columns:
                center_column = 'r_center_cm'
                unit_scale = 1/constants.rsun
            elif 'r_center_km' in hist.columns:
                center_column = 'r_center_km'
                unit_scale = 1e-3/constants.rsun

            if 'radius' in hist.columns:
                y_scale = hist.get('radius')
            elif 'log_R' in hist.columns:
                y_scale = 10**hist.get('log_R')
            else:
                print('Warning! Using photosphere radius as model radius.')
                y_scale = hist.get('photosphere_R')

        num_cols = len([c for c in hist.columns if c.startswith(template_top.format(''))])
        # for y_data, there is an implied column "0" where y=y_center (typically Y_center=0), so inlcude it explicitly
        y_data = np.zeros((num_cols+1, len(hist)))

        z_data = np.zeros((num_cols, len(hist)), dtype=int)
        for i in range(num_cols):
            y_data[i+1] = hist.get(template_top.format(i + 1)) * y_scale
            z_data[i] = hist.get(template_type.format(i + 1))
        if center_column != '':
            y_data[0] = unit_scale * hist.get(center_column)

        if kind == 'mix':
            z_data = uf.convert_mixing_type(z_data, hist.header['version_number'])

        self.ymax = y_scale.max()
        self.ymin = y_data[0].min()

        return y_data, z_data, bad_value

    def create_adjacency_matrix(self, y_data, z_data, bad_value):

        ny, nx = z_data.shape
        num_zones = ny * nx
        adjacency_matrix = sparse.dok_array((num_zones, num_zones), dtype=int)

        y_top = y_data[1:]
        y_bot = y_data[:-1]

        for i in range(ny):
            q_top = y_top[i]
            q_bot = y_bot[i]
            q_type = z_data[i]

            # zones must overlap in q and must be of same type and must not be bad_value
            connected_right = ~((q_top[np.newaxis, :-1] < y_bot[:, 1:]) | (q_bot[np.newaxis, :-1] > y_top[:, 1:])) & \
                               (q_type[np.newaxis, :-1] == z_data[:, 1:]) & (q_type[np.newaxis, :-1] != bad_value)
            i_y, i_x = np.where(connected_right)
            i_left = i_x + i*nx
            i_right = (i_x + 1) + i_y*nx
            for ii, ij in zip(i_left, i_right):
                adjacency_matrix[ii, ij] = 1  # Only need upper triangle as creating undirected graph

        return adjacency_matrix.tocsr()


    # def make_path(self, i, zone_ids, y_data, z_data):
    #     ny, nx = z_data.shape
    #     mask = zone_ids == i
    #     wmask = np.where(mask)[0]
    #     tops = y_data[1:].reshape((nx*ny))[mask]
    #     bots =  y_data[:-1].reshape((nx*ny))[mask]
    #     kind = z_data.reshape((nx*ny))[wmask[0]]
    #
    #     x = wmask % nx
    #     rects = np.zeros((len(x), 5, 2))
    #     rects[:, [0, 3, 4], 0] = x[:, np.newaxis]
    #     rects[:, [1, 2], 0] = x[:, np.newaxis] + 1
    #     rects[:, [0, 1, 4], 1] = tops[:, np.newaxis]
    #     rects[:, [2, 3], 1] = bots[:, np.newaxis]
    #
    #     rects = sh.polygons(rects)
    #     chunk_size = 32  # ~10% faster if doing chunks of rectangles
    #     poly = sh.union_all([sh.union_all(rects[i*chunk_size:(i+1)*chunk_size]) for i in range(len(rects)//chunk_size + 1)])
    #     # poly = sh.union_all(rects)
    #
    #     if isinstance(poly, sh.MultiLineString) or isinstance(poly, sh.MultiPolygon):  # Has holes
    #         vertices = []
    #         for sub_poly in poly.boundary.geoms:
    #             vertices.append(np.array(sub_poly.xy).T)
    #
    #         lengths = [len(_) for _ in vertices]
    #         num_vert = sum(lengths)
    #         vertices = np.row_stack(vertices)
    #         codes = np.zeros(num_vert)
    #         codes[:] = Path.LINETO
    #         codes[0] = Path.MOVETO
    #         for j in lengths[:-1]:
    #             codes[j + 1] = Path.MOVETO
    #         path = Path(vertices, codes)
    #     elif isinstance(poly, sh.Polygon):
    #         path = Path(np.array(poly.exterior.xy).T)
    #     else:
    #         print(i, type(poly))
    #         raise ValueError
    #     return kind, path


    def make_path2(self, zone_id, adjacency_matrix, zone_ids, y_data, z_data, return_indices=False):
        ny, nx = z_data.shape

        mask = zone_ids == zone_id
        i_adj = np.where(mask)[0]
        order_iadj = np.argsort(i_adj % nx, kind='stable')
        i_adj = i_adj[order_iadj]  # Sort by x then y

        kind = z_data.reshape((nx * ny))[i_adj[0]]
        tops = y_data[1:].reshape((nx * ny))[i_adj]
        bots = y_data[:-1].reshape((nx * ny))[i_adj]
        x = i_adj % nx

        num_nodes = len(i_adj)

        ll = 0  # lower left
        ul = 1  # upper left
        lr = 2  # lower right
        ur = 3  # upper right

        rects = np.zeros((num_nodes, 4, 2))
        rects[:, [ll, ul], 0] = x[:, np.newaxis]
        rects[:, [lr, ur], 0] = x[:, np.newaxis] + 1
        rects[:, [ll, lr], 1] = bots[:, np.newaxis]
        rects[:, [ur, ul], 1] = tops[:, np.newaxis]
        rects = rects.reshape((num_nodes * 4, 2))

        # More convenient indexing for vertices
        i4 = 4 * np.arange(num_nodes)

        un, i_x = np.unique(rects[:, 0], return_inverse=True)
        order = np.argsort(i_x, kind='stable')
        change_x = np.where(np.diff(i_x[order], prepend=-1, append=un[-1]+1))[0]

        # need int32 as otherwise sparse.csgraph.shortestpath breaks
        indices = np.zeros((2, num_nodes*4), dtype=np.int32)
        # Horizontal indices
        indices[:, :num_nodes] = i4 + ll, i4 + lr
        indices[:, num_nodes:num_nodes * 2] = i4 + ur, i4 + ul

        # Vertical indices
        no_holes = True
        max_num = 0
        i_ind_start = 2 * num_nodes
        for i in range(len(change_x)-1):
            i_start = change_x[i]
            i_end = change_x[i + 1]
            num = i_end - i_start
            i_ind_end = i_ind_start + num//2
            if no_holes:
                max_num = max(max_num, num)
            if no_holes and (num < max_num):
                no_holes = False

            o = order[i_start:i_end]
            v = o[np.argsort(rects[o, 1])]  # vertical order
            indices[:, i_ind_start:i_ind_end] = v[::2], v[1::2]
            i_ind_start = i_ind_end

        # Get correct directions for building Path
        # even nodes go to higher node number, odd nodes go to lower node number
        # if mixed, ul -> ll and lr -> ur if the corners belong to the same zone
        # otherwise it is the opposite
        is_odd = np.logical_and(*(indices%2).astype(bool))  # both odd
        is_even = np.logical_and(*np.logical_not((indices % 2).astype(bool)))  # both even
        is_both = ~np.logical_xor(is_even, is_odd)  # one odd one even
        is_left = is_both & (indices[0] % 4 <= 1)
        is_right = is_both & (indices[0] % 4 >= 2)
        is_same = (np.diff(indices//4, axis=0) == 0)[0]

        odd_like = is_odd | (is_right & ~is_same) | (is_left & is_same)
        even_like = is_even | (is_left & ~is_same) | (is_right & is_same)
        indices[:, odd_like] = np.sort(indices[:, odd_like], axis=0)[::-1]
        indices[:, even_like] = np.sort(indices[:, even_like], axis=0)

        if return_indices:
            return rects, indices, no_holes
        return kind, self.path_from_adjacency_matrix(indices, rects, no_holes)


    def path_from_adjacency_matrix(self, indices, nodes, no_holes):
        adjacency_matrix = sparse.coo_array((np.ones(len(nodes), dtype=np.int32), indices),
                                   shape=(len(nodes), len(nodes))).tocsr()
        n_comp, labels = sparse.csgraph.connected_components(adjacency_matrix, return_labels=True)
        path_indices = [np.where(labels==i_path)[0] for i_path in range(n_comp)]
        starts = [pathi[0] for pathi in path_indices]
        dist_mat = sparse.csgraph.shortest_path(adjacency_matrix, unweighted=True, indices=starts)

        offsets = np.cumsum([0] + [len(pathi) for pathi in path_indices])

        order = np.zeros(len(nodes) + n_comp, dtype=int)
        codes = np.full(len(order), Path.LINETO)
        for i in range(n_comp):
            order[offsets[i]+i:offsets[i+1]+i] = np.argsort(dist_mat[i], kind='stable')[:offsets[i+1]-offsets[i]]
            order[offsets[i+1]+i] = order[offsets[i]+i]
        vertices = nodes[order]
        codes[offsets[:-1] + np.arange(n_comp, dtype=int)] = Path.MOVETO

        # Remove double points
        mask = np.sum(np.abs(np.diff(vertices, axis=0, prepend=[[1e99, 1e99],])), axis=1) != 0
        vertices = vertices[mask]
        codes = codes[mask]
        return Path(vertices, codes)

    def calc_zones(self, y_data, z_data, bad_value):
        self.bad_value = bad_value
        ny, nx = z_data.shape
        adjacency_matrix = self.create_adjacency_matrix(y_data, z_data, bad_value)

        _, zone_ids = sparse.csgraph.connected_components(adjacency_matrix, directed=False)
        zone_ids[z_data.reshape(nx*ny) == bad_value] = -1
        zone_ids = np.digitize(zone_ids, bins=np.unique(zone_ids)) - 2

        num_zones = np.max(zone_ids) + 1

        path_function = self.make_path2

        if self.parallel:
            try:
                zones = Parallel(n_jobs=-1)(
                    delayed(self.make_path2)(i, adjacency_matrix, zone_ids, y_data, z_data) for i in range(num_zones))
                # zones = Parallel(n_jobs=1)(
                #     delayed(self.make_path)(i, zone_ids, y_data, z_data) for i in range(num_zones))
            except joblib.externals.loky.process_executor.TerminatedWorkerError as exc:
                print('Parallel failed, falling back to single threaded.')
                zones = [self.make_path2(i, adjacency_matrix, zone_ids, y_data, z_data) for i in range(num_zones)]
        else:
            zones = [self.make_path2(i, adjacency_matrix, zone_ids, y_data, z_data) for i in range(num_zones)]
            # zones = [self.make_path(i, zone_ids, y_data, z_data) for i in range(num_zones)]
        return zones


    def add_mixing(self, ax, xlims, ylims, mixing_min_height, kwargs_mixing=None):
        if self.verbose:
            print('Adding mixing to axis.')
        if kwargs_mixing is None:
            kwargs_mixing = pu.get_default_mixing_kwargs()

        self.has_mixtype = {mix_type:False for mix_type in kwargs_mixing.keys()}

        min_ix = 1e99
        max_ix = -1e99
        get_xlim = True
        if xlims is not None:
            min_ix = np.argwhere(xlims[0] < self.xaxis_data)[0][0]
            min_ix = max(0, min_ix - 1)
            max_ix = np.argwhere(xlims[1] > self.xaxis_data)[0][-1]
            max_ix = min(len(self.xaxis_data) - 1, max_ix + 1)
            get_xlim = False

        if get_xlim:
            if self.color_zones is not None:
                if isinstance(self.color_zones, list):
                    for _, path in self.color_zones:
                        min_ix = min(min_ix, min(path.vertices[:, 0]))
                        max_ix = max(max_ix, max(path.vertices[:, 0]))
                elif isinstance(self.color_zones, np.ndarray):
                    x = self.color_zones[0]  # Only need 1 column
                    min_ix = min(min_ix, min(x))
                    max_ix = max(max_ix, max(x))
            else:
                for _, path in self.mixing_zones:
                    min_ix = min(min_ix, min(path.vertices[:, 0]))
                    max_ix = max(max_ix, max(path.vertices[:, 0]))

        patches_dict = {key:[] for key in kwargs_mixing.keys()}
        patches_dict['order'] = np.zeros((len(kwargs_mixing), 2), dtype=int)
        patches_dict['order'][:,1] = -1  # If still -1 then none were added
        mix_type_index = dict(zip(kwargs_mixing.keys(), np.arange(len(kwargs_mixing))))
        for i, (mix_type, path) in enumerate(self.mixing_zones):
            if mix_type in kwargs_mixing.keys():
                mix_info = kwargs_mixing[mix_type]
                color = mix_info['color']
                hatch = mix_info['hatch']
                line = mix_info['line']
                show = mix_info['show']
                if not show:
                    continue
            else:
                continue

            min_ix = int(min_ix)
            max_ix = int(max_ix)
            x_extent = self.xaxis_data[[min_ix, max_ix]]

            # Skip zones outside xlims
            if path.vertices[:,0].min() > max_ix:
                continue
            if path.vertices[:,0].max() < min_ix:
                continue
            if mixing_min_height > 0:
                if path.vertices[:,1].max() - path.vertices[:,1].min() < mixing_min_height:
                    continue

            # Convert hist index coords to x-data coords
            new_vert = path.vertices.copy()
            new_vert[:,0] = self.xaxis_data[path.vertices[:,0].astype(int)]
            path = Path(new_vert, path.codes)

            # ax.add_patch(PathPatch(path, fill=False, hatch=hatch, edgecolor=color, linewidth=line))
            patches_dict[mix_type].append(PathPatch(path, fill=False, hatch=hatch, edgecolor=color, linewidth=line))
            patches_dict['order'][mix_type_index[mix_type]] = mix_type, i
            self.has_mixtype[mix_type] = True
        patches_dict['order'] = patches_dict['order'][patches_dict['order'][:,1] != -1]  # Remove unused mix_types
        ordered_mixtypes = patches_dict['order'][:,0][np.argsort(patches_dict['order'][:,1])]

        for mix_type in ordered_mixtypes:
            patches = patches_dict[mix_type]
            mix_info = kwargs_mixing[mix_type]
            color = mix_info['color']
            hatch = mix_info['hatch']
            line = mix_info['line']
            show = mix_info['show']
            if not show:
                continue
            ax.add_collection(mpl.collections.PatchCollection(patches, match_original=True, hatch=hatch, edgecolor=color, linewidth=line))
        return x_extent

    def add_color(self, ax, xlims, ylims, clims, norm=None, cmap=None, kwargs_profile_color=None):
        if self.verbose:
            print('Adding color to axis.')
        min_ix = 1e99
        max_ix = -1e99
        get_xlim = True
        if xlims is not None:
            min_ix = np.argwhere(xlims[0] < self.xaxis_data)[0][0]
            min_ix = max(0, min_ix - 1)
            max_ix = np.argwhere(xlims[1] > self.xaxis_data)[0][-1]
            max_ix = min(len(self.xaxis_data)-1, max_ix+1)
            get_xlim = False

        if (self.burn_zones is not None) and self.caxis == 'eps_net':  # Colors from burn_type_* from history.
            if clims is None:
                vmin = min([_[0] for _ in self.burn_zones])
                vmax = max([_[0] for _ in self.burn_zones])
            else:
                vmin, vmax = clims

            if norm is None:
                norm = pu.MidpointBoundaryNorm(np.linspace(vmin, vmax, int(vmax - vmin + 1)), 256, 0)
            if cmap is None:
                cmap = pu.cm.RdBu

            self.color_info = (norm, cmap)

            ax.set_facecolor(cmap(0.5))
            if get_xlim:
                for burn_type, path in self.burn_zones:
                    min_ix = min(min_ix, min(path.vertices[:, 0]))
                    max_ix = max(max_ix, max(path.vertices[:, 0]))
            min_ix = int(min_ix)
            max_ix = int(max_ix)
            x_extent = self.xaxis_data[[min_ix, max_ix]]

            patches = []
            for burn_type, path in self.burn_zones:
                # Keep no/very low burning as middle color and skip drawing as it is already the background color
                if burn_type == 0 and cmap is pu.cm.RdBu:
                    continue
                # Skip zones outside xlims
                if path.vertices[:,0].min() > max_ix:
                    continue
                if path.vertices[:,0].max() < min_ix:
                    continue

                # Convert hist index coords to x-data coords
                new_vert = path.vertices.copy().astype(float)
                new_vert[:, 0] = self.xaxis_data[path.vertices[:, 0].astype(int)]

                path = Path(new_vert, path.codes)
                patches.append(
                    PathPatch(path, fill=True, edgecolor=None, color=cmap(norm(burn_type)),
                              zorder=burn_type - len(self.burn_zones)))
            ax.add_collection(mpl.collections.PatchCollection(patches, match_original=True))

        else:
            x, y, c = self.color_zones
            x = self.xaxis_data[x.astype(int)]

            if clims is None:
                vmin = np.nanmin(c)
                vmax = np.nanmax(c)
            else:
                vmin, vmax = clims

            if kwargs_profile_color is None:
                kwargs_profile_color = {'shading': 'gouraud'}
            else:
                shading = {'shading': 'gouraud'}
                shading.update(kwargs_profile_color)
                kwargs_profile_color = shading

            if norm is not None:
                if 'norm' in kwargs_profile_color.keys():
                    print(f'Using norm from add_color argument.')
                kwargs_profile_color['norm'] = norm
            else:
                if 'norm' not in kwargs_profile_color.keys() or kwargs_profile_color['norm'] is None:
                    kwargs_profile_color['norm'] = mpl.colors.Normalize(vmin, vmax)
                    norm = kwargs_profile_color['norm']
            if cmap is not None:
                if 'cmap' in kwargs_profile_color.keys():
                    print(f'Using cmap from add_color argument.')
            kwargs_profile_color['cmap'] = cmap

            triangulation_pts = self.color_zones.copy().T

            all_simplices = self._triangulate(triangulation_pts[:, :2])

            ax.tripcolor(x.flat, y.flat, c.flat, triangles=all_simplices, **kwargs_profile_color)
            # ax.triplot(x.flat, y.flat, triangles=all_simplices, marker='.')
            x_extent = np.array([x[0], x[-1]])

            self.color_info = (kwargs_profile_color['norm'], kwargs_profile_color['cmap'])
            self.triangulation_pts = triangulation_pts
            self.simplices = all_simplices

        return x_extent

    def _triangulate(self, pts):
        """
        Normalize and triangulate `pts`.

        Args:
            pts:

        Returns:
            Triangulation simplices.
        """
        if not np.all(np.isfinite(pts)):
            raise ValueError(f'Non-finite value in triangulation points.')
        sort_order = np.argsort(pts[:,0], kind='stable')
        undo_sort = np.argsort(sort_order, kind='stable')
        pts = pts[sort_order]

        unique_x, cts = np.unique(pts[:,0], return_counts=True)
        i_start_end = np.cumsum([0, *cts])
        i_start_end[-1] = -1

        pts[:,0] = np.digitize(pts[:,0], unique_x, right=True)

        pts = (pts - pts.min(axis=0)) / (pts.max(axis=0) - pts.min(axis=0))

        all_simplices = []
        for i in range(len(cts) - 1):  # Generate triangulation
            i_start = i_start_end[i]
            i_end = i_start_end[i + 2]
            delan = Delaunay(pts[i_start:i_end], qhull_options='')
            all_simplices.append(delan.simplices + np.sum(cts[:i]))
        all_simplices = np.concatenate(all_simplices)
        all_simplices = undo_sort[all_simplices]
        return all_simplices

    def make_kipp(self, ax=None, xlims=None, ylims=None, clims=None, norm=None, cmap=None, mixing_min_height=0, kwargs_mixing=None,
                  kwargs_profile_color=None):
        if self.verbose:
            print(f'Making figure.')

        f, ax = pu.get_figure(ax)

        mixing_min_height *= (self.ymax - self.ymin)
        if (self.color_zones is not None or self.burn_zones is not None) and self.caxis:
            c_extent = self.add_color(ax, xlims, ylims, clims, norm, cmap, kwargs_profile_color)
        else:
            c_extent = [1e99, -1e99]
        m_extent = self.add_mixing(ax, xlims, ylims, mixing_min_height, kwargs_mixing)

        xextent = np.zeros(2)
        xextent[0] = min(c_extent[0], m_extent[0])
        xextent[1] = max(c_extent[1], m_extent[1])
        ax.set_xlim(xextent)

        if ylims is None:
            ax.set_ylim(self.ymin, self.ymax)

        if self.verbose:
            print('Done.')
        return f, ax

    # def plot_graph(self, adjacency_matrix, z_data, bad_value):
    #     ny, nx = z_data.shape
    #     _, zone_ids = sparse.csgraph.connected_components(adjacency_matrix, directed=False)
    #     mask_bad = z_data.reshape(nx*ny) == bad_value
    #     if np.any(mask_bad):
    #         zone_ids[mask_bad] = -1
    #         zone_ids = np.digitize(zone_ids, bins=np.unique(zone_ids)) - 2
    #     else:
    #         zone_ids = np.digitize(zone_ids, bins=np.unique(zone_ids)) - 2
    #     colors = np.digitize(z_data.reshape(nx*ny), bins=np.unique(z_data))
    #
    #     f, ax = plt.subplots()
    #     ax.imshow(zone_ids.reshape(z_data.shape), aspect='auto', origin='lower')
    #     G = nwx.from_numpy_array(adjacency_matrix)
    #     nwx.draw_networkx(G, pos={node: (node % nx, node // nx) for node in G.nodes},
    #                       node_color=colors,
    #                       with_labels=False, ax=ax,
    #                       node_size=32, hide_ticks=False)

    def calc_color(self, hist, profs, norm=None):
        if self.caxis == 'eps_net' and 'burn_qtop_1' in hist.columns:
            return self.calc_zones(*self.get_hist_data(hist, 'burn'))

        if ((profs is None) or len(profs) == 0):
            profs = ld.load_profs(hist, prefix=self.prof_prefix, suffix=self.prof_suffix)
            if len(profs) == 0:
                return None
            # Check if profiles have required columns
            # required_columns = []
            # if self.yaxis == 'mass':
            #     required_columns.append('mass')
            # elif self.yaxis == 'radius':
            #     required_columns.append('radius')
            # required_columns.append(self.caxis)
            required_columns = [self.yaxis, self.caxis]
            missing_cols = []
            for p in profs:
                for c in required_columns:
                    if not c in p.columns:
                        missing_cols.append((c, p.fname))

            if len(missing_cols) > 0:
                missing_cols_str = ''
                for c, pname in missing_cols:
                    missing_cols_str += f'{pname} {c}\n'
                raise ValueError(f'Missing required columns in profile files:\n'
                                 f'{missing_cols_str}')

        vmin = 1e99
        vmax = -1e99
        for p in profs:
            c = p.get(self.caxis)
            vmin = min(vmin, np.min(c))
            vmax = max(vmax, np.max(c))

        if norm is not None:
            if norm.vmin is not None:
                vmin = norm.vmin
            if norm.vmax is not None:
                vmax = norm.vmax
            norm = norm.__class__(vmin=vmin, vmax=vmax, clip=norm.clip)
        else:
            norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

        return self.get_profile_xyz_data(hist, profs, norm)
