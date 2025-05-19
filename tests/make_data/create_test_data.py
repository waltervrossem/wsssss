#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os

from wsssss.inlists import create_grid as cg

this_dir = os.path.dirname(__file__)
mesa_dir = os.environ['MESA_DIR']

grid = cg.MesaGrid()

grid.star_job['profile_columns_file'] = 'profile_test.list'
grid.star_job['history_columns_file'] = 'history_test.list'
grid.star_job['pgstar_flag'] = False
grid.star_job['create_pre_main_sequence_model'] = True

grid.kap['use_Type2_opacities'] = True
grid.kap['Zbase'] = 0.02


grid.controls['initial_mass'] = [1, 2]
grid.controls['initial_z'] = 0.02
grid.controls['max_model_number'] = 1000

grid.controls['profile_interval'] = 100
grid.controls['history_interval'] = 1

grid.controls['write_profiles_flag'] = True
grid.controls['write_pulse_data_with_profile'] = True
grid.controls['pulse_data_format'] = 'GYRE'

grid.add_file(grid.star_job['profile_columns_file'])
grid.add_file(grid.star_job['history_columns_file'])

for fname in ['clean', 'rn', 'mk', 're']:
    grid.add_file(os.path.join(mesa_dir, 'star/work', fname))
for dirname in ['make', 'src']:
    grid.add_dir(os.path.join(mesa_dir, 'star/work', dirname))

if __name__ == '__main__':
    grid.create_grid(os.path.join(this_dir, '../data/mesa'))
