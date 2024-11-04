#/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import shutil
import sys
import unittest
import subprocess
from wsssss._bin import mesa_go
from wsssss.inlists import create_grid as cg

must_have_environ = ['MESA_DIR', 'MESASDK_ROOT']
for env in must_have_environ:
    if env not in os.environ:
        raise EnvironmentError(f'{env} not set.')
if (os.environ['MESASDK_ROOT'] not in os.environ['PATH']) or 'MESASDK_VERSION' not in os.environ:
    raise EnvironmentError('The MESASDK has not been initialized.')

mesa_dir = os.environ['MESA_DIR']

class TestMesaGO(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.init_dir = os.path.abspath('.')
        cls.base_grid_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/mesago'))
        grid = cg.MesaGrid()

        grid.star_job['history_columns_file'] = 'history_test.list'
        grid.star_job['pgstar_flag'] = False
        grid.star_job['create_pre_main_sequence_model'] = False

        grid.kap['use_Type2_opacities'] = True
        grid.kap['Zbase'] = 0.02

        grid.controls['initial_mass'] = [1, 2]
        grid.controls['initial_z'] = 0.02
        grid.controls['max_model_number'] = 10
        grid.controls['mesh_delta_coeff'] = 5

        grid.controls['history_interval'] = 1
        grid.controls['photo_interval'] = 8

        grid.controls['write_profiles_flag'] = False

        grid.add_file(os.path.join(os.path.dirname(__file__), '..', 'make_data', grid.star_job['history_columns_file']))

        for fname in ['clean', 'rn', 'mk', 're']:
            grid.add_file(os.path.join(mesa_dir, 'star/work', fname))
        for dirname in ['make', 'src']:
            grid.add_dir(os.path.join(mesa_dir, 'star/work', dirname))

        cls.grid = grid

    def setUp(self):
        testname = self.id().split('.')[-1]
        self.grid_dir = os.path.join(self.base_grid_dir, testname)
        self.grid.create_grid(self.grid_dir)
        print(self.grid_dir, os.path.isdir(self.grid_dir))
        os.chdir(self.grid_dir)
        sys.argv = ['mesa-go', '']

    @classmethod
    def tearDownClass(cls):
        # if os.path.isdir(cls.base_grid_dir):
        #     shutil.rmtree(f'{cls.base_grid_dir}')
        os.chdir(cls.init_dir)

    def check_output(self):
        os.chdir(self.grid_dir)
        output = subprocess.run(['check-grid', '--no-slurm', '--out-file', '../out_{}'], stdout=subprocess.PIPE)

        expected = ("--------------------------------------------\n"
                    "  termination_code                   count\n"
                    "--------------------------------------------\n"
                    "max_model_number                           2\n"
                    "--------------------------------------------\n"
                    "\n")
        out_str = output.stdout.decode()
        self.assertEqual(expected, out_str)

    def test_mesago(self):
        sys.argv = ['mesa-go', '--verbose', '--cmd-pre', 'touch pre', '--cmd-post', 'touch post']
        ierr = mesa_go.run()
        if ierr != 0:
            raise SystemError(ierr)
        self.check_output()
        self.assertTrue(os.path.isfile(f'{self.grid_dir}/pre'))
        self.assertTrue(os.path.isfile(f'{self.grid_dir}/post'))
        os.remove(f'{self.grid_dir}/pre')
        os.remove(f'{self.grid_dir}/post')
        for dirname in self.grid.dirnames:
            with open(f'{self.grid_dir}/out_{dirname}', 'r') as handle:
                lines = handle.readlines()
            self.assertEqual(131, len(lines))

    def test_mesago_each(self):
        sys.argv = ['mesa-go', '--verbose', '--cmd-pre-each', 'cp ../../test_mesago/0000/star ./; touch preeach', '--cmd-post-each', 'touch posteach']
        ierr = mesa_go.run()
        if ierr != 0:
            raise SystemError(ierr)
        self.check_output()
        for dirname in self.grid.dirnames:
            self.assertTrue(os.path.isfile(f'{self.grid_dir}/{dirname}/preeach'))
            self.assertTrue(os.path.isfile(f'{self.grid_dir}/{dirname}/posteach'))
            os.remove(f'{self.grid_dir}/{dirname}/preeach')
            os.remove(f'{self.grid_dir}/{dirname}/posteach')

        for dirname in self.grid.dirnames:
            with open(f'{self.grid_dir}/out_{dirname}', 'r') as handle:
                lines = handle.readlines()
            self.assertEqual(131, len(lines))

    def test_mesago_restart(self):
        sys.argv = ['mesa-go', '--verbose', '--restart']
        for dirname in self.grid.dirnames:
            os.makedirs(f'{self.grid_dir}/{dirname}/photos/')
            shutil.copy2(os.path.join(self.base_grid_dir, 'test_mesago', dirname, 'photos/x008'),
                          f'{self.grid_dir}/{dirname}/photos/')
            shutil.copy2(os.path.join(self.base_grid_dir, 'test_mesago', dirname, 'star'),
                                      f'{self.grid_dir}/{dirname}/')
        ierr = mesa_go.run()
        if ierr != 0:
            raise SystemError(ierr)
        self.check_output()
        for dirname in self.grid.dirnames:
            with open(f'{self.grid_dir}/out_{dirname}', 'r') as handle:
                lines = handle.readlines()
            self.assertEqual(60, len(lines))
