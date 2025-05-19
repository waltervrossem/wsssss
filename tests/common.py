#!/usr/bin/env python

import os

curdir = os.path.dirname(__file__)
def have_mesa_data():
    test_data = os.path.join(os.path.dirname(__file__), 'data', 'mesa')
    if not os.path.exists(test_data):
        print('Extracting mesa_test_data.tgz')
        os.system(f'tar -xzvf {curdir}/data/mesa_test_data.tgz -C {curdir}/data/')
    available_data = os.listdir(test_data)
    for req_data in ['0000', '0001', 'out_0000', 'out_0001']:
        if not req_data in available_data:
            print('Extracting mesa_test_data.tgz')
            os.system(f'rm -r {curdir}/data/mesa && tar -xzvf {curdir}/data/mesa_test_data.tgz -C {curdir}/data/')
            break