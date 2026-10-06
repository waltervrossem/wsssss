#!/usr/bin/env python

import os
import subprocess
import shutil

curdir = os.path.dirname(__file__)


def have_mesa_data():
    test_data = os.path.join(curdir, "data", "mesa")
    if os.path.isdir(test_data):
        shutil.rmtree(test_data)

    print("Extracting mesa_test_data.tgz")
    subprocess.call(f"tar -xzvf {curdir}/data/mesa_test_data.tgz -C {curdir}/data/", shell=True)
