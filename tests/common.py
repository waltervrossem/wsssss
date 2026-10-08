#!/usr/bin/env python
# -*- coding: utf-8 -*-

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


def check_required_environment(required, raise_on_missing=True):
    missing = []
    for env in required:
        if env not in os.environ:
            missing.append(env)
    if raise_on_missing and len(missing) > 0:
        raise EnvironmentError(f"Environment variables not set: {missing}")
    return missing


def check_if_running_in_CI():
    on_github_ci = bool(os.environ.get("GITHUB_ACTIONS", False))
    on_gitlab_ci = bool(os.environ.get("GITLAB_CI", False))

    return on_github_ci or on_gitlab_ci
