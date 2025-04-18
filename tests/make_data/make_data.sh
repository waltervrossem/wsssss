#/usr/bin/env bash
# -*- coding: utf-8 -*-

set -e

curdir=$(pwd)
if [ -d ../data/mesa ]; then
  echo "test data/mesa already exists"
else
  ./create_test_data.py
  mesa-go ../data/mesa/ --sub-dirs 0 1 --cmd-post-each "gyre-driver 0 MESA LOGS/profile10.data.GYRE --save-modes --gyre G7; gyre-driver 01 MESA LOGS/*.GYRE --gyre G7 -v; rm star .mesa_temp*" -v
fi
