#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# TODO: check environment for $MESA_DIR and get constants from there, if this version and a history/profile have a
# TODO: mismatch raise an error.

from . import functions
from . import load_data
from . import plotting
from . import inlists
import importlib.metadata

__all__ = ['functions', 'load_data', 'plotting', 'inlists']
__version__ = importlib.metadata.version(__package__ or __name__)
del importlib
