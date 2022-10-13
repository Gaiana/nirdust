#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of the
#   NIRDust Project (https://github.com/Gaiana/nirdust)
# Copyright (c) 2020, 2021 Gaia Gaspar, Jose Alacoria
# License: MIT
#   Full Text: https://github.com/Gaiana/nirdust/LICENSE

# ==============================================================================
# DOCS
# ==============================================================================

"""NIRDust: Near Infrared Dust Finder.

NIRDust is a python package that uses K-band (2.2 micrometers) spectra to
measure the temperature of the dust heated by an Active Galactic Nuclei (AGN)
accretion disk.

"""

# =============================================================================
# CONSTANTS
# =============================================================================

__version__ = "0.2.0"

# =============================================================================
# IMPORTS
# =============================================================================

from .core import *  # noqa
from .bbody import *  # noqa
from .preprocessing import *  # noqa
from .io import *  # noqa
