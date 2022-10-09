#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of the
#   NIRDust Project (https://github.com/Gaiana/nirdust)
# Copyright (c) 2020, Gaia Gaspar, Jose Alacoria
# License: MIT
#   Full Text: https://github.com/Gaiana/nirdust/LICENSE


# =============================================================================
# DOCS
# =============================================================================

"""This file is for distribute and install NIRDust
"""


# =============================================================================
# IMPORTS
# =============================================================================

import os
import pathlib

from setuptools import setup

# =============================================================================
# CONSTANTS
# =============================================================================

REQUIREMENTS = [
    "numpy",
    "scipy",
    "astropy",
    "specutils",
    "matplotlib",
    "attrs",
]

PATH = pathlib.Path(os.path.abspath(os.path.dirname(__file__)))

with open(PATH / "README.md") as fp:
    LONG_DESCRIPTION = fp.read()

with open(PATH / "nirdust" / "__init__.py") as fp:
    for line in fp.readlines():
        if line.startswith("__version__ = "):
            VERSION = line.split("=", 1)[-1].replace('"', "").strip()
            break


DESCRIPTION = "Measure temperature of hot dust in type 2 AGN"


# =============================================================================
# FUNCTIONS
# =============================================================================


def do_setup():
    setup(
        name="nirdust",
        version=VERSION,
        description=DESCRIPTION,
        long_description=open("README.md").read(),
        long_description_content_type="text/markdown",
        author=["Gaia Gaspar", "Jose Alacoria"],
        author_email="gaiagaspar@gmail.com",
        url="https://github.com/Gaiana/nirdust",
        py_modules=["ez_setup"],
        packages=[
            "nirdust",
        ],
        license="MIT",
        keywords=["nirdust", "dust", "AGN", "NIR", "temperatures"],
        classifiers=[
            "Intended Audience :: Education",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
            "Programming Language :: Python",
            "Programming Language :: Python :: 3.8",
        ],
        install_requires=REQUIREMENTS,
    )


if __name__ == "__main__":
    do_setup()
