#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of the
#   NirDust Project (https://github.com/Gaiana/nirdust)
# Copyright (c) 2020, Gaiana Gaspar, Jose Alacoria
# License: MIT
#   Full Text: https://github.com/Gaiana/nirdust/LICENSE ####corregir paht


# =============================================================================
# DOCS
# =============================================================================

"""This file is for distribute and install NirDust
"""


# =============================================================================
# IMPORTS
# =============================================================================

import os
import pathlib

from ez_setup import use_setuptools
use_setuptools()

from setuptools import setup


# =============================================================================
# CONSTANTS
# =============================================================================

REQUIREMENTS = ["numpy", "scipy", "astropy", "specutils"]

PATH = pathlib.Path(os.path.abspath(os.path.dirname(__file__)))

with open(PATH / "README.md") as fp:
    LONG_DESCRIPTION = fp.read()

with open(PATH / "grispy" / "__init__.py") as fp:
    for line in fp.readlines():
        if line.startswith("__version__ = "):
            VERSION = line.split("=", 1)[-1].replace('"', '').strip()
            break


DESCRIPTION = "Grid Search in Python"


# =============================================================================
# FUNCTIONS
# =============================================================================

def do_setup():
    setup(
        name="nirdust",
        version= "0.0.1",
        description="Measure the temperature of the dust heated by an Active Galactic Nuclei (AGN) accretion disk.",
        long_description=open("README.md").read(),
        long_description_content_type='text/markdown',

        author=[
            "Gaiana Gaspar",
            "Jose Alacoria"],
        author_email="gaiagaspar@gmail.com",       
        url="https://github.com/Gaiana/nirdust",
        py_modules=["nirdust", "ez_setup"],
        license="MIT",

        keywords=["nirdust", "emission", "spectra", "nuclear", "temperatures"],

        classifiers=[
            "Intended Audience :: Education",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
            "Programming Language :: Python",
            "Programming Language :: Python :: 3.8"],

        packages=["nirdust"],
        py_modules=["ez_setup"],

        install_requires=REQUIREMENTS)


if __name__ == "__main__":
    do_setup()
