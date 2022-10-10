.. Nirdust documentation master file, created by
   sphinx-quickstart on Wed Feb 17 15:19:30 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

NIRDust
========================

.. only:: html


.. image:: https://badge.fury.io/py/nirdust.svg
    :target: https://badge.fury.io/py/nirdust
    :alt: PyPi Version

.. image:: https://travis-ci.com/Gaiana/nirdust.svg?branch=main
    :target: https://travis-ci.com/github/Gaiana/nirdust
    :alt: Build Status


.. image:: https://github.com/Gaiana/nirdust/actions/workflows/nirdust_ci.yml/badge.svg
    :target: https://github.com/Gaiana/nirdust/actions/workflows/nirdust_ci.yml
    :alt: Build Status

.. image:: https://readthedocs.org/projects/nirdust/badge/?version=latest
   :target: https://nirdust.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

.. image:: https://img.shields.io/badge/DiSoftCompCi-FAMAF-ffda00
   :target: https://github.com/leliel12/diseno_sci_sfw
   :alt: Curso doctoral FAMAF: Diseño de software para cómputo científico

.. image:: https://coveralls.io/repos/github/Gaiana/nirdust/badge.svg?branch=main
   :target: https://coveralls.io/github/Gaiana/nirdust?branch=main
   :alt: Coverage

.. image:: https://img.shields.io/badge/python-3.8+-blue.svg
   :target: https://badge.fury.io/py/uttrs
   :alt: Python 3.8+

.. image:: https://img.shields.io/badge/License-MIT-blue.svg
   :target: https://tldrlegal.com/license/mit-license
   :alt: License

**NIRDust** is a python package that uses K-band (2.2 micrometers) spectra to measure the temperature of the dust heated by an Active Galactic Nuclei (AGN) accretion disk.

Motivation
----------

K-band nuclear spectral continuum of Type 2 AGNs is often composed of two
components: the stellar population emission and a hot 800 - 1600 K dust component.
Via substraction of the stellar emission dust component fitting can be performed
to map its prescence and estimate its temperature.

Features
--------

The package uses the modeling features of astropy to fit the hot dust component
of a AGN K-band spectrum with black body functions. And provide a class with
methods for spectrum manipulation and normalized-blackbody-fitting. Because
NIRDust normalizes the spectra before fitting, is not necessary to
flux-calibrate spectra to use it.

NIRDust needs a minimum of two spectra to run: a nuclear one, where the dust
temperature will be determined, and an off-nuclear spectrum, where the emission
is considered to be purely stellar. The off-nuclear spectrum will be used by
NIRDust to subtract the stellar emission from the nuclear spectrum.

Footnote: the hot dust component may or may not be present in your type 2
nuclei, do not get disappointed if NIRDust finds nothing.


User Documentation
------------------

.. toctree::
   :maxdepth: 1

   installation
   api
   tutorials/NGC5128.ipynb
   licence


Requeriments
------------

You will need Python 3.8 or higher to run NIRDust.


Repository and Issues
---------------------

| To view NIRDust source code visit the repository: https://github.com/Gaiana/nirdust
| If you find any issues or bugs please let us know here: https://github.com/Gaiana/nirdust/issues


Authors
-------
| Gaia Gaspar (gaiagaspar@gmail.com)
| Jose Alacoria (josealacoria@gmail.com)
| Juan B. Cabral (jbc.develop@gmail.com)
| Martin Chalela (tinchochalela@gmail.com) 
