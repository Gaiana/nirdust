.. Nirduts documentation master file, created by
   sphinx-quickstart on Wed Feb 17 15:19:30 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Nirdust documentation!
========================

.. only:: html

    .. image:: _static/logo.png
        :align: center
        :scale: 50 %

.. image:: https://travis-ci.com/Gaiana/nirdust.svg?branch=main
    :target: https://travis-ci.com/github/Gaiana/nirdust
    :alt: Build Status

.. image:: https://img.shields.io/badge/DiSoftCompCi-FAMAF-ffda00
   :target: https://github.com/leliel12/diseno_sci_sfw
   :alt: Curso doctoral FAMAF: Diseño de software para cómputo científico

.. image:: https://readthedocs.org/projects/nirdust/badge/?version=latest
   :target: https://nirdust.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status
   
.. image:: https://img.shields.io/badge/License-MIT-blue.svg
   :target: https://tldrlegal.com/license/mit-license

**Nirdust** is a python package that uses K-band (2.2 micrometers) spectra to measure the temperature of the dust heated by an Active Galactic Nuclei (AGN) accretion disk.

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
Nirdust normalizes the spectra before fitting, is not necessary to 
flux-calibrate spectra to use it. 

Nirdust needs a minimum of two spectra to run: a nuclear one, where the dust 
temperature will be determined, and an off-nuclear spectrum, where the emission 
is considered to be purely stellar. The off-nuclear spectrum will be used by
Nirdust to subtract the stellar emission from the nuclear spectrum. 

Footnote: the hot dust component may or may not be present in your type 2 
nuclei, do not get disappointed if Nirdust finds nothing.


Requeriments
------------

You will need Python 3.8 or 3.9 to run Nirdust.

Installation
------------

Clone this repo and then inside the local directory execute

``` python
$ pip install -e .
```



| **Authors**
| Gaia Gaspar (E-mail: gaiagaspar@gmail.com)
| Jose Alacoria (E-mail: josealacoria@gmail.com)


Repository and Issues
---------------------

https://github.com/Gaiana/nirdust

--------------------------------------------------------


.. toctree::
   :maxdepth: 1
   :caption: Contents:

   api
   licence
   installation
   tutorial.ipynb

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
