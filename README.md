# NIRDust: Near Infrared Dust Finder

<p align="center">
<img src="https://github.com/Gaiana/nirdust/blob/main/docs/source/_static/logo.png?raw=true" alt="logo" height="200"/>
</p>

[![PyPi Version](https://badge.fury.io/py/nirdust.svg)](https://badge.fury.io/py/nirdust)
[![Nirdust](https://github.com/Gaiana/nirdust/actions/workflows/nirdust_ci.yml/badge.svg)](https://github.com/Gaiana/nirdust/actions/workflows/nirdust_ci.yml)
[![https://github.com/leliel12/diseno_sci_sfw](https://img.shields.io/badge/DiSoftCompCi-FAMAF-ffda00)](https://github.com/leliel12/diseno_sci_sfw)
[![Documentation Status](https://readthedocs.org/projects/nirdust/badge/?version=latest)](https://nirdust.readthedocs.io/en/latest/?badge=latest)
[![Coverage Status](https://coveralls.io/repos/github/Gaiana/nirdust/badge.svg?branch=main)](https://coveralls.io/github/Gaiana/nirdust?branch=main)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-370/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)


NIRDust is a python package that uses K-band (2.2 micrometers) spectra to 
measure the temperature of the dust heated by an Active Galactic Nuclei (AGN) 
accretion disk. 


## Motivation:

K-band nuclear/circumnuclear spectral continuum of Type 2 AGNs is often composed of two 
components: the stellar population emission and a hot 700 - 1600 K dust component.
If a Reference Spectrum is used to model the stellar emission, the dust component 
can be fitted by a blackbody function in order to obtain its temperature.


## Features

The package provides several functionalities to pre-process spectra and fit the
hot dust component of a AGN K-band spectrum with black body functions. 

NIRDust needs a minimum of two spectra to run: a target one, where the dust 
temperature will be determined, and a reference spectrum, where the emission 
is considered to be purely stellar. The reference spectrum spectrum will be used by
NIRDust to model the stellar emission from the target spectrum. 


Footnote: the hot dust component may or may not be present in your type 2 
nuclei, do not get disappointed if NIRDust finds nothing.


## Requeriments

You will need Python 3.8 or higher to run NIRDust.

## Installation

You can install the least stable version of NIRDust from pip:


``` python
$ pip install nirdust
```

Or, for the develovepment instalation clone this repository and then inside the local directory execute

``` python
$ pip install -e .
```

Authors: Gaia Gaspar, José Alacoria, Juan B. Cabral & Martin Chalela




