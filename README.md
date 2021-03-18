[![Build Status](https://travis-ci.com/Gaiana/nirdust.svg?branch=main)](https://travis-ci.com/Gaiana/nirdust)
[![https://github.com/leliel12/diseno_sci_sfw](https://img.shields.io/badge/DiSoftCompCi-FAMAF-ffda00)](https://github.com/leliel12/diseno_sci_sfw)
[![Documentation Status](https://readthedocs.org/projects/nirdust/badge/?version=latest)](https://nirdust.readthedocs.io/en/latest/?badge=latest)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-370/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

# Nirdust: Near Infrared Dust Finder

Nirdust is a python package that uses K-band (2.2 micrometers) spectra to 
measure the temperature of the dust heated by an Active Galactic Nuclei (AGN) 
accretion disk. 


## Motivation:

K-band nuclear spectral continuum of Type 2 AGNs is often composed of two 
components: the stellar population emission and a hot 800 - 1600 K dust component.
Via subtraction of the stellar emission dust component fitting can be performed
to map its presence and estimate its temperature.


## Features

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


## Requeriments

You will need Python 3.8 or 3.9 to run Nirdust.

## Installation

You can install the least stable version of Nirdust from pip:


``` python
$ pip install nirdust
```

Or, for the develovepment instalation clone this repository and then inside the local directory execute

``` python
$ pip install -e .
```

Authors: Gaia Gaspar, José Alacoria.



