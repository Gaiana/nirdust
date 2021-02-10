[![Build Status](https://travis-ci.com/Gaiana/nirdust.svg?branch=main)](https://travis-ci.com/Gaiana/nirdust)

# Nirdust: Near Infrared Dust Finder

Nirdust is a python package that uses K-band (2.2 micrometers) spectra to 
measure the temperature of the dust heated by an Active Galactic Nuclei (AGN) 
accretion disk. 


## Motivation:

K-band nuclear spectral continuum of AGNs is often composed of different 
components that can be separated:

.) The stellar component, consisting in the integrated flux of the stellar
population of the galaxy.

.) The hot dust component, consisting in a sum of blackbody shape fluxes arising 
in dust of several temperatures. The temperature of the dust emitting in the K 
band ranges from 800 to 1800 K. This component is not always present.

.) The acretion disk emission, consisting of a power law shaped spectrum. (Only 
present in type 1 AGNs)

This means that, for type 2 AGNs, dust component fitting can be performed after 
the stellar population is substracted. Nirdust works under the hypothesis that
in the Near Infrared (NIR) the stellar popullation is rather old and hence 
relatively homogeneous, this allows to asume that the stellar popullation at a
prudential distance from the nucleus is equal to the nuclear one. The meaning 
of 'prudential distance' will vary depending on the galaxy and on the spatial
resolution of the spectra, and should be determined carefully by the user.


Currently Nirdust is being developed to work with type 2 AGNs only. An
extension to type 1 AGNs will be hopefully offered in the future.


## Features

The package uses the modeling features of astropy to fit the hot dust component 
of a AGN K-band spectrum with black body functions. And provide a class with
methods for spectrum manipulation and normalized-blackbody-fitting. Because 
Nirdust normalizes the spectra before fitting, is not necessary to 
flux-calibrate spectra to use it. This is important in the near-infrared since 
the fast sky emission variations introduces high uncertainties in such 
flux-calibration. In fact, Nirdust will asume that your spectra are in arbitrary
units.

Nirdust needs a minimum of two spectra to run: a nuclear one, where the dust 
temperature will be determined, and an off-nuclear spectrum, where the emission 
is considered to be purely stellar. The off-nuclear spectrum will be used by
Nirdust to subtract the stellar emission from the nuclear spectrum. 




Footnote: the hot dust component may or may not be present in your type 2 
nuclei, do not get disappointed if Nirdust finds nothing.

Authors: Gaia Gaspar, José Alacoria.



