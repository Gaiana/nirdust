# Nirdust: Near Infrared Dust Finder

Nirdust is a python package that uses K-band (2.2 micrometers) spectra to 
measure the temperature of the dust heated by an Active Galactic Nuclei (AGN) 
accretion disk. 

Currently Nirdust is being developed to work with type 2 AGNs only. An
extension to type 1 AGNs will be hopefully offered in the future.

The package uses the modeling features of astropy to fit the hot dust component 
of a AGN K-band spectrum with black body functions.

Nirdust needs a minimum of two spectra to run: a nuclear one, where the dust 
temperature will be determined, and an off-nuclear spectrum, where the emission 
is considered to be purely stellar. The off-nuclear spectrum will be used by
Nirdust to subtract the stellar emission from the nuclear spectrum. 

It is possible to input more than one nuclear spectrum. In the case of various 
spectra for different radii of the same target, Nirdust will provide a radial 
temperature distribution. In the case of various spectra for different targets
Nirdust will provide the distribution of dust temperatures for the sample.
The expected temperatures for AGN heated dust are in the range 800-2000 K.

Footnote: the hot dust component may or may not be present in your type 2 
nuclei, do not get disappointed if Nirdust finds nothing. 



