Nirdust API
==============

Two input files in .FITS format must be provided in order to run Nirdust, the 
one where the dust component is expected to be found and the one that quantify 
the stellar population. 

**Nirdust** provides the following class methods for finding the dust component 
in a spectrum:

#- cut_edges():  cuts the spectrum given two limits.  
#- converttofrequency():  converts  the  spectral  axis  of  thespectrum to 
unities of Hz. 
#- normalize():  normalize the intensity axis of the spectrumby dividing it by 
its numerical mean.
#- fitblackbody():  uses the normalizedblackbody model tofit data

The sp_correction() function must be applied after both imput spectra have been
cuted and converted to frqeuency space.

See the turorial for details.



-------------------------------------------------------------------------------------

.. toctree::
   :maxdepth: 2

Module ``nirdust``
----------------------
.. automodule:: nirdust
   :members:
   :show-inheritance:
   :member-order: bysource
