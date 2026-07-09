work in progress. More details and examples will be added here.

Gammapy-mwl 
=======
A Python package to enhance the capacities of gammapy to support more data format and telescopes in particular from X-ray observatories.
This adds the support to read X-ray spectral data and provides an interface to link with X-ray spectral Xspec models via sherpa (optional dependency).


## Installation and Set-up

These instructions assume that you have previously installed a version of gammapy (>= 2.1).
See installation instructions [here](https://docs.gammapy.org/2.1/getting-started/index.html#recommended-setup).

## Install the package 
```
pip install gammapy-mwl 
```

Additionally, if you wish to add the support for X-ray models, you will need to install sherpa separately. If you want sherpa Xspec models, you will need to install `xspec_models`.




Citing
+++++++++++++++++++++++++++++++++++++++++++++

A software description is provided in the following publication: TBD


If you use gammapyXray for work/research presented in a publication (whether directly, or as a dependency to another package), we ask that you please cite it using the following links

???

We encourage you to also include citations to the gammapy paper in the main text
wherever appropriate, using the recommended BibTeX entry shown in the [gammapy docs](https://gammapy.org/acknowledging.html).


Licence
+++++++
This folder is licensed under a 3-clause BSD style license - see the
`LICENSE.rst <https://github.com/gammapy/gammapy/blob/master/LICENSE.rst>`_ file.


--------OLDER NOTES FOR LATER--------------

# gammapy-mwl
A repository for tools to convert various MWL data to gammapy

## Repositories with MWL tools using gammapy

###  A MWL workflow from optical to GeV 
- from @mireianievas : https://github.com/mireianievas/gammapy_mwl_workflow

### X-ray OGIP data manipulation 

- Latest fork of gammapyXray : https://github.com/mireianievas/gammapy-ogip-spectra

## Test data
- HEACIT curated list of X-ray test data : https://github.com/HEACIT/curated-test-data

