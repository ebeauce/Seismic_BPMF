# Seismic_BPMF
Complete framework for earthquake detection and location: Backprojection and matched-filtering (BPMF), coming with methods for automatic picking, relocation and efficient waveform stacking. This package is built upon the codes used in Beauce et al. 2019, DOI: [10.1029/2019JB018110](https://doi.org/10.1029/2019JB018110), for which a Github repository was created: [https://github.com/ebeauce/earthquake_detection_EB_et_al_2019](https://github.com/ebeauce/earthquake_detection_EB_et_al_2019). The main update from this previous repository is the addition of [PhaseNet](https://github.com/wayneweiqiang/PhaseNet) and [NLLoc](http://alomax.free.fr/nlloc/) in the workflow.

This repository is not ready for wide public usage as I don't have time to polish all the modules.

#List of things to implement:
- Clean docstrings for each function, and an online documentation.
- Add PhaseNet and NLLoc wrappers.
- Data downloading/preprocessing ObspyDMT routines in a separate sub-package.
