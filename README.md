# Seismic_BPMF
Complete framework for earthquake detection and location: Backprojection and matched-filtering (BPMF), packaged with methods for automatic picking, relocation and efficient waveform stacking. This package is built upon the codes used in Beauce et al. 2019, DOI: [10.1029/2019JB018110](https://doi.org/10.1029/2019JB018110), for which a Github repository was created: [https://github.com/ebeauce/earthquake_detection_EB_et_al_2019](https://github.com/ebeauce/earthquake_detection_EB_et_al_2019). The main conceptual difference from this previous repository is the addition of [PhaseNet](https://github.com/wayneweiqiang/PhaseNet) and [NLLoc](http://alomax.free.fr/nlloc/) in the workflow. Automatic picking + relocation replaces the ML classification of the previous repository.

## Examples
```python
    import BPMF

    T = BPMF.dataset.Template('template12', 'template_db', db_path='project_root')
    T.read_waveforms()
```

## Upcoming features
- Tutorials.
- Addition of [beamnetresponse](https://github.com/ebeauce/beamnetresponse),
  our Python package dedicated to beamforming/backpropagation.
- More docstrings and a documentation website.
- Data I/O will be based on
  [pyasdf](https://seismicdata.github.io/pyasdf/installation.html).


## Suggested Python environment
I suggest creating a new environment with `conda`.
```shell
  conda create --name BPMF python=3.8

  conda config --add channels conda-forge

  conda install compilers
  conda install numpy, scipy, h5py, pandas, matplotlib, obspy
```
and then install `fast_matched_filter` [code and instructions
here](https://github.com/beridel/fast_matched_filter). I also recommend
installing my customized version of PhaseNet
[https://github.com/ebeauce/PhaseNet](https://github.com/ebeauce/PhaseNet) that
has a wrapper module to call it from a python script.

## Reference
Please, cite:

Beaucé, E., Frank, W. B., Paul, A., Campillo, M., & van der Hilst, R. D.
(2019). Systematic detection of clustered seismicity beneath the Southwestern
Alps. Journal of Geophysical Research: Solid Earth, 124(11), 11531-11548.

If you use this package for your research. An updated publication is coming
soon!

## Contact
Questions? Contact me at:<br/>
ebeauce@ldeo.columbia.edu
