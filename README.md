# Seismic_BPMF
Complete framework for earthquake detection and location: Backprojection and matched-filtering (BPMF), packaged with methods for automatic picking, relocation and efficient waveform stacking. This project uses the deep neural network phase picker [PhaseNet](https://github.com/wayneweiqiang/PhaseNet) and the earthquake locator [NLLoc](http://alomax.free.fr/nlloc/). The backprojection earthquake detector uses our package [beamnetresponse](https://github.com/ebeauce/beamnetresponse) and the template matching earthquake detector uses our package [fast_matched_filter](https://github.com/beridel/fast_matched_filter).  

The last stable release is v1.0.1, but v2.0.0 is coming soon with polished modules, documentations, and a set of tutorials so that you can start your own earthquake detection and location project.  


## Examples
```python
    import BPMF

    T = BPMF.dataset.Template('template12', 'template_db', db_path='project_root')
    T.read_waveforms()
```

## Upcoming features (in v2.0.0)
- Tutorials.
- More docstrings and a documentation website.
- Data I/O will be based on
  [pyasdf](https://seismicdata.github.io/pyasdf/installation.html).
- Integrated and easy use of PhaseNet and NLLoc.


## Suggested Python environment
I suggest creating a new environment with `conda`.
```shell
  conda create --name BPMF python=3.8

  conda config --add channels conda-forge

  conda install compilers
  conda install numpy, scipy, h5py, pandas, matplotlib, obspy
```
and then install `beamnetresponse` ([https://github.com/ebeauce/beamnetresponse](https://github.com/ebeauce/beamnetresponse)) and `fast_matched_filter` ([code and instructions
here](https://github.com/beridel/fast_matched_filter)). I also recommend
installing my customized version of PhaseNet
([https://github.com/ebeauce/PhaseNet](https://github.com/ebeauce/PhaseNet)) that
has a wrapper module to simplify its use from a python script. Note: installing
the `compilers` package allows you to have recent C/Fortran compilers locally
installed in the BPMF environment.

## Installation

Download the v1.0.1 source code at [https://github.com/ebeauce/Seismic_BPMF/releases/tag/v1.0.1](https://github.com/ebeauce/Seismic_BPMF/releases/tag/v1.0.1). Unzip or untar the content. Open a terminal from the root folder where Makefile and setup.py are located. Activate your virtual environment if using one.
```shell
    python setup.py build_ext
    pip install .
```
The first line, `python setup.py build_ext`, executes the Makefile and compiles the C and CUDA-C librairies. If you don't have an Nvidia GPU and/or the nvcc compiler, you will see a warning message (and every time you will load the BPMF librairy). You can still use BPMF on CPUs. 

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
