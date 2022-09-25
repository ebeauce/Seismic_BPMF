# Seismic_BPMF
Complete framework for earthquake detection and location: Backprojection and matched-filtering (BPMF), packaged with methods for automatic picking, relocation and efficient waveform stacking. This project uses the deep neural network phase picker [PhaseNet](https://github.com/wayneweiqiang/PhaseNet) and the earthquake locator [NLLoc](http://alomax.free.fr/nlloc/). The backprojection earthquake detector uses our package [beamnetresponse](https://github.com/ebeauce/beamnetresponse) and the template matching earthquake detector uses our package [fast_matched_filter](https://github.com/beridel/fast_matched_filter).  


BPMF v2.0.0 is now out but is still under development (v2.0.0.dev). It is ready
to run the full detection and location workflow (tutorial coming soon!), but I
am still implementing features, correcting bugs, improving docstrings, etc.


## New features in v2.0.0
- Integrated and easy use of PhaseNet and NLLoc.
- New classes to represent events and templates with methods to conveniently
  pick P-/S-waves and locate them.


## Suggested Python environment
I suggest creating a new environment with `conda`.
```shell
  conda create --name BPMF python=3.10

  conda config --add channels conda-forge

  conda install numpy, scipy, h5py, pandas, matplotlib, obspy
```
and then install `beamnetresponse` ([https://github.com/ebeauce/beamnetresponse](https://github.com/ebeauce/beamnetresponse)) and `fast_matched_filter` ([code and instructions
here](https://github.com/beridel/fast_matched_filter)). I also recommend
installing my customized version of PhaseNet
([https://github.com/ebeauce/PhaseNet](https://github.com/ebeauce/PhaseNet)) that
has a wrapper module to simplify its use from a python script.

Before I release the documentation and tutorials, I suggest you create your
python environment by following the instructions at
[https://ebeauce.github.io/beamnetresponse/tutorial/general.html](https://ebeauce.github.io/beamnetresponse/tutorial/general.html).

## Installation

Download or clone the repository. Go to the root folder, activate your virtual
environment, and execute the following command lines:
```shell
    python setup.py build_ext
    pip install .
```
The first line, `python setup.py build_ext`, executes the Makefile and compiles the C and CUDA-C librairies. If you don't have an Nvidia GPU and/or the nvcc compiler, you will see a warning message (and every time you will load the BPMF librairy). You can still use BPMF on CPUs. 


Note: Our paper Beaucé et al., 2022 (see References below) was prepared with
BPMF v1.0.1, than you can find at
[https://github.com/ebeauce/Seismic_BPMF/releases/tag/v1.0.1](https://github.com/ebeauce/Seismic_BPMF/releases/tag/v1.0.1).

## Reference
Please, cite:

Beaucé, E., Frank, W. B., Paul, A., Campillo, M., & van der Hilst, R. D.
(2019). Systematic detection of clustered seismicity beneath the Southwestern
Alps. Journal of Geophysical Research: Solid Earth, 124(11), 11531-11548.

and/or

Beaucé, E., van der Hilst, R. D., & Campillo M. (2022). Microseismic Constraints
on the Mechanical State of the North Anatolian Fault Zone Thirteen Years after
the 1999 M7.4 Izmit Earthquake. Journal of Geophysical Research: Solid Earth.
DOI:
[https://doi.org/10.1029/2022JB024416](https://doi.org/10.1029/2022JB024416).

If you use this package for your research.

## To do:
- [ ] PhaseNet and NLLoc independent relocation method.
- [ ] Convert `availability` and `source_receiver_dist` to properties.

## Contact
Questions? Contact me at:<br/>
ebeauce@ldeo.columbia.edu
