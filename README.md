# Seismic_BPMF

Complete framework for earthquake detection and location with GPU-accelerated processing.

Backprojection and matched-filtering (BPMF) is a two-step earthquake detection workflow with 1) backprojection to build an initial earthquake catalog  and 2) matched-filtering (template matching) to densify the catalog, that is, lower the magnitude of completeness of the catalog and the shortest resolved inter-event time. BPMF offers wrapper functions for the deep neural network phase picker [PhaseNet](https://github.com/wayneweiqiang/PhaseNet) (handled with [seisbench](https://github.com/seisbench/seisbench)) and the earthquake locator [NLLoc](http://alomax.free.fr/nlloc/) as part of the automated location workflow. It also supports fully BPMF-native location with classic backprojection.  BPMF leverages the low-level C and CUDA-C programming languages for the efficient processing of large data volumes. The core routines for backprojection and matched-filtering are provided in our two packages [beampower](https://github.com/ebeauce/beampower) and [fast_matched_filter](https://github.com/beridel/fast_matched_filter), respectively.  

`BPMF` v2.0.0-beta is now out! Checkout the online tutorial at [https://ebeauce.github.io/Seismic_BPMF/tutorial](https://ebeauce.github.io/Seismic_BPMF/tutorial) to learn how to use our fully automated workflow and build your own earthquake catalog.

## Documentation

Check out the online documentation at [https://ebeauce.github.io/Seismic_BPMF/index.html](https://ebeauce.github.io/Seismic_BPMF/index.html).

## Installation

You may want to follow the instructions from the tutorial at
[https://ebeauce.github.io/Seismic_BPMF/tutorial/general.html](https://ebeauce.github.io/Seismic_BPMF/tutorial/general.html).

Download or clone the repository. Go to the root folder, activate your virtual
environment, and execute the following command lines:
```shell
    python setup.py build_ext
    pip install .
```
The first line, `python setup.py build_ext`, executes the Makefile and compiles the C and CUDA-C librairies. If you don't have an Nvidia GPU and/or the nvcc compiler, you will see a warning message (and every time you will load the BPMF librairy). You can still use BPMF on CPUs. 


Details on how to set up a working environment at [https://ebeauce.github.io/Seismic_BPMF/tutorial/general.html](https://ebeauce.github.io/Seismic_BPMF/tutorial/general.html).


## Reference
Please, if you use this package for your research, cite:

Beaucé, Eric and Frank, William B. and Seydoux, Léonard and Poli, Piero and Groebner, Nathan
and van der Hilst, Robert D. and Campillo, Michel (2023). BPMF: A Backprojection and Matched‐Filtering Workflow for Automated Earthquake Detection and Location. *Seismological Research Letters*. DOI: [https://doi.org/10.1785/0220230230](https://doi.org/10.1785/0220230230).

The methodology is detailed in:

Beaucé, E., Frank, W. B., Paul, A., Campillo, M., & van der Hilst, R. D.
(2019). Systematic detection of clustered seismicity beneath the Southwestern
Alps. Journal of Geophysical Research: Solid Earth, 124(11), 11531-11548.

and

Beaucé, E., van der Hilst, R. D., & Campillo M. (2022). Microseismic Constraints
on the Mechanical State of the North Anatolian Fault Zone Thirteen Years after
the 1999 M7.4 Izmit Earthquake. Journal of Geophysical Research: Solid Earth.
DOI:
[https://doi.org/10.1029/2022JB024416](https://doi.org/10.1029/2022JB024416).


## Contact
Questions? Contact me at:<br/>
ebeauce@ldeo.columbia.edu
