Introduction
============

Description
-----------
Fully automated workflow for earthquake detection and location with the backprojection and matched filtering methods.

:py:data:`BPMF` is available at `https://github.com/ebeauce/Seismic_BPMF <https://github.com/ebeauce/Seismic_BPMF>`_ and can be downloaded with:

.. code-block:: console

    $ git clone https://github.com/ebeauce/Seismic_BPMF.git

Reference
---------

Please, refer one of the following articles if you use BPMF for your research:

    
  - Eric Beaucé, William B. Frank, Léonard Seydoux, Piero Poli, Nathan Groebner, Robert D. van der Hilst, and Michel Campillo. BPMF: A Backprojection and Matched‐Filtering Workflow for Automated Earthquake Detection and Location. *Seismological Research Letters*. (2024): DOI: `https://doi.org/10.1785/0220230230 <https://doi.org/10.1785/0220230230>`_.
  - Eric Beaucé, Robert D. van der Hilst, Michel Campillo. Microseismic Constraints on the Mechanical State of the North Anatolian Fault Zone 13 Years After the 1999 M7.4 Izmit Earthquake. *Journal of Geophysical Research: Solid Earth*. (2022) DOI: `https://doi.org/10.1029/2022JB024416 <https://doi.org/10.1029/2022JB024416>`_.
  - Eric Beaucé, William B. Frank, Anne Paul, Michel Campillo and Robert D. van der Hilst. Systematic Detection of Clustered Seismicity Beneath the Southwestern Alps. *Journal of Geophysical Research: Solid Earth*. (2019) DOI: `https://doi.org/10.1029/2019JB018110 <https://doi.org/10.1029/2019JB018110>`_.

Installation
-------------

You may need to edit the Makefile according to your OS (instructions in the Makefile's comments).

From source
^^^^^^^^^^^
A simple make + whichever implementation does the trick. Possible make commands are:

.. code-block:: console

    $ make python_cpu
    $ make python_gpu
    $ pip install .

Using pip (recommended)
^^^^^^^^^^^^^^^^^^^^^^^

Installation as a Python module is possible via pip (which supports clean uninstalling):

.. code-block:: console

    $ python setup.py build_ext
    $ pip install .

or simply:

.. code-block:: console

    $ pip install git+https://github.com/ebeauce/Seismic_BPMF
