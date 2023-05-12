Pre-requisites
==============

The goal of this tutorial is to learm how to implement a full earthquake detection and location workflow with :py:data:`BPMF`.

Environment
-----------

Base
^^^^

Creating the virtual environment as described here is *essential* for running the tutorial successfully. We will use the python package manager `Anaconda`, or rather its ligther version `Miniconda`. Follow the instructions here `https://docs.conda.io/en/latest/miniconda.html <https://docs.conda.io/en/latest/miniconda.html>`_ to install `Miniconda`.

Once `Anaconda` or `Miniconda` is installed, you can use the :py:data:`conda` commands. We will first make sure that :py:data:`conda` uses packages from :py:data:`conda-forge` when necessary:


.. code-block:: console

    $ conda config --add channels conda-forge

We will then create a Python 3.10 environment named `BPMF_tuto`:


.. code-block:: console

    $ conda create -n BPMF_tuto python=3.10

We now need to activate this environment:

.. code-block:: console

    $ conda activate BPMF_tuto

In general, you can use :py:data:`conda` to locally install a C and a CUDA-C compiler.

.. code-block:: console

    $ conda install gcc
    $ conda install -c nvidia cuda-nvcc cuda-toolkit


Run the following command to install (almost) all the packages need for this tutorial:


.. code-block:: console

    $ conda install obspy numpy scipy pandas matplotlib h5py ipython jupyter cartopy colorcet

Beampower
^^^^^^^^^

:py:data:`beampower` is our package with C and CUDA-C routines for
backprojection wrapped in Python code. You can download and install
:py:data:`beampower` with:

.. code-block:: console

   $ pip install git+https://github.com/ebeauce/beampower

Note: You might have to modify the `Makefile` depending on your compilers. See
`https://ebeauce.github.io/beampower/introduction.html#installation
<https://ebeauce.github.io/beampower/introduction.html#installation>`_ for more
information.

Fast Matched Filter
^^^^^^^^^^^^^^^^^^^

:py:data:`fastmatchedfilter` is our package with C and CUDA-C routines for
template matching wrappedin Python code. You can download and install
:py:data:`fastmatchedfilter`:

.. code-block:: console

   $ pip install git+https://github.com/beridel/fast_matched_filter

Note: You might have to modify the `Makefile` depending on your compilers. See `https://ebeauce.github.io/FMF_documentation/introduction.html#installation
<https://ebeauce.github.io/FMF_documentation/introduction.html#installation>`_
for more information.


PyKonal
^^^^^^^

Then, download Pykonal from `https://github.com/malcolmw/pykonal <https://github.com/malcolmw/pykonal>`_. Pykonal is the package we will use for computing the P- and S-wave travel times. Once downloaded and unpacked, go to Pykonal's root folder and run:

.. code-block:: console

    $ pip install .

PhaseNet
^^^^^^^^

Several important features of :py:data:`BPMF` relies on the deep neural network phase picker PhaseNet. In order to use PhaseNet, you have to install :py:data:`phasenet` from E.B.'s Github (modified version with wrapper functions to use PhaseNet from within a python script) at: `https://github.com/ebeauce/PhaseNet <https://github.com/ebeauce/PhaseNet>`_. Go to PhaseNet's root folder and run: 

.. code-block:: console

    $ pip install .

This should download the package :py:data:`tensorflow` and may take some time.

NonLinLoc
^^^^^^^^^

To benefit from the best location routines, you need to install the :py:data:`NLLoc` software (`http://alomax.free.fr/nlloc/ <http://alomax.free.fr/nlloc/>`_). You can download :py:data:`NLLoc` at `http://alomax.free.fr/nlloc/soft7.00/tar/NLL7.00_src.tgz <http://alomax.free.fr/nlloc/soft7.00/tar/NLL7.00_src.tgz>`_. For Unix and Mac users, I recommend doing something along the lines (modify as necessary):

.. code-block:: console

    $ mkdir ${HOME}/NLLoc
    $ cd ${HOME}/NLLoc
    $ wget http://alomax.free.fr/nlloc/soft7.00/tar/NLL7.00_src.tgz
    $ tar -xvf archive_name

And then, create a `bin` folder where `NLLoc`'s binary executable files will be stored after compilation.

.. code-block:: console
    
    $ mkdir ${HOME}/bin
    $ export MYBIN=${HOME}/bin/
    $ export PATH=${MYBIN}:$PATH

and add the last two lines to your `.bashrc` file (Mac users might need to do the equivalent for zsh or csh instead of bash). After that, you can run the `Makefile` from `${HOME}/NLLoc`.

.. code-block:: console

    $ make


Finally, we need to install :py:data:`BPMF` to our new environment. We refer you to the :ref:`Installation` Section of the documentation.


Running the Tutorial
--------------------

The tutorial is made of a series of Ipython notebooks that are meant to be run from 0 to 10.


Reference
---------

Zhu, Weiqiang, and Gregory C. Beroza. "PhaseNet: a deep-neural-network-based seismic arrival-time picking method." Geophysical Journal International 216, no. 1 (2019): 261-273.
