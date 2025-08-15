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

We will then create a Python 3.12 environment named `BPMF_tuto`:


.. code-block:: console

    $ conda create -n BPMF_tuto python=3.12

We now need to activate this environment:

.. code-block:: console

    $ conda activate BPMF_tuto


C and CUDA-C Compilers
^^^^^^^^^^^^^^^^^^^^^^

In general, you can use :py:data:`conda` to locally install a C and a CUDA-C compiler.

.. code-block:: console

    $ conda install gcc
    $ conda install -c nvidia cuda-nvcc cuda-toolkit

OR, you may need `clang` if your machine uses one of the Apple Silicon chips. In which case, run:

.. code-block:: console

    $ conda install clang lld
    $ conda install -c nvidia cuda-nvcc cuda-toolkit


Standard Python packages
^^^^^^^^^^^^^^^^^^^^^^^^

Run the following command to install (almost) all the packages need for this tutorial:


.. code-block:: console

    $ conda install obspy numpy scipy pandas matplotlib h5py ipython jupyter cartopy colorcet tqdm

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

:py:data:`pykonal` is the package we will use for computing the P- and S-wave travel times (`https://github.com/malcolmw/pykonal <https://github.com/malcolmw/pykonal>`_). Install it with pip:

.. code-block:: console

    $ pip install pykonal

SeisBench
^^^^^^^^^

The best results with :py:data:`BPMF` are achieved when using deep-learning-based phase pickers. :py:data:`seisbench` (`https://github.com/seisbench/seisbench <https://github.com/seisbench/seisbench>`_) provides a convenient and comprehensive API to effortlessly use a number of well known models, such as PhaseNet. Install :py:data:`seisbench` with: 

.. code-block:: console

    $ pip install seisbench

This will also install the package :py:data:`torch`, including GPU support if your machine fits the requirements.

NonLinLoc
^^^^^^^^^

:py:data:`BPMF` provides an interface with the :py:data:`NLLoc` software (`http://alomax.free.fr/nlloc/ <http://alomax.free.fr/nlloc/>`_) for earthquake location. Get :py:data:`NLLoc` at `https://github.com/ut-beg-texnet/NonLinLoc <https://github.com/ut-beg-texnet/NonLinLoc>` and follow the installation instructions in the README.

.. code-block:: console

    $ git clone https://github.com/ut-beg-texnet/NonLinLoc.git
    $ cd NonLinLoc
    $ cd src
    $ rm CMakeCache.txt
    $ cmake .
    $ make

This will create a series of executable in the `bin` folder. Make this folder is added to your shell `PATH` variable. For example, add the following

.. code-block:: console
    
    $ export PATH={/pathtononlinloc/}NonLinLoc/bin/:$PATH

to your `.bashrc` file (Mac users might need to do the equivalent for zsh or csh instead of bash).

BPMF
^^^^

Finally, we need to install :py:data:`BPMF` to our new environment. We refer you to the :ref:`Installation` Section of the documentation.


Running the Tutorial
--------------------

The tutorial is made of a series of Ipython notebooks that are meant to be run from 0 to 10.


References
----------

Lomax, Anthony, Alberto Michelini, and Andrew Curtis. "Earthquake location, direct, global-search methods." In Encyclopedia of complexity and systems science, pp. 1-33. Springer, New York, NY, 2014.

White, Malcolm CA, Hongjian Fang, Nori Nakata, and Yehuda Ben‐Zion. "PyKonal: a Python package for solving the eikonal equation in spherical and Cartesian coordinates using the fast marching method." Seismological Research Letters 91, no. 4 (2020): 2378-2389.

Woollam, Jack, Jannes Münchmeyer, Frederik Tilmann, Andreas Rietbrock, Dietrich Lange, Thomas Bornstein, Tobias Diehl et al. "SeisBench—A toolbox for machine learning in seismology." Seismological Society of America 93, no. 3 (2022): 1695-1709.

Zhu, Weiqiang, and Gregory C. Beroza. "PhaseNet: a deep-neural-network-based seismic arrival-time picking method." Geophysical Journal International 216, no. 1 (2019): 261-273
