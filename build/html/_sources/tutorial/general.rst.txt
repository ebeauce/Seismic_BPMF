Pre-requisites
==============

The goal of this tutorial is to learm how to implement a full earthquake detection and location workflow with :py:data:`BPMF`.
Environment
-----------

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

    $ conda install obspy numpy scipy pandas matplotlib h5py ipython jupyter cartopy


Then, download Pykonal from `https://github.com/malcolmw/pykonal <https://github.com/malcolmw/pykonal>`_. Pykonal is the package we will use for computing the P- and S-wave travel times. Once downloaded and unpacked, go to Pykonal's root folder and run:

.. code-block:: console

    $ pip install .

Several important features of :py:data:`BPMF` relies on the deep neural network phase picker PhaseNet. In order to use PhaseNet, you have to install :py:data:`phasenet` from E.B.'s Github (modified version with wrapper functions to use PhaseNet from within a python script) at: `https://github.com/ebeauce/PhaseNet <https://github.com/ebeauce/PhaseNet>`_. Go to PhaseNet's root folder and run: 

.. code-block:: console

    $ pip install .

This should download the package :py:data:`tensorflow` and may take some time.


Finally, we need to install :py:data:`BPMF` to our new environment. We refer you to the :ref:`Installation` Section of the documentation.


Running the Tutorial
--------------------

The tutorial is made of a series of Ipython notebooks that are meant to be run from 0 to 3.


Reference
---------

Zhu, Weiqiang, and Gregory C. Beroza. "PhaseNet: a deep-neural-network-based seismic arrival-time picking method." Geophysical Journal International 216, no. 1 (2019): 261-273.
