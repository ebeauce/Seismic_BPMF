Introduction
============

Description
-----------
Fully automated workflow for earthquake detection and location with the backprojection and matched filtering methods.

:py:data:`BPMF` is available at `https://github.com/ebeauce/Seismic_BPMF <https://github.com/ebeauce/Seismic_BPMF>`_ and can be downloaded with:

.. code-block:: console

    $ git clone https://github.com/ebeauce/Seismic_BPMF.git

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

