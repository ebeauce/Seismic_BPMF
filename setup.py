"""
Minimal setup file for the BPMF library for Python packaging.
:copyright:
    Eric Beauce
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.en.html)
"""

from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext as build_ext_original
from subprocess import call


class BPMFExtension(Extension):
    def __init__(self, name):
        # Don't run the default setup-tools build commands, use the custom one
        Extension.__init__(self, name=name, sources=[])


# Define a new build command
class BPMFBuild(build_ext_original):
    def run(self):
        # Build the Python libraries via Makefile
        cpu_make = ['make', 'python_cpu']
        #gpu_make = ['make', 'python_gpu']

        #gpu_built = False
        cpu_built = False

        ret = call(cpu_make)
        if ret == 0:
            cpu_built = True
        #ret = call(gpu_make)
        #if ret == 0:
        #    gpu_built = True
        #if gpu_built is False:
        #    print("Could not build GPU code")
        if cpu_built is False:
            raise OSError("Could not build cpu code")

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


setup(
    name="BPMF",
    version="2.0.0.beta1",
    author="Eric Beaucé",
    author_email="ebeauce@ldeo.columbia.edu",
    description="Package for automated earthquake detection and location",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ebeauce/Seismic_BPMF",
    project_urls={
        "Bug Tracker": "https://github.com/ebeauce/Seismic_BPMF/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GPL License",
        "Operating System :: OS Independent",
    ],
    license="GPL",
    packages=["BPMF"],
    install_requires=[
        "beampower",
        "FastMatchedFilter",
        "h5py",
        "matplotlib",
        "numpy",
        "obspy",
        "pandas",
        "scipy",
        ],
    python_requires=">=3.6",
    zip_safe=False,
    cmdclass={
        "build_ext": BPMFBuild},
    include_package_data=True,
    ext_modules=[BPMFExtension("BPMF.lib.libc")]
)
