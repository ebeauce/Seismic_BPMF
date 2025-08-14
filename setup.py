from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext as build_ext_original
from subprocess import call


class BPMFExtension(Extension):
    def __init__(self, name):
        super().__init__(name=name, sources=[])

# Define a new build command
class BPMFBuild(build_ext_original):
    def run(self):
        cpu_make = ['make', 'python_cpu']

        cpu_built = call(cpu_make) == 0

        if not cpu_built:
            raise OSError("Could not compile C libraries")

setup(
    packages=['BPMF'],
    include_package_data=True,
    zip_safe=False,
    cmdclass={'build_ext': BPMFBuild},
    ext_modules=[
        BPMFExtension('BPMF.lib.libc'),
                 ]
)
