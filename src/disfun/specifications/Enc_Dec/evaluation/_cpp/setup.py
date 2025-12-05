"""Setup script for building the C++ bruteforce decoder extension."""

from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
import sys
import os

# Determine compiler flags based on platform
# Use C++14 for better compatibility
extra_compile_args = ['-O3', '-std=c++14', '-D_GNU_SOURCE']
extra_link_args = []

# Try to enable OpenMP
if sys.platform == 'darwin':
    # macOS with Homebrew libomp
    if os.path.exists('/opt/homebrew/opt/libomp'):
        extra_compile_args += ['-Xpreprocessor', '-fopenmp']
        extra_compile_args += ['-I/opt/homebrew/opt/libomp/include']
        extra_link_args += ['-L/opt/homebrew/opt/libomp/lib', '-lomp']
    elif os.path.exists('/usr/local/opt/libomp'):
        extra_compile_args += ['-Xpreprocessor', '-fopenmp']
        extra_compile_args += ['-I/usr/local/opt/libomp/include']
        extra_link_args += ['-L/usr/local/opt/libomp/lib', '-lomp']
else:
    # Linux - OpenMP usually available with gcc
    extra_compile_args += ['-fopenmp']
    extra_link_args += ['-fopenmp']

# Define the extension module
ext_modules = [
    Pybind11Extension(
        "bruteforce_cpp",
        ["bruteforce.cpp"],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        define_macros=[('VERSION_INFO', '"1.0.0"')],
    ),
]

setup(
    name="bruteforce_cpp",
    version="1.0.0",
    author="DistributedFunSearch",
    description="C++ extension for brute-force decoder validation",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.8",
)
