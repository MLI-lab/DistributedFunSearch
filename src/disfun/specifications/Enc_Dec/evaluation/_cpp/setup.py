"""Setup script for building the C++ bruteforce decoder extension."""

from setuptools import setup, Extension
import pybind11
import sys
import os

# Get pybind11 include path
pybind11_include = pybind11.get_include()

# Compiler flags
extra_compile_args = ['-O3', '-std=c++17', '-fPIC', '-fopenmp']
extra_link_args = ['-fopenmp']

# macOS specific
if sys.platform == 'darwin':
    extra_compile_args = ['-O3', '-std=c++17', '-fPIC']
    if os.path.exists('/opt/homebrew/opt/libomp'):
        extra_compile_args += ['-Xpreprocessor', '-fopenmp', '-I/opt/homebrew/opt/libomp/include']
        extra_link_args = ['-L/opt/homebrew/opt/libomp/lib', '-lomp']
    else:
        extra_link_args = []

# Define the extension module using base Extension class
ext_modules = [
    Extension(
        "bruteforce_cpp",
        ["bruteforce.cpp"],
        include_dirs=[pybind11_include],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language='c++',
    ),
]

setup(
    name="bruteforce_cpp",
    version="1.0.0",
    ext_modules=ext_modules,
)
