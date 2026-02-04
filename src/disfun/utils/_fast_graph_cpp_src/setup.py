"""Build script for fast_graph_cpp module."""

import os
import sys
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


class BuildExt(build_ext):
    """Custom build extension to add C++17 and optimization flags."""

    def build_extensions(self):
        # Add compiler flags
        for ext in self.extensions:
            ext.extra_compile_args = ['-std=c++17', '-O3', '-march=native', '-fPIC']
            ext.extra_link_args = ['-O3']
        super().build_extensions()


def get_pybind11_include():
    """Get pybind11 include path."""
    import pybind11
    return pybind11.get_include()


# Find LMDB
lmdb_include = '/usr/include'
lmdb_lib = '/usr/lib'

# Check common locations
for path in ['/usr/local/include', '/opt/homebrew/include']:
    if os.path.exists(os.path.join(path, 'lmdb.h')):
        lmdb_include = path
        break

ext_modules = [
    Extension(
        'fast_graph_cpp',
        sources=['fast_graph.cpp', 'bindings.cpp'],
        include_dirs=[
            '.',
            get_pybind11_include(),
            lmdb_include,
        ],
        libraries=['lmdb'],
        library_dirs=[lmdb_lib],
        language='c++',
    ),
]

setup(
    name='fast_graph_cpp',
    version='1.0.0',
    author='DisFun',
    description='Fast graph with CSR storage and C++ greedy solver',
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExt},
    python_requires='>=3.8',
    install_requires=['pybind11>=2.6'],
)
