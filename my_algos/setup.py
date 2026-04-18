"""
Build the HNSW pybind11 extension.

Usage
-----
    pip install pybind11 numpy
    python setup.py build_ext --inplace

This produces  hnsw_index.cpython-<ver>-<platform>.so  in the current directory.
You can then do:  import hnsw_index
"""

from setuptools import setup, Extension
import pybind11
import sys

# MSVC (Windows) uses different flags from GCC/Clang
if sys.platform == "win32":
    extra_compile_args = ["/std:c++17", "/O2", "/EHsc"]
    extra_link_args    = []
else:
    extra_compile_args = ["-std=c++17", "-O3", "-march=native", "-ffast-math"]
    extra_link_args    = []
    if sys.platform == "darwin":
        extra_compile_args += ["-stdlib=libc++"]
        extra_link_args    += ["-stdlib=libc++"]

ext = Extension(
    name="hnsw_index",
    sources=["hnsw_bind.cpp"],
    include_dirs=[pybind11.get_include()],
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
    language="c++",
)

setup(
    name="hnsw_index",
    version="1.0.0",
    description="HNSW approximate nearest-neighbour index (pybind11)",
    ext_modules=[ext],
)