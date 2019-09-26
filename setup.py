#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys

from setuptools import setup
from setuptools.command.test import test
from distutils.core import Extension
try:
    import numpy
    from Cython.Build import cythonize
    CYTHON = True
except Exception:
    CYTHON = False


class PyTest(test):
    user_options = [("pytest-args=", "a", "Arguments to pass into py.test")]

    def initialize_options(self):
        test.initialize_options(self)
        self.pytest_args = ""

    def run_tests(self):
        import pytest
        import shlex

        errno = pytest.main(shlex.split(self.pytest_args))
        sys.exit(errno)


def generate_extensions():
    # We have to handle the .pyx modules
    # If Cython is available and .pyx file is available, extension is built with the .pyx file
    # If Cython is not available or .pyx file is not available, extension is built with .c file
    # It means that for each .pyx module, the compiled .c file must be put in version control
    if CYTHON:
        file_ext = '.c'
        extensions = [
            Extension(
                'scared.signal_processing._c_find_peaks', ['scared/signal_processing/_c_find_peaks' + file_ext],
                include_dirs=[numpy.get_include()]
            )
        ]
        return cythonize(extensions, compiler_directives={'always_allow_keywords': True})
    else:
        return []


setup(
    cmdclass={"test": PyTest},
    ext_modules=generate_extensions()
)
