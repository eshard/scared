#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys

from setuptools import setup
from setuptools.command.test import test
from setuptools import Extension
import logging

logger = logging.getLogger(__name__)


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
    try:
        import numpy
        extensions = [
            Extension(
                'scared.signal_processing._c_find_peaks', ['scared/signal_processing/_c_find_peaks.pyx'],
                include_dirs=[numpy.get_include()],
                compiler_directives={'always_allow_keywords': True}
            )
        ]
        return extensions
    except ModuleNotFoundError as e:
        logger.error(
            '''Numpy is required to build or install scared.
You can install numpy with pip if you are trying to build,
or use "pip install ." instead of "python setup.py install"
if you just want to use scared from local files.
            ''')
        raise e


setup(
    cmdclass={"test": PyTest},
    ext_modules=generate_extensions()
)
