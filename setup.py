#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from setuptools import setup
from setuptools.command.test import test


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


setup(
    cmdclass={"test": PyTest},
)
