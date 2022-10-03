#!/usr/bin/env python

import sys

from setuptools import setup
from setuptools.command.test import test
import logging
import versioneer

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


setup(
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(cmdclass={"test": PyTest}),
)
