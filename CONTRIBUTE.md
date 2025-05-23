<!-- TOC -->

- [How to contribute to scared library](#how-to-contribute-to-scared-library)
- [The developer workflow](#the-developer-workflow)
- [Pre-requisites](#pre-requisites)
  - [Python](#python)
- [Testing](#testing)
  - [Development testing](#development-testing)
    - [Run tests](#run-tests)
    - [Develop tests](#develop-tests)
- [Code style and formatting](#code-style-and-formatting)
  - [Tooling](#tooling)
  - [Project configuration](#project-configuration)
  - [Linting](#linting)
- [Docstrings](#docstrings)
  - [Conventions](#conventions)
  - [Tooling](#tooling-1)

<!-- /TOC -->

# How to contribute to scared library

This guide describes all you need to know to develop, test, build and distribute scared library.

# The developer workflow

- have an issue on Gitlab. No issue, no dev.
- checkout the project
- create a branch from the master or identify the branch you should be working on, be it a feature or a bug branch. Reference the issue number in the branch name
- work on the branch
- commit often small functionnal perimeter - some passing tests with it's working code. Typical commit frequency is several times each day - several begins at 2 …
- push less often, but regularly. Typical push frequency is daily.
- when work is finished, ask for final review and merge request to the upstream.

# Pre-requisites

## Python

You need **Python 3.6** installed. To easily get and manage Python versions, see Anaconda distributions - for example [Miniconda](https://conda.io/miniconda.html)

You can use pip with or without virtualenv, conda to manage your dependencies locally.

# Testing

## Development testing

### Run tests

The project is configured to use pytest for as a runner for testing.

To run all the test suite, simply run from you directory:

```bash
pytest
```

### Develop tests

You can use both the basic Python `unittest` package, or `pytest` to develop your test cases. `pytest` allows for less verbose tests and is recomended.

- [Pytest documentation](https://docs.pytest.org/en/latest/contents.html)

As far as possible, tests should reflects the organization of the source code. Create folders and keep everything as clean as possible.


# Code style and formatting

You must follow the Python standards defined by [PEP8](https://www.python.org/dev/peps/pep-0008/). The only exception is that the maximum line length is 160 characters. [Google style guide](https://google.github.io/styleguide/pyguide.html) is a good complementary source, but is not used as a reference.

## Tooling

`flake8` is a Python linting tool wrapping other tools, used to lint Python code around:

- PEP8 style guide with `pycodestyle`
- check for errors (unused import for example) with `pyflakes`
- check for McCabe complexity with `mccabe`

The library `pep8-naming` is also use, through `flake8` plugin, to check PEP8 naming conventions.

[Flake8 documentation](http://flake8.pycqa.org/en/latest/index.html)

## Project configuration

The linter configuration is in `.flake8` file. It defines the **law** for the project.

## Linting

You have to install a pre-commit hook, from the root of the project:

```bash
flake8 --install-hook git
git config --bool flake8.strict true
```

You will not be able to commit any Python file if some errors are reported by `flake8` when you try to commit. 

You are strongly encouraged to configure your preferred text editor / environment development to get realtime linting when you edit files. A lot of plugins exist - check you tools documentation to set up your environment.

Additionally, you can lint your code on the command line, by simply launching flake8 from your virtualenv on a file:

```bash
flake8 my_file.py
```

# Docstrings

Each **public API** should be documented following [PEP257](https://www.python.org/dev/peps/pep-0257/).

It is up to you to decide if a there is a value in a docstring. In some cases, there is no value in a docstring for trivial methods or functions, like getters and setters ( but, well, maybe you shouldn't do too much getters and setters).

For private APIs, it is up to you to decide if there is value in documentation and code comments.

## Conventions

We use the Google style guide conventions for docstrings, and the API documentation is generated by Sphinx.

- As far as possible, arguments and return types should be documented.
- [Google docstring examples](http://www.sphinx-doc.org/en/stable/ext/example_google.html) - Really complete examples
- [Support of Google docstring in Sphinx](http://www.sphinx-doc.org/en/stable/ext/napoleon.html#docstring-sections), with the list supported section headers
- [Python domain directives in Sphinx](http://www.sphinx-doc.org/en/1.7/domains.html?highlight=python%20domain#the-python-domain), to enhance API documentation with cross-linking

## Tooling

The `flake8-docstrings` plugin is used to check for your doc strings. However, errors related to missing docstring are ignored. Presence of a dosctring should be a human decision ( based on the principle that each public API should be documented ).

If you need it, you can check for missing docstrings by running:

```bash
flake8 --select D1 <path_or_filename>
```
