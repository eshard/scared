<!-- TOC -->

- [Contributing to scared](#contributing-to-scared)
  - [Code of conduct](#code-of-conduct)
  - [Contribution Suitability](#contribution-suitability)
  - [Bug reports](#bug-reports)
  - [Feature Requests](#feature-requests)
  - [Documentation contributions](#documentation-contributions)
  - [Code contributions](#code-contributions)
    - [Steps for Submitting Code](#steps-for-submitting-code)
      - [Merge requests approval check-list](#merge-requests-approval-check-list)
    - [Code Review](#code-review)
    - [New Contributors](#new-contributors)
    - [Get Early Feedback](#get-early-feedback)
    - [Python development guidelines](#python-development-guidelines)
      - [Requirements](#requirements)
      - [Test suite](#test-suite)
    - [Numpy compressed array files](#numpy-compressed-array-files)
      - [Code style and formatting](#code-style-and-formatting)
      - [Docstrings](#docstrings)
      - [Building for Pypi](#building-for-pypi)
      - [Building for Conda](#building-for-conda)

<!-- /TOC -->

# Contributing to scared

Thank you for considering to contribute to scared project! There are various ways to contribute to the project, and every contributions are welcomed.

This document lays out guidelines and advice for contributing to this project. If you’re thinking of contributing, please start by reading this document and getting a feel for how contributing to this project works.

If you have any questions, please feel free to reach to the core contributors at scared@eshard.com.

## Code of conduct

All forms of contribution and discussions on this project must respect some basic behavioral rules:

- be kind
- be respectful

## Contribution Suitability

The core project maintainers have the last word on whether or not a contribution is suitable. All contributions will be considered carefully, but from time to time, contributions will be rejected because they do not suit the current goals or needs of the project.

If your contribution is rejected, don’t despair! As long as you followed these guidelines, you will have a much better chance of getting your next contribution accepted.

## Bug reports

Bug reports are hugely important! Before you raise one, though, please check through the Gitlab issues, both open and closed, to confirm that the bug hasn’t been reported before. Duplicate bug reports are a huge drain on the time of other contributors, and should be avoided as much as possible.

When filling a new bug report, please use the appropriate template on the bug tracker, and gives as much as possible a reproducible scenario with all context elements.

## Feature Requests

If you believe there is a feature missing, feel free to raise a feature request. You should use create an issue in the Gitlab issue tracker, with the appropriate template. Gives as much as possible context elements and rationale around why the feature you wish should be implemented.

Please do be aware that the overwhelming likelihood is that your feature request will not be accepted, or will take time to be implemented.

## Documentation contributions

Documentation improvements are always welcome! The documentation files live in the `docs/` directory of the codebase. They’re written in Markdown, and use Sphinx to generate the full suite of documentation.

When contributing documentation, please do your best to follow the style of the documentation files. When presenting Python code, use single-quoted strings ('hello' instead of "hello").

## Code contributions

### Steps for Submitting Code

When contributing code, you’ll want to follow this checklist:

- Fork the repository on Gitlab.
- Run the tests to confirm they all pass on your system. If they don’t, you’ll need to investigate why they fail. If you’re unable to diagnose this yourself, raise it as a bug report by following the guidelines in this document: [Bug Reports](#bug-reports).
- Write tests that demonstrate your bug or feature. Ensure that they fail.
- Make your change. Add you to the AUTHORS file.
- Run the entire test suite again, confirming that all tests pass including the ones you just added.
- Send a Gitlab Merge Request to the main repository’s master branch. Gitlab Merge Requests are the expected method of code collaboration on this project.

#### Merge requests approval check-list

For your merge request to be reviewed and eventually merged into master, please use the following check-list:

- [ ] Related issue, documented and qualified
- [ ] The development is finished (pipeline passing) and include new or modified tests
- [ ] Docstrings have been updated if needed
- [ ] Documentation is updated if relevant
- [ ] A commit message following [conventional commit](https://www.conventionalcommits.org/) is proposed by the submitter for merge message in the master
- [ ] Approval by one of the maintainer is obtained

The following sub-sections go into more detail on some of the points above.

### Code Review

Contributions will not be merged until they’ve been code reviewed. You should implement any code review feedback unless you strongly object to it. In the event that you object to the code review feedback, you should make your case clearly and calmly. If, after doing so, the feedback is judged to still apply, you must either apply the feedback or withdraw your contribution.

### New Contributors

If you are new or relatively new to Open Source, welcome! If you’re concerned about how best to contribute, please consider mailing a maintainer (listed above) and asking for help.

Please also check the [Get Early Feedback](#get-early-feedback) section.

### Get Early Feedback

If you are contributing, do not feel the need to sit on your contribution until it is perfectly polished and complete. It helps everyone involved for you to seek feedback as early as you possibly can. Submitting an early, unfinished version of your contribution for feedback in no way prejudices your chances of getting that contribution accepted, and can save you from putting a lot of work into a contribution that is not suitable for the project.

- have an issue on Gitlab. No issue, no dev.
- checkout the project
- create a branch from the master or identify the branch you should be working on, be it a feature or a bug branch. Reference the issue number in the branch name
- work on the branch
- commit often small functional perimeter - some passing tests with it's working code. Typical commit frequency is several times each day - several begins at 2 …
- push less often, but regularly. Typical push frequency is daily.
- when work is finished, ask for final review and merge request to the upstream.

### Python development guidelines

For code contributions, please follows these guidelines.

#### Requirements

All developments should be compatible with **Python 3.6+** versions.

To develop, you'll need to have:

- setuptools **0.40 or greater** (just run `pip install -U pip setuptools`)

To start running your test suite, you must install the library in development mode:

```bash
pip install -e .
```

#### Test suite

The project uses pytest. To run all the test suite, simply run from you directory:

```bash
pytest
```

Please refer to [Pytest documentation](https://docs.pytest.org/en/latest/contents.html).
As far as possible, tests should reflects the organization of the source code. Create folders and keep everything as clean as possible.

### Numpy compressed array files

To save visual and disk space, some test modules use numpy compressed files ending with .npz to store array samples.

You can put a new array into an existing file like this:

```python
my_dict = dict(numpy.load('tests/samples/the_appropriate_file.npz'))
my_dict['my_new_array_name'] = my_new_numpy_array
numpy.savez('tests/samples/the_appropriate_file.npz', **my_dict)
```

#### Code style and formatting

The projects follow the Python standards defined by [PEP8](https://www.python.org/dev/peps/pep-0008/), with some exceptions. The maximum line length is 160 characters.

You should lint your code with `flake8`, `pep8-namin` and `flake8-docstrings`. The linter configuration is in `.flake8` file.

#### Docstrings

Each **public API** should be documented following [PEP257](https://www.python.org/dev/peps/pep-0257/).

It is up to you to decide if a there is a value in a docstring. In some cases, there is no value in a docstring for trivial methods or functions, like getters and setters ( but, well, maybe you shouldn't do too much getters and setters).

We use the Google style guide conventions for docstrings, and the API documentation is generated by Sphinx.

- As far as possible, arguments and return types should be documented.
- [Google docstring examples](http://www.sphinx-doc.org/en/stable/ext/example_google.html) - Really complete examples
- [Support of Google docstring in Sphinx](http://www.sphinx-doc.org/en/stable/ext/napoleon.html#docstring-sections), with the list supported section headers
- [Python domain directives in Sphinx](http://www.sphinx-doc.org/en/1.7/domains.html?highlight=python%20domain#the-python-domain), to enhance API documentation with cross-linking

#### Building for Pypi

To build for Pypi, you will need to follow these instructions:

- `pip install -U pip setuptools wheel`
- Run `python setup.py bdist_wheel` from the root folder
- Get build files in `dist/` directory

#### Building for Conda

To build for Conda, you will need to create a dedicated conda environment for build:

- `conda create -n buildenv python=3.6 conda-build conda-verify`
- `conda config --add channels eshard`
- Run `conda build --output_folder out .recipe` from the root source folder
- Get build files in `out` sub directory
