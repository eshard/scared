# SCAred

[![pipeline status](https://gitlab.com/eshard/scared/badges/master/pipeline.svg)](https://gitlab.com/eshard/scared/commits/master)
[![PyPI version](https://badge.fury.io/py/scared.svg)](https://pypi.org/project/scared/)
[![Conda installer](https://anaconda.org/eshard/scared/badges/installer/conda.svg)](https://anaconda.org/eshard/scared)
[![Latest Conda release](https://anaconda.org/eshard/scared/badges/latest_release_date.svg)](https://anaconda.org/eshard/scared)

scared is a side-channel analysis framework maintained by [eShard](http://www.eshard.com) team.

## Getting started

### Requirements

Scared need python **3.6**, **3.7** or **3.8**.

You can install `scared`, depending on your setup:

- from source
- with `pip`
- with `conda`

#### Install with `conda`

You just have to run:

```bash
conda install -c eshard scared
```

#### Install with `pip`

Python wheels are available from Pypi, just run:

```bash
pip install scared
```

#### Install from sources

To install from sources, you will need to run:

```bash
pip install .
```

from the source folder.

If you are planning to contribute, see [CONTRIBUTING.md](CONTRIBUTING.md) to install the library in development mode and run the test suite.

### Make a first cool thing

Start using scared by doing a cool thing:

```python
# First import the lib
import scared
import numpy as np

# Define a selection function
@scared.attack_selection_function
def first_add_key(plaintext, guesses):
    res = np.empty((plaintext.shape[0], len(guesses), plaintext.shape[1]), dtype='uint8')
    for i, guess in enumerate(guesses):
        res[:, i, :] = np.bitwise_xor(plaintext, guess)
    return res

# Create an analysis CPA
a = scared.CPAAttack(
        selection_function=first_add_key,
        model=scared.HammingWeight(),
        discriminant=scared.maxabs)

# Load some traces, for example a dpa v2 subset
ths = scared.traces.read_ths_from_ets_file('dpa_v2.ets')

# Create a container for your ths
container = scared.Container(ths)

# Run!
a.run(container)
```

## Documentation

To go further and learn all about scared, please go to [the full documentation](https://eshard.gitlab.io/scared).
You can also have an interactive introduction to scared by launching these [notebooks with Binder](https://mybinder.org/v2/gl/eshard%2Fscared-notebooks/master).

## Contributing

All contributions, starting with feedbacks, are welcomed.
Please read [CONTRIBUTING.md](CONTRIBUTING.md) if you wish to contribute to the project.

## License

This library is licensed under LGPL V3 license. See the [LICENSE](LICENSE) file for details.

It is mainly intended for non-commercial use, by academics, students or professional willing to learn the basics of side-channel analysis.

If you wish to use this library in a commercial or industrial context, [eShard](https://www.eshard.com) provides commercial licenses under fees. [Contact us](mailto:scared@eshard.com)!

## Authors

See [AUTHORS](AUTHORS.md) for the list of contributors to the project.

## Binary builds available

Binary builds (wheels on pypi and conda builds) are available for the following platforms and Python version.

Platforms:

- Linux x86 64
- Macosx x86 64

Python version:

- 3.6
- 3.7
- 3.8
