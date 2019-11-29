# SCAred

[![pipeline status](https://gitlab.com/eshard/scared/badges/master/pipeline.svg)](https://gitlab.com/eshard/scared/commits/master)
[![PyPI version](https://badge.fury.io/py/scared.svg)](https://pypi.org/project/scared/)
[![Conda installer](https://anaconda.org/eshard/scared/badges/installer/conda.svg)](https://anaconda.org/eshard/scared)
[![Latest Conda release](https://anaconda.org/eshard/scared/badges/latest_release_date.svg)](https://anaconda.org/eshard/scared)

scared is a side-channel analysis framework.

## Getting started

### Pre-requisites

To use scared, you will need the the following Python **3.6** or **3.7**.

If your configuration has a binary build available ([see list here](#binary-builds-available)), you can just run:

```bash
pip install scared
```

```bash
conda install -c eshard scared
```

If your system isn't yet supported ([build contributions are welcome!](./CONTRIBUTING.md#building-for-conda)), or you want to build from source or develop, you'll also need:

- setuptools **0.40 or greater** (just run `pip install -U pip setuptools`)
- a C compiler to compile C extension

To install from source, you just need to run:

```bash
pip install .
```

from the source folder.

### Make a first cool thing

Start using scared by doing a cool thing:

```python
# First import the lib
import scared

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
ths = scared.traces.read_ths_from_ets('dpa_v2.ets')

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

If you wish to use this library in a commercial or industrial context, eshard provides commercial licenses under fees. Contact us!

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
