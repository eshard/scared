# SCAred

[![pipeline status](https://gitlab.com/eshard/scared/badges/master/pipeline.svg)](https://gitlab.com/eshard/scared/commits/master)
[![PyPI version](https://badge.fury.io/py/scared.svg)](https://pypi.org/project/scared/)
[![Conda installer](https://anaconda.org/eshard/scared/badges/installer/conda.svg)](https://anaconda.org/eshard/scared)
[![Latest Conda release](https://anaconda.org/eshard/scared/badges/latest_release_date.svg)](https://anaconda.org/eshard/scared)

scared is a side-channel analysis framework.

## Getting started

### Prerequisites

You will need **Python 3.6+** to use and install scared. You can use pip (or any pip based tool like pipenv) or conda to install it.

### Installation

To install scared, you can use pip (or pipenv, or any other pip based-tool) or conda:

```bash
$ pip install scared
# or with Conda
$ conda install -c eshard scared
```

### Make a first cool thing

Start using scared by doing a cool thing:

```python
# First import the lib
import scared

# Define a selection function
@scared.attack_selection_function
def first_sub_bytes(plaintext, guesses):
    res = np.empty((plaintext.shape[0], len(guesses), plaintext.shape[1]), dtype='uint8')
    for guess in guesses:
        res[:, guess, :] = np.bitwise_xor(plaintext, guess)
    return res

# Create an analysis CPA
a = scared.CPAAttack(
        selection_function=first_sub_bytes,
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

## Contributing

All contributions, starting with feedbacks, are welcomed.
Please read [CONTRIBUTING.md](CONTRIBUTING.md) if you wish to contribute to the project.

## License

This library is licensed under LGPL V3 license. See the [LICENSE](LICENSE) file for details.

It is mainly intended for non-commercial use, by academics, students or professional willing to learn the basics of side-channel analysis.

If you wish to use this library in a commercial or industrial context, eshard provides commercial licenses under fees. Contact us!

## Authors

See [AUTHORS](AUTHORS.md) for the list of contributors to the project.
