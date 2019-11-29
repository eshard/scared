# Intermediate values

`scared` provides tools to compute intermediate values from trace sets metadata.

There are two components to compute intermediate values: selection functions and leakage models.

## Selection functions

Selection functions are used to compute intermediate values from a trace set metadata.

`scared` provides basic tools to define selection functions compatible with the leakage analysis and attack framework.

### The selection_function decorator

At its core, a selection function is a callable which, given a set of data, compute and returns some values.

To create a basic selection function, the simplest way is to decorate your function:

```python
import scared

@scared.selection_function
def identity(plaintext):
    return plaintext
```

This function can then be safely used with the analysis framework. The most important constraint is that a selection function is expected to keep unchanged the first dimension of data arguments (typically, if 10 plaintexts are used, 10 intermediate values are expected).

Selection functions arguments are expected to be found in the metadata of your trace header set when using it in the analysis framework.

The base selection function also manage the selection of some words for you. If you are interested only in a subset of words, you can pass the value directly to the decorator:

```python
@scared.selection_function(words=[1, 2, 8])
def subset(plaintext):
    return plaintext
```

The decorator will take care of slicing the result for you on the targeted words.

Alternatively to the decorator use, you can also define a selection function with a more classic functionnal syntax, which is practical for the use of lambdas:

```python
identity = scared.selection_function(lambda plaintext: plaintext)
```

### The attack selection function

A specialized decorator `attack_selection_function` is provided for the special case of attack analysis.

Additionaly to base selection function, an attack selection function:

- must have a `guesses` arguments
- can have an `expected_key_function`, which is used by the decorated function to provide a `compute_expected_key` function

The decorator takes care of you for all the internal mechanisms of passing guesses to the real function.

For example, an attack selection function can be defined like this:

```python
@scared.attack_selection_function(guesses=np.arange(128, dtype="uint8"), expected_key_function=lambda key: key)
def identity(guesses, plaintext):
    ...
```

Attack selection function are used by the analysis framework.

### Ready to use selection functions for AES and DES

`scared` includes AES and DES ciphers utilities. More specifically, you can use ready-to-use attack selection functions for classical intermediate values.

These selection functions are classes that you must instantiate, so that you can customize the guesses, words or other paramaters as needed.

For example, with AES:

```python
from scared import aes

s = aes.selection_functions.encrypt.FirstSubBytes()
```

is an attack selection function for the first sub bytes output of AES.

## Leakage models

Once selection function is used, the resulting values must be passed through a leakage model to obtain intermediate values.

The API for models provided by `scared` is pretty simple.
At its core, a leakage model is a subclass of the abstract base class `scared.Model`. This base class handles all the details of checking parameters and so on, and ensure compatibility with the overall framework.

A model implementation must have a `_compute` method, which takes a `data` argument and makes the computation of the model.

Three standard leakage models are provided with `scared`:

- `scared.Value` model
- `scared.Monobit` model
- `scared.HammingWeight` model
