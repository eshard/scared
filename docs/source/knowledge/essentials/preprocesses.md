# Preprocesses

With `scared`, preprocesses are all functions which:

- are applied to traces samples
- produce new traces, conserving the number of traces
- are applied before proceeding to leakage analysis or attack

The `preprocesses` API allows to apply transformations to traces in-memory.

This document presents the basic concepts of this abstraction.

## The preprocess decorator

In the simplest case, `scared` provides a decorator to define a preprocess. The `scared.preprocess` decorator will some types and shapes compatibility test to the preprocess function, in order to have compatibility with the `scared` framework. 

A typical preprocess definition could thus be:

```python
import scared

@scared.preprocess
def square(traces):
    return traces ** 2
```

The first preprocess argument is expected to be a 2 dimension Numpy array.
The value return by a preprocess function will be expected to be a 2 dimension Numpy array, and to keep the first dimension of the input array intact.

## The Preprocess class

Sometimes, you will need to define more complex preprocess. In that case, you can define a class inheriting from `scared.Preprocess`. This class must implement a `__call__` method. 

A power preprocess class could look like this:

```python
import scared
class Power(scared.Preprocess):

    def __init__(self, n):
        self.n =

    def __call__(self, traces):
        return traces ** self.n

square = Power(2)
```

## Available pre-processes

`scared` provides a bunch of ready to use preprocesses functions and classes.

There are two categories:

- simple preprocesses that applies all traces, like `center`, `serialize_bit`, `standardize`, ...
- combination preprocesses to product high order analysis by combining several frames of traces, available in `high_order` package
