# Analysis and distinguishers

Leakage analysis and attacks in `scared` are handled by two complementary concepts:

- Base analysis classes, in analysis package, are responsible to run the side-channel analysis on a container, given all the parameters (selection function, models, ...)
- Distinguisher mixins are responsible for the analysis method: CPA, DPA, ...

The combination of a base analysis class and distinguisher mixin is a usable analysis class, as provided by `scared`: `CPAAttack`, `CPAReverse`, `DPAAttack`, `DPAReverse`, ...

## The distinguishers

The distinguisher concept is the core of leakage and attack side-channel analysis.

The responsibility of a distinguisher in `scared` is to provide methods to compute some scores from batches of traces samples and corresponding intermediate values computed from metadata.

The basic piece used to build all distinguisher, and by extension analysis objects, is the `DistinguisherMixin` abstract class.

### The DistinguisherMixin

`scared.DistinguisherMixin` is abstract base class, and a mixin. As such, it must be:

- subclassed, to provides concrete implementations of abtsract method
- combined to another class to be instantiated

As a mixin, the `DistinguisherMixin` provides two public APIs used to compute side-channel analysis:

- `update` takes a set of traces samples of values `traces` and corresponding intermediate values `data` as arguments. It is responsible to update internal state of the distingusher and proceed to any computation needed to accumulate on the trace set.
- `compute`method proceeds to scores computation and returns the result, based on the current internal state

For both public methods, the base `DistinguisherMixin` handles common tricky tasks: linearization of data, some basic memory management and warning, ...

### Distinguishers concrete implementation

A concrete implementation needs to have:

- an `_initialize` method, which is responsible to initialize the state of the distinguisher at first update call
- an `_update` method, which is responsible for updating the concrete distinguisher
- a `compute` method, which is responsible for the concrete score computation of the distinguisher

`scared` provided several concrete implementations mixin: `CPADistinguisherMixin`, `DPADistinguisherMixin`, ...

### Initialized state of a distinguisher

To be usable, one must initialize some expected attributes for a mixin:

- `processed_traces`
- `precision`
- `_is_checked` internal state

This concrete implementation is taken care of by the base analysis class, but you can also have the need of a standalone distinguisher.

The `Distinguisher` class provided by `scared` is an abstract base class implementing a simple `__init__` method. It can be sub-classed to create new standalone distinguisher.

Concrete, stand-alone distinguishers are also provided for all standard methods: `CPADistinguisher`, `DPADistingusher`, ...

These classes can be used as stand-alone distinguishers, but in most cases you will use directly an analysis object.

## The analysis base class

The analysis base classes are objects that takes selection function, leakage model and precision as parameter and are responsible to provide a `run` method, which takes a `Container` instance and process it.

It takes care of computing intermediate values, looping over trace set batches.

These classes are also specialized for leakage analysis `BaseReverse` and attack `BaseAttack`.  The attack base class provides mechanism to handle scores computation with a discriminant function, convergence results by keeping intermediate results, and so on.

To obtain a complete attack or reverse class, you need to create a class which inherit from a distinguisher mixin and a base class, like this:

```python
class CPAAttack(BaseAttack, CPADistinguisherMixin):
    pass
```

`scared` provides out-of-the-box reverse and attack objects for CPA, DPA, ANOVA, NICV, SNR and MIA analysis. Distinguishers mixins and standalone distinguishers classes are also provided.
