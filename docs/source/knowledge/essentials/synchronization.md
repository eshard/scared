# The synchronization and signal processing APIs

`scared` provides tools to apply signal processing functions to a trace set, and synchronize it.
We present here the basic concepts underlying these.

## The signal processing package

The `scared.signal_processing` package provides several signal processing functions.

Each signal processing function takes a Numpy array argument, plus parameters specific to the processing and returns a new Numpy array.

The signal processing functions provided include:

- basic functions, like `pad`
- `filters`  with a  `butterworth`
- `moving_operators` with sum, mean, kurtosis, ...
- `pattern_detection` with correlation, distance, bcdc
- `peaks_detection` functions
- `fft` analysis

## The synchronization API

In order to proceed to synchronization of a trace set, `scared` provides the `Synchronizer` class.

A `Synchronizer` instance will take a `TraceHeaderSet` instance as input, a synchronization function to apply to traces samples, and a filename to store the output trace header set.

The synchronization function can be tested on a small sample of traces with the `check` method.

The output file is an ETS file.

