import abc
import numpy as _np


class DistinguisherError(Exception):
    pass


class DistinguisherMixin(abc.ABC):
    """Abstract distinguisher mixin class.

    A distinguisher mixin is used either as a component of :class:`BaseAnalysis` child, or as a standalone object by subclassing it directly.
    It defines the distinguisher interface, which has two public methods:

        - `update` takes traces and intermediate values data and update the state of the distinguisher.
        - `compute` returns the result of the distinguisher applied on the current state

    This abstrace class defines three private methods that must be implemented by each distinguisher mixin:
        - `_initialize` which is called on first `update` call and can be leverage to complete process init.
        - `_update` which implements the specific of the distinguisher update process.
        - `_compute` which implements the specific of the distinguisher compute process.

    """

    def update(self, traces, data):
        if not isinstance(traces, _np.ndarray):
            raise TypeError(f'traces must be numpy ndarray, not {type(traces)}.')
        if not isinstance(data, _np.ndarray):
            raise TypeError(f'data must be numpy ndarray, not {type(data)}.')

        o_shape = data.shape
        data = data.reshape((o_shape[0], -1))
        try:
            self._origin_shape
        except AttributeError:
            self._origin_shape = o_shape
            self._initialize(traces, data)

        self.processed_traces += traces.shape[0]
        self._update(traces, data)

    @abc.abstractmethod
    def _initialize(self, traces, data):
        pass

    @abc.abstractmethod
    def _update(self, traces, data):
        pass

    def compute(self):
        try:
            assert self.processed_traces > 0
        except (AttributeError, AssertionError):
            raise DistinguisherError('Distinguisher has not been initialized, or no traces have been processed.\
                Please initialize and update the distinguisher before trying to use compute function.')
        return self._compute().reshape((self._origin_shape[1:] + (-1, )))

    @abc.abstractmethod
    def _compute(self):
        pass


class DPADistinguisherMixin(DistinguisherMixin):
    """Differential Power Analysis mixin.

    Attributes:
        accumulator_traces (numpy.ndarray): accumulator of all traces with shape (trace_size,)
        accumulator_ones (numpy.ndarray): accumulator of traces corresponding to intermediate value 1, with shape (trace_size,)
        processed_ones (numpy.ndarray): number of processed traces in accumulator_ones for each data word considered, with shape (data_words,)

    """

    def _initialize(self, traces, data):
        max_data = _np.nanmax(data)
        trace_size = traces.shape[1]
        data_words = data.shape[1]
        if not isinstance(max_data, (int, _np.uint8)):
            raise TypeError(f'Error during DPA initialization: max data value should be an integer, not {type(max_data)}. Are you using a monobit model?')
        if max_data < 0 or max_data > 1:
            raise ValueError(f'Error during DPA initialization: max data should be between 0 and 1, not {max_data}. Are you using a monobit model?')
        try:
            self.accumulator_traces = _np.zeros((trace_size), dtype=self.precision)
            self.accumulator_ones = _np.zeros((data_words, trace_size), dtype=self.precision)
            self.processed_ones = _np.zeros((data_words), dtype='uint32')
        except (ValueError, MemoryError) as e:
            raise type(e)(f'Trace size and data words are too large to proceed with accumulation for DPA {e}')

    def _update(self, traces, data):
        if traces.shape[1] != self.accumulator_traces.shape[0]:
            raise DistinguisherError(f'traces have different size {traces.shape[1]} than already processed traces {self.accumulator_traces.shape[0]}.')
        self.processed_ones += _np.sum(data, axis=0)
        traces = traces.astype(self.precision)
        data = data.astype(self.precision)
        self.accumulator_traces += _np.sum(traces, axis=0)
        self.accumulator_ones += _np.dot(data.T, traces)

    def _compute(self):
        normalized_ones = (self.accumulator_ones.swapaxes(0, 1) / self.processed_ones).swapaxes(0, 1)
        processed_zeros = self.processed_traces - self.processed_ones
        accumulator_zeros = self.accumulator_traces - self.accumulator_ones
        normalized_zeros = (accumulator_zeros.swapaxes(0, 1) / processed_zeros).swapaxes(0, 1)
        return (normalized_ones - normalized_zeros)


class CPADistinguisherMixin(DistinguisherMixin):
    """Correlation Power Analysis using Pearson coefficients mixin.

    Attributes:
        ex (numpy.ndarray): traces accumulator with shape (trace_size,)
        ex2 (numpy.ndarray): squared traces accumulator with shape (trace_size,)
        ey (numpy.ndarray): intermediate data accumulator with shape (data_words,)
        ey2 (numpy.ndarray): squared intermediate data accumulator with shape (data_words,)
        exy (numpy.ndarray): dot product (intermediate data, traces) accumulator with shape (data_words, trace_size)

    """

    def _initialize(self, traces, data):
        trace_size = traces.shape[1]
        data_words = data.shape[1]
        try:
            self.ex = _np.zeros((trace_size), dtype=self.precision)
            self.ex2 = _np.zeros((trace_size), dtype=self.precision)
            self.ey = _np.zeros((data_words), dtype=self.precision)
            self.ey2 = _np.zeros((data_words), dtype=self.precision)
            self.exy = _np.zeros((data_words, trace_size), dtype=self.precision)
        except (ValueError, MemoryError) as e:
            raise type(e)(f'Trace size and data words are too large to proceed with accumulation for CPA {e}')

    def _update(self, traces, data):
        if traces.shape[1] != self.ex.shape[0]:
            raise DistinguisherError(f'traces have different size {traces.shape[1]} than already processed traces {self.ex.shape[0]}.')

        _traces = traces.astype(self.precision)
        _data = data.astype(self.precision)

        self.ey += _np.sum(_data, axis=0)
        self.ey2 += _np.sum(_data ** 2, axis=0)
        self.ex += _np.sum(_traces, axis=0)
        self.ex2 += _np.sum(_traces ** 2, axis=0)
        self.exy += _np.dot(_data.T, _traces)

    def _compute(self):
        sigma_traces = _np.sqrt(self.processed_traces * self.ex2 - (self.ex) ** 2)
        sigma_data = _np.sqrt(self.processed_traces * self.ey2 - (self.ey) ** 2)
        enum = self.processed_traces * self.exy - _np.matmul(self.ey[:, None], self.ex[None, :])
        denom = _np.matmul(sigma_data[:, None], sigma_traces[None, :])
        return (enum / denom)

    def _alternative_compute(self):
        trace_size = self.ex.shape[0]
        data_words = self.ey.shape[0]
        common1 = _np.sqrt(self.ex2 - self.processed_traces * ((self.ex / self.processed_traces)**2))
        common2 = _np.sqrt(self.ey2 - self.processed_traces * ((self.ey / self.processed_traces)**2))
        result = _np.empty((data_words, trace_size), dtype=self.precision)
        for d, (y, xy, com2) in enumerate(zip(self.ey, self.exy, common2)):
            numerator = xy - (self.ex * (y / self.processed_traces))
            denominator = common1 * com2
            tmp_result = numerator / denominator
            tmp_result[_np.isinf(tmp_result)] = _np.nan
            if tmp_result.dtype != self.precision:
                tmp_result = tmp_result.astype(self.precision)
            result[d] = tmp_result
        return result


def _set_precision(obj, precision):
    try:
        precision = _np.dtype(precision)
    except TypeError:
        raise TypeError(f'precision should be a valid dtype, not {precision}.')

    if precision.kind != 'f':
        raise ValueError(f'precision should be a float dtype, not {precision.kind}.')
    obj.precision = precision


def _initialize_distinguisher(obj, precision, processed_traces):
    _set_precision(obj, precision)
    obj.processed_traces = processed_traces


class _StandaloneDistinguisher:

    def __init__(self, precision='float32'):
        _initialize_distinguisher(self, precision, processed_traces=0)


class Distinguisher(_StandaloneDistinguisher, DistinguisherMixin):
    """Abstract stand-alone distinguisher class.

    This class must be inherited and implements the distinguisher methods.
    """


class CPADistinguisher(_StandaloneDistinguisher, CPADistinguisherMixin):
    """Standalone distinguisher class using CPA."""


class DPADistinguisher(_StandaloneDistinguisher, DPADistinguisherMixin):
    """Standalone distinguisher class using DPA."""
