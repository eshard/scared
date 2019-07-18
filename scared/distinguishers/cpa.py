from .base import _StandaloneDistinguisher, DistinguisherMixin, DistinguisherError
import numpy as _np
import logging

logger = logging.getLogger(__name__)


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
        logger.info(f'Start updating accumulators for {self.__class__.__name__} with traces {traces.shape} and data {data.shape}.')

        self.ey += _np.sum(_data, axis=0)
        self.ey2 += _np.sum(_data ** 2, axis=0)
        self.ex += _np.sum(_traces, axis=0)
        self.ex2 += _np.sum(_traces ** 2, axis=0)
        self.exy += _np.dot(_data.T, _traces)
        logger.info(f'End updating accumulators for {self.__class__.__name__}.')

    def _compute(self):
        trace_size = self.ex.shape[0]
        data_words = self.ey.shape[0]
        common_1 = _np.sqrt(self.ex2 - self.processed_traces * ((self.ex / self.processed_traces)**2))
        common_2 = _np.sqrt(self.ey2 - self.processed_traces * ((self.ey / self.processed_traces)**2))
        result = _np.empty((data_words, trace_size), dtype=self.precision)
        for d, (y, xy, com_2) in enumerate(zip(self.ey, self.exy, common_2)):
            tmp_result = (xy - (self.ex * (y / self.processed_traces))) / (common_1 * com_2)
            tmp_result[_np.isinf(tmp_result)] = _np.nan
            result[d] = tmp_result.astype(self.precision)
        return result

    @property
    def _distinguisher_str(self):
        return 'CPA'


class CPAAlternativeDistinguisherMixin(CPADistinguisherMixin):
    """Correlation Power Analysis using Pearson coefficients mixin.

    This version uses a compute method slightly different than the standard CPADistinguisherMixin.
    A matrix product is used, instead of a loop, which can improves computing time, but is paid with
    a peak memory consumption twice higher.

    Attributes:
        ex (numpy.ndarray): traces accumulator with shape (trace_size,)
        ex2 (numpy.ndarray): squared traces accumulator with shape (trace_size,)
        ey (numpy.ndarray): intermediate data accumulator with shape (data_words,)
        ey2 (numpy.ndarray): squared intermediate data accumulator with shape (data_words,)
        exy (numpy.ndarray): dot product (intermediate data, traces) accumulator with shape (data_words, trace_size)

    """

    def _compute(self):
        sigma_traces = _np.sqrt(self.processed_traces * self.ex2 - (self.ex) ** 2)
        sigma_data = _np.sqrt(self.processed_traces * self.ey2 - (self.ey) ** 2)
        enum = self.processed_traces * self.exy - _np.matmul(self.ey[:, None], self.ex[None, :])
        denom = _np.matmul(sigma_data[:, None], sigma_traces[None, :])
        return (enum / denom)


class CPADistinguisher(_StandaloneDistinguisher, CPADistinguisherMixin):
    """Standalone distinguisher class using CPA."""


class CPAAlternativeDistinguisher(_StandaloneDistinguisher, CPAAlternativeDistinguisherMixin):
    """Standalone distinguisher class using CPA."""
