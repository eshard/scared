from .base import DistinguisherMixin, _StandaloneDistinguisher, DistinguisherError
import numpy as _np
import logging

logger = logging.getLogger(__name__)


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

        logger.info(f'Start updating accumulators for {self.__class__.__name__} with traces {traces.shape} and data {data.shape}.')

        self.processed_ones += _np.sum(data, axis=0)
        traces = traces.astype(self.precision)
        data = data.astype(self.precision)
        self.accumulator_traces += _np.sum(traces, axis=0)
        self.accumulator_ones += _np.dot(data.T, traces)
        logger.info(f'End updating accumulators for {self.__class__.__name__}.')

    def _compute(self):
        normalized_ones = (self.accumulator_ones.swapaxes(0, 1) / self.processed_ones).swapaxes(0, 1)
        processed_zeros = self.processed_traces - self.processed_ones
        accumulator_zeros = self.accumulator_traces - self.accumulator_ones
        normalized_zeros = (accumulator_zeros.swapaxes(0, 1) / processed_zeros).swapaxes(0, 1)
        return (normalized_ones - normalized_zeros)

    @property
    def _distinguisher_str(self):
        return 'DPA'


class DPADistinguisher(_StandaloneDistinguisher, DPADistinguisherMixin):
    """Standalone distinguisher class using DPA."""
