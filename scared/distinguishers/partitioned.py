from .base import DistinguisherMixin, _StandaloneDistinguisher
from ..utils.misc import _use_parallel

import numpy as _np
import numba as _nb
import time as _time
import logging as _logging

_parallel = _use_parallel()

logger = _logging.getLogger(__name__)


class _PartitionnedDistinguisherBaseMixin(DistinguisherMixin):

    def _memory_usage(self, traces, data):
        self._init_partitions(data)
        dtype_size = _np.dtype(self.precision).itemsize
        return 3 * dtype_size * data.shape[1] * traces.shape[1] * len(self.partitions)

    def _init_partitions(self, data):
        maxdata = _np.nanmax(data)
        mindata = _np.nanmin(data)
        if self.partitions is None:
            if maxdata > 255:
                raise ValueError('max value for intermediate data is greater than 255, you need to provide partitions explicitly at init.')
            if mindata < 0:
                raise ValueError('min value for intermediate data is lower than 0, you need to provide partitions explicitly at init.')
            ls = [0, 9, 64, 256]
            for r in ls:
                if maxdata <= r:
                    break
            self.partitions = _np.arange(r, dtype='int32')

    def _initialize(self, traces, data):
        self._trace_length = traces.shape[1]
        self._data_words = data.shape[1]
        self._data_to_partition_index = _define_lut_func(self.partitions)
        self._initialize_accumulators()

    def _update(self, traces, data):
        if traces.shape[1] != self._trace_length:
            raise ValueError(f'traces has different length {traces.shape[1]} than already processed traces {self._trace_length}.')
        if data.shape[1] != self._data_words:
            raise ValueError(f'data has different number of data words {data.shape[1]} than already processed data {self._data_words}.')
        if not _np.issubdtype(data.dtype, _np.integer):
            raise TypeError(f'data dtype for partitioned distinguisher, including MIA and Template, must be an integer dtype, not {data.dtype}.')
        logger.info(f'Update of partitioned distinguisher {self.__class__.__name__} in progress.')
        data = self._data_to_partition_index(data)
        self._accumulate(traces, data)
        logger.info(f'End of accumulations of traces for {self.__class__.__name__}.')


@_nb.njit()
def _build_lut(partitions):
    lut = _np.zeros(2**17, dtype='int32') - 1
    for i in _np.arange(len(partitions)):
        lut[partitions[i]] = i
    return lut


def _define_lut_func(partitions):
    lut = _build_lut(partitions)

    @_nb.vectorize([_nb.int32(_nb.uint8), _nb.int32(_nb.uint16), _nb.int32(_nb.uint32), _nb.int32(_nb.uint64),
                    _nb.int32(_nb.int8), _nb.int32(_nb.int16), _nb.int32(_nb.int32), _nb.int32(_nb.int64)])
    def _lut_function(x):
        return lut[x]

    return _lut_function


class PartitionedDistinguisherMixin(_PartitionnedDistinguisherBaseMixin):
    """Base mixin for various traces partitioning based attacks (ANOVA, NICV, SNR, ...).

    Attacks differs mainly in the metric computation, not in the accumulation process.

    Attributes:
        partitions (numpy.ndarray or range, default=None): partitions used to categorize traces according to intermediate data value.
            if None, it will be automatically estimated at first update of distinguisher.
        sum (numpy.ndarray): sum of traces accumulator with shape (trace_size, data_words, len(partitions))
        sum_square (numpy.ndarray): sum of traces squared accumulator with shape (trace_size, data_words, len(partitions))
        counters (numpy.ndarray): number of traces accumulated by data word and partitions, with shape (data_words, len(partitions)).

    """

    def _initialize_accumulators(self):
        self.sum = _np.zeros((self._trace_length, self._data_words, len(self.partitions)), dtype=self.precision)
        self.sum_square = _np.zeros((self._trace_length, self._data_words, len(self.partitions)), dtype=self.precision)
        self.counters = _np.zeros((self._data_words, len(self.partitions)), dtype=self.precision)

    @staticmethod
    @_nb.njit(parallel=_parallel)
    def _accumulate_core_1(traces, data, self_sum, self_sum_square, self_counters, self_precision):
        for sample_idx in _nb.prange(traces.shape[1]):
            tmp_sum = _np.zeros((self_counters.shape[0], self_counters.shape[1]), dtype='float64')
            tmp_sum_square = _np.zeros((self_counters.shape[0], self_counters.shape[1]), dtype='float64')
            for trace_idx in range(traces.shape[0]):
                x = traces[trace_idx, sample_idx]
                xx = x * x
                for data_idx in range(data.shape[1]):
                    data_value = data[trace_idx, data_idx]
                    if data_value != -1:
                        tmp_sum[data_idx, data_value] += x
                        tmp_sum_square[data_idx, data_value] += xx
                        if sample_idx == 0:
                            self_counters[data_idx, data_value] += 1
            self_sum[sample_idx] += tmp_sum
            self_sum_square[sample_idx] += tmp_sum_square

    @staticmethod
    @_nb.njit(parallel=_parallel)
    def _accumulate_core_2(traces, data, self_sum, self_sum_square, self_counters, self_precision):
        """Faster when number of partitions is <=9."""
        ftraces = traces.astype(self_precision)
        bool_mask = _np.empty((traces.shape[0], data.shape[1] * self_counters.shape[1]), dtype=self_precision)
        for p in range(self_counters.shape[1]):
            tmp_bool = data == p  # Data are already transformed to correspond to partition indexes.
            self_counters[:, p] += tmp_bool.sum(0)
            bool_mask[:, p * data.shape[1]:(p + 1) * data.shape[1]] = tmp_bool
        self_sum += (bool_mask.T @ ftraces).reshape(self_counters.shape[1], data.shape[1], traces.shape[1]).T
        self_sum_square += (bool_mask.T @ (ftraces ** 2)).reshape(self_counters.shape[1], data.shape[1], traces.shape[1]).T

    def _accumulate(self, traces, data):
        """If the number of partitions is >9, the method 1 is selected.

        Otherwise, the fastest method is selected empirically.
        """
        if len(self.partitions) > 9:
            self._accumulate_core_1(traces, data, self.sum, self.sum_square, self.counters, self.precision)
        else:
            if not hasattr(self, '_timings'):
                self._timings = [-2, -1]
            function_idx = _np.argmin(self._timings)
            function = [self._accumulate_core_1, self._accumulate_core_2][function_idx]
            t0 = _time.process_time()
            function(traces, data, self.sum, self.sum_square, self.counters, self.precision)
            self._timings[function_idx] = _time.process_time() - t0

    def _compute(self):
        self.sum = self.sum.swapaxes(0, 1)
        self.sum_square = self.sum_square.swapaxes(0, 1)

        result = _np.empty((self._data_words, self._trace_length), dtype=self.precision)

        for i in range(self._data_words):
            non_zero_indices = self.counters[i] > 0
            non_zero_counters = self.counters[i][non_zero_indices]
            sums = self.sum[i][:, non_zero_indices]
            sums_squared = self.sum_square[i][:, non_zero_indices]
            number_non_zero = _np.sum(non_zero_counters)

            tmp_result = self._compute_metric(
                non_zero_indices, non_zero_counters, sums, sums_squared, number_non_zero
            )
            tmp_result[_np.isinf(tmp_result)] = _np.nan
            result[i] = tmp_result.astype(self.precision)

        self.sum = self.sum.swapaxes(0, 1)
        self.sum_square = self.sum_square.swapaxes(0, 1)

        return result


def _set_partitions(obj, partitions):
    if partitions is not None:
        if not isinstance(partitions, (_np.ndarray, list, range)):
            raise TypeError(f'partitions should be a ndarray, list or range instance, not {type(partitions)}.')
        if not isinstance(partitions, _np.ndarray):
            partitions = _np.array(partitions, dtype='int32')
        elif partitions.dtype.kind not in 'iu':
            raise ValueError(f'partitions should be an integer array, not {partitions.dtype}.')
        if _np.max(partitions) >= 2**16:
            raise ValueError(f'partition values must be in ]-2^16, 2^16[, but {_np.max(partitions)} found.')
        if _np.min(partitions) <= -2**16:
            raise ValueError(f'partition values must be in ]-2^16, 2^16[, but {_np.min(partitions)} found.')
    obj.partitions = partitions


class PartitionedDistinguisherBase(_StandaloneDistinguisher):
    def __init__(self, partitions=None, precision='float32'):
        super().__init__(precision=precision)
        _set_partitions(self, partitions=partitions)


class PartitionedDistinguisher(PartitionedDistinguisherBase, PartitionedDistinguisherMixin):
    pass


class ANOVADistinguisherMixin(PartitionedDistinguisherMixin):
    """This standalone partitioned distinguisher applies the ANOVA F-test metric."""

    def _compute_metric(self, non_zero_indices, non_zero_counters, sums, sums_squared, number_non_zero):
        total_non_empty_partitions = _np.count_nonzero(non_zero_indices)

        partitions_means = (sums / non_zero_counters)
        mean = _np.sum(sums, axis=-1, keepdims=True) / number_non_zero

        numerator = _np.sum(
            (non_zero_counters * (partitions_means - mean) ** 2),
            axis=-1
        ) / (total_non_empty_partitions - 1)

        denominator = _np.sum(
            (sums_squared - sums ** 2 / non_zero_counters),
            axis=-1
        ) / (number_non_zero - total_non_empty_partitions)

        return numerator / denominator

    @property
    def _distinguisher_str(self):
        return 'ANOVA'


class ANOVADistinguisher(PartitionedDistinguisherBase, ANOVADistinguisherMixin):
    __doc__ = PartitionedDistinguisherMixin.__doc__ + ANOVADistinguisherMixin.__doc__


class NICVDistinguisherMixin(PartitionedDistinguisherMixin):
    """This standalone partitioned distinguisher applies the NICV (Normalized Inter-Class Variance) metric."""

    def _compute_metric(self, non_zero_indices, non_zero_counters, sums, sums_squared, number_non_zero):
        mean = _np.sum(sums, axis=1) / number_non_zero

        numerator = (((sums / non_zero_counters).T - mean).T)**2
        numerator *= non_zero_counters / number_non_zero
        numerator = _np.sum(numerator, axis=1)

        denominator = _np.sum(sums_squared, axis=1) / number_non_zero - (mean)**2

        return numerator / denominator

    @property
    def _distinguisher_str(self):
        return 'NICV'


class NICVDistinguisher(PartitionedDistinguisherBase, NICVDistinguisherMixin):
    __doc__ = PartitionedDistinguisherMixin.__doc__ + NICVDistinguisherMixin.__doc__


class SNRDistinguisherMixin(PartitionedDistinguisherMixin):
    """This standalone partitioned distinguisher applies the SNR (Signal to Noise Ratio) metric."""

    def _compute_metric(self, non_zero_indices, non_zero_counters, sums, sums_squared, number_non_zero):
        mean = _np.sum(sums, axis=1) / number_non_zero
        numerator = (((sums / non_zero_counters).T - mean).T)**2
        numerator = _np.sum(numerator, axis=1) / non_zero_indices.shape[0]

        denominator = (sums_squared / non_zero_counters) - (sums / non_zero_counters)**2
        denominator = _np.sum(denominator, axis=1) / non_zero_indices.shape[0]

        return numerator / denominator

    @property
    def _distinguisher_str(self):
        return 'SNR'


class SNRDistinguisher(PartitionedDistinguisherBase, SNRDistinguisherMixin):
    __doc__ = PartitionedDistinguisherMixin.__doc__ + SNRDistinguisherMixin.__doc__
