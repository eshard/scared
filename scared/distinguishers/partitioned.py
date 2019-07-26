from .base import DistinguisherMixin, _StandaloneDistinguisher
import numpy as _np
import logging

logger = logging.getLogger(__name__)


class _PartitionnedDistinguisherBaseMixin(DistinguisherMixin):

    def _memory_usage_coefficient(self, trace_size):
        return 3 * len(self.partitions) * trace_size

    def _initialize(self, traces, data):
        maxdata = _np.nanmax(data)
        if self.partitions is None:
            if maxdata > 255:
                raise ValueError(f'max value for intermediate data is greater than 255, you need to provide partitions explicitly at init.')
            ls = [0, 9, 64, 256]
            for r in ls:
                if maxdata <= r:
                    break
            self.partitions = range(r)
        self._trace_length = traces.shape[1]
        self._data_words = data.shape[1]
        try:
            self._initialize_accumulators()
        except MemoryError:
            raise MemoryError(f'Trace size and data words are too large to proceed with accumulation.')

    def _update(self, traces, data):
        if traces.shape[1] != self._trace_length:
            raise ValueError(f'traces has different length {traces.shape[1]} than already processed traces {self._trace_length}.')
        if data.shape[1] != self._data_words:
            raise ValueError(f'data has different number of data words {data.shape[1]} than already processed data {self._data_words}.')

        traces = traces.astype(self.precision)
        data = data.astype(self.precision)

        logger.info(f'Update of partitioned distinguisher {self.__class__.__name__} in progress.')

        bool_mask = _np.empty((traces.shape[0], 0), dtype=bool)
        for j, partition in enumerate(self.partitions):
            logger.info(f'Start processing partition {j} for boolean mask.')
            tmp_bool = data == partition
            bool_mask = _np.append(bool_mask, tmp_bool, axis=1)
            self._accumulate_partition(
                partition_indice=j,
                partition_value=partition,
                part_bool=tmp_bool
            )

        logger.info(f'Start accumulation of traces with boolean mask.')
        self._accumulate(traces, data, bool_mask)
        logger.info(f'End of accumualtions of traces for {self.__class__.__name__}.')

    def _accumulate_partition(self, partition_indice, partition_value, part_bool):
        pass


class PartitionedDistinguisherMixin(_PartitionnedDistinguisherBaseMixin):
    """Base mixin for various traces partitionning based attacks (ANOVA, NICV, SNR, ...).

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

    def _accumulate(self, traces, data, bool_mask):
        self.sum += _np.dot(traces.T, bool_mask).reshape(
            (traces.shape[1], data.shape[1], len(self.partitions)), order='F'
        )
        self.sum_square += _np.dot((traces ** 2).T, bool_mask).reshape(
            (traces.shape[1], data.shape[1], len(self.partitions)), order='F'
        )

    def _accumulate_partition(self, partition_indice, partition_value, part_bool):
        self.counters[:, partition_indice] += _np.sum(part_bool, axis=0)

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
        if not isinstance(partitions, _np.ndarray):
            raise TypeError(f'partitions should be a Numpy ndarray instance, not {type(partitions)}.')
        if not partitions.dtype.kind == 'i':
            raise ValueError(f'partitions should be an integer array, not {partitions.dtype}.')
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
