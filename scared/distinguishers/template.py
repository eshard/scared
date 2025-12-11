from scared.utils.inplace_dot_sum import inplace_dot_sum as _inplace_dot_sum
from scared.utils.fast_astype import fast_astype as _fast_astype
from . import partitioned, base
import logging as _logging
import numpy as _np


logger = _logging.getLogger(__name__)


class _TemplateBuildDistinguisherMixin(partitioned._PartitionnedDistinguisherBaseMixin):

    def _initialize_accumulators(self):
        self._exi = _np.zeros(shape=(len(self.partitions), self._trace_length), dtype=self.precision, order='C')
        self._exxi = _np.zeros(shape=(len(self.partitions), self._trace_length, self._trace_length), dtype=self.precision, order='C')
        self._counters = _np.zeros(shape=(len(self.partitions)), dtype=self.precision)
        self.pooled_covariance = None
        self.pooled_covariance_inv = None

    def _accumulate(self, traces, data):
        """Do the core of the data partitioning."""
        logger.debug('Accumulate build distinguisher')
        bool_mask = _np.empty((len(self.partitions), len(data)), dtype='bool')
        for p in range(len(self.partitions)):
            bool_mask[p] = data[:, 0] == p  # Data are already transformed to correspond to partition indexes
        self._counters += _np.sum(bool_mask, axis=1)
        traces = _fast_astype(traces, self.precision)
        _inplace_dot_sum(bool_mask, traces, self._exi)
        for p in range(len(self.partitions)):
            tmp = traces[bool_mask[p]]
            tmp = _np.dot(tmp.T, tmp)
            self._exxi[p] += tmp

    def _compute(self):
        self.pooled_covariance = _np.zeros((self._trace_length, self._trace_length), dtype=self.precision)

        tmp_counters = _np.copy(self._counters).astype(self.precision)
        if _np.any(tmp_counters <= 1):
            logger.warning('Some template categories have less than 2 traces to build template')
        tmp_counters[tmp_counters <= 1] = 2

        templates = (self._exi.swapaxes(0, 1) / tmp_counters).swapaxes(0, 1)
        tmp_matrix = None
        for i, p in enumerate(self.partitions):
            tmp_matrix = (_np.outer(templates[i], templates[i]) * tmp_counters[i])
            self.pooled_covariance += (self._exxi[i] - tmp_matrix) / (tmp_counters[i] - 1)

        self.pooled_covariance /= len(self.partitions)
        self.pooled_covariance_inv = _np.linalg.pinv(self.pooled_covariance)
        return templates

    @property
    def _distinguisher_str(self):
        return 'Template build'

    def _check(self, traces, data):
        if data.shape[1] != 1:
            raise base.DistinguisherError(f'Intermediate data for template attack must return only 1 word after model, not {data.shape[1]}')
        return super()._check(traces, data)


class _BaseTemplateAttackDistinguisherMixin(base.DistinguisherMixin, partitioned.PartitionedDistinguisherBase):

    def _initialize(self, traces, data):
        if not self.is_build:
            raise base.DistinguisherError('Template must be build before you can run it on a matching container.')
        if traces.shape[1] != self.pooled_covariance.shape[1]:
            raise base.DistinguisherError(
                f'Trace size for matching {traces.shape[1]} is different than trace size used for building {self.pooled_covariance.shape[1]}.'
            )
        self._scores = self._initialize_scores(data)

    def _compute(self):
        return (10 - (self._scores / self.processed_traces))


class TemplateAttackDistinguisherMixin(_BaseTemplateAttackDistinguisherMixin):
    """Static template attack mixin.

    This mixin distinguisher proceeds to the matching phase, once the template are build.
    """

    def _update(self, traces, data):
        for i in range(len(self.partitions)):
            tmp_traces = traces - self.templates[i]
            tmp = _np.dot(tmp_traces, self.pooled_covariance_inv)
            tmp = tmp * tmp_traces
            self._scores[i] += tmp.sum() / traces.shape[1]

    def _initialize_scores(self, data):
        return _np.zeros(shape=(len(self.partitions), ), dtype=self.precision)

    @property
    def _distinguisher_str(self):
        return 'Template'


class TemplateDPADistinguisherMixin(_BaseTemplateAttackDistinguisherMixin):
    """DPA template attack mixin.

    This mixin distinguisher proceeds to the matching phase, once the template are build.
    """

    def _update(self, traces, data):
        for i in range(len(self.partitions)):
            tmp_traces = traces - self.templates[i]
            tmp = _np.dot(tmp_traces, self.pooled_covariance_inv)
            tmp = tmp * tmp_traces
            tmp = tmp.sum(1) / traces.shape[1]
            self._scores += _np.dot(tmp, data == i)

    def _initialize_scores(self, data):
        return _np.zeros(shape=(data.shape[1], ), dtype=self.precision)

    @property
    def _distinguisher_str(self):
        return 'TemplateDPA'
