from .partitioned import PartitionedDistinguisherBase, _PartitionnedDistinguisherBaseMixin
from ..utils.misc import _use_parallel
import numpy as _np
import numba as _nb
import logging

_parallel = _use_parallel()
logger = logging.getLogger(__name__)


class MIADistinguisherMixin(_PartitionnedDistinguisherBaseMixin):
    """This partitioned distinguisher mixin applies a mutual information computation."""

    def _memory_usage(self, traces, data):
        self._init_partitions(data)
        self._init_bin_edges(traces)
        dtype_size = _np.dtype(self.precision).itemsize
        return 3 * dtype_size * data.shape[1] * traces.shape[1] * len(self.partitions) * self.bins_number

    def _init_bin_edges(self, traces):
        if self.bin_edges is None:
            logger.info('Start setting y_window and bin_edges.')
            self.y_window = (_np.min(traces), _np.max(traces))
            self.bin_edges = _np.linspace(*self.y_window, self.bins_number + 1)
            logger.info('Bin edges set.')

    @property
    def bin_edges(self):
        return self._bin_edges

    @bin_edges.setter
    def bin_edges(self, bin_edges):
        if bin_edges is None or not isinstance(bin_edges, (list, _np.ndarray, range)):
            raise TypeError(f'bin_edges must be a ndarray, a list or a range, not {type(bin_edges)}.')
        if len(bin_edges) <= 1:
            raise ValueError(f'bin_edges length must be >1, but {len(bin_edges)}, found.')
        if not isinstance(bin_edges, _np.ndarray):
            bin_edges = _np.array(bin_edges, dtype='float64')
        for a, b in zip(bin_edges, bin_edges[1:]):
            if not a < b:
                raise ValueError(f'bin_edges must be sorted, but {a} >= {b}.')
        if _np.sum(_np.diff(_np.diff(bin_edges))) > 1e-9:
            raise ValueError('bin_edges must be uniform (i.e with bins equally spaced.')
        self._bin_edges = bin_edges
        self.bins_number = len(bin_edges) - 1

    def _set_precision(self, precision):
        try:
            precision = _np.dtype(precision)
        except TypeError:
            raise TypeError(f'precision should be a valid dtype, not {precision}.')
        self.precision = precision

    def _initialize_accumulators(self):
        self.accumulators = _np.zeros((self._trace_length, self.bins_number, len(self.partitions), self._data_words),
                                      dtype=self.precision)

    @staticmethod
    @_nb.njit(parallel=_parallel)
    def _accumulate_core(traces, data, self_bin_edges, self_accumulators):
        nbins = len(self_bin_edges) - 1
        min_edge = self_bin_edges[0]
        max_edge = self_bin_edges[-1]
        norm = nbins / (max_edge - min_edge)

        for sample_idx in _nb.prange(traces.shape[1]):
            for trace_idx in range(traces.shape[0]):
                x = traces[trace_idx, sample_idx]
                if x >= min_edge and x < max_edge:
                    bin_idx = int((x - min_edge) * norm)
                elif x == max_edge:
                    bin_idx = nbins - 1
                else:
                    continue
                for data_idx in range(data.shape[1]):
                    self_accumulators[sample_idx, bin_idx, data[trace_idx, data_idx], data_idx] += 1

    def _accumulate(self, traces, data):
        self._accumulate_core(traces, data, self.bin_edges, self.accumulators)

    def _compute_pdf(self, array, axis):
        s = array.sum(axis=axis)
        s[s == 0] = 1
        return (array.swapaxes(0, 1) / s).swapaxes(0, 1)

    def _compute(self):
        background = self.accumulators.sum(axis=2)

        pdfs_background = self._compute_pdf(background, axis=1)
        pdfs_background[pdfs_background == 0] = 1

        pdfs_of_histos = self._compute_pdf(self.accumulators, axis=1)
        pdfs_of_histos[pdfs_of_histos == 0] = 1

        histos_sums = self.accumulators.sum(axis=1)
        ratios = (histos_sums.swapaxes(0, 1) / background.sum(axis=1)).swapaxes(0, 1)
        expected = pdfs_background * _np.log(pdfs_background)
        real = pdfs_of_histos * _np.log(pdfs_of_histos)
        delta = (real.swapaxes(1, 2).swapaxes(0, 1) - expected).swapaxes(0, 1).swapaxes(1, 2)
        res = delta.sum(axis=1) * ratios
        return _np.sum(res, axis=1).swapaxes(0, 1)

    @property
    def _distinguisher_str(self):
        return 'MIA'


def _set_histogram_parameters(obj, bins_number, bin_edges):
    if not isinstance(bins_number, int):
        raise TypeError(f'bins_number must be an integer, not {type(bins_number)}.')
    obj.bins_number = bins_number
    obj._bin_edges = None
    obj.y_window = None
    if bin_edges is not None:
        obj.bin_edges = bin_edges


class MIADistinguisher(PartitionedDistinguisherBase, MIADistinguisherMixin):

    def __init__(self, bins_number=128, bin_edges=None, partitions=None, precision='uint32'):
        _set_histogram_parameters(self, bins_number=bins_number, bin_edges=bin_edges)
        return super().__init__(partitions=partitions, precision=precision)
