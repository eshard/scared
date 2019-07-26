from .partitioned import PartitionedDistinguisherBase, _PartitionnedDistinguisherBaseMixin
import numpy as _np
import logging

logger = logging.getLogger(__name__)


def _is_valid_bin_edges(bin_edges):
    return (
        bin_edges is not None and isinstance(
            bin_edges, (list, _np.ndarray, range)
        ) and len(bin_edges) > 1 and all(a < b for a, b in zip(bin_edges, bin_edges[1:]))
    )


class MIADistinguisherMixin(_PartitionnedDistinguisherBaseMixin):
    """This partitioned distinguisher mixin applies a mutual information computation."""

    def _memory_usage_coefficient(self, trace_size):
        return 3 * len(self.partitions) * self.bins_number * trace_size

    @property
    def bin_edges(self):
        return self._bin_edges

    @bin_edges.setter
    def bin_edges(self, bin_edges):
        if not _is_valid_bin_edges(bin_edges):
            raise TypeError(f'{bin_edges} bins are not valid bins.')
        self._bin_edges = bin_edges
        self.bins_number = len(bin_edges) - 1

    def _initialize_accumulators(self):
        self.accumulators = _np.zeros(
            (self._trace_length, self.bins_number, len(self.partitions), self._data_words),
            dtype=self.precision
        )

    def _accumulate(self, traces, data, bool_mask):
        logger.info(f'Start accumulation for {self.__class__.__name__}.')
        if self.bin_edges is None:
            logger.info(f'Start setting y_window and bin_edges.')
            self.y_window = (_np.min(traces), _np.max(traces))
            self.bin_edges = _np.linspace(*self.y_window, self.bins_number + 1)
            logger.info(f'Bin edges set.')

        bool_mask = bool_mask.astype('uint8')
        final_shape = (self.bins_number, len(self.partitions), self._data_words)
        logger.info(f'Will start loop on samples.')
        for s in range(self._trace_length):
            logger.info(f'Start processing histograms for samples {s}.')
            histos = _np.apply_along_axis(
                lambda a: _np.histogram(a, bins=self.bin_edges)[0],
                axis=-1,
                arr=traces[:, s: s + 1]
            ).astype(self.precision)
            logger.info(f'Histograms computed, add to accumulators.')
            logger.info(f'Histo shape {histos.shape} and dtype {histos.dtype}.')
            logger.info(f'Mask shape {bool_mask.shape} and dtype {bool_mask.dtype}.')
            dot_prod = _np.dot(histos.T, bool_mask)
            logger.info(f'Dot product computed.')
            self.accumulators[s, :, :, :] += dot_prod.reshape(final_shape)
            logger.info(f'Dot product added to accumulators.')

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

    def __init__(self, bins_number=128, bin_edges=None, partitions=None, precision='float32'):
        _set_histogram_parameters(self, bins_number=bins_number, bin_edges=bin_edges)
        return super().__init__(partitions=partitions, precision=precision)
