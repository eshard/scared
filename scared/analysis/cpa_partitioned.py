"""CPA attack computed through trace partitioning, without any SCALib dependency.

This attack reproduces :class:`scared.CPAAttack` key ranking *and* correlation values, but
removes the guess dimension from the accumulation. Traces are partitioned once by their raw
target class, and every guess is enumerated cheaply at correlation time from a precomputed
model table.

The model table is built by evaluating the actual selection function on the canonical class
set for the whole guess space, so no assumption is made on how the guess and the target
combine (XOR, modular addition, multiplicative, ...): any structured selection function is
supported. The accumulation reuses scared's own
:class:`scared.distinguishers.partitioned.PartitionedDistinguisherMixin`, so the per-class
``sum``, ``sum_square`` and ``counters`` are exposed as attributes. The final step is a plain
Pearson correlation built from those statistics, which makes the result numerically identical
to :class:`scared.CPAAttack`.
"""

import numpy as _np

from .base import BasePartitionedAttack as _BasePartitionedAttack
from ..distinguishers.partitioned import PartitionedDistinguisherMixin as _PartitionedDistinguisherMixin
from ..distinguishers.base import DistinguisherError as _DistinguisherError

_MAX_PARTITION = 2 ** 16
_VERIFY_SAMPLE = 1024


class CPAPartitionedAttack(_BasePartitionedAttack, _PartitionedDistinguisherMixin):
    """CPA attack partitioning traces by raw target class and enumerating guesses via a model table.

    Rather than applying the leakage model per guess, the traces are partitioned once by the raw
    target value ``x``. A per-word model table ``M[x, g] = model(sf(x, g))`` is built once by
    evaluating the selection function on the canonical class set for the whole guess space, and the
    guess dimension is then swept with a single Pearson correlation over the per-class statistics.

    Because the table is obtained from the selection function itself, no assumption is made on the
    combination between target and guess: XOR, modular addition, multiplicative or any other
    structured selection function is supported, and the correlation matches :class:`scared.CPAAttack`
    exactly (not only on ranking). Selection functions whose modeled output is not a function of the
    partition class and the guess alone (for instance joint multi-word guessing) are rejected with a
    :class:`DistinguisherError`; use :class:`scared.CPAAttack` instead.

    Partitions follow the same convention as the other partitioned attacks
    (:class:`scared.ANOVAAttack`, :class:`scared.SNRAttack`, ...): the ``partitions`` argument is the
    set of raw target class values. If ``None``, it is derived from the data on the first batch as
    ``range(max_value + 1)``. The per-class accumulators ``sum``, ``sum_square`` and ``counters`` are
    available after processing for custom post-treatment.

    Notes:
        - The target must be integer-valued; class values must be in ``]-65536, 65536[``.
        - Accumulator memory grows as ``n_samples * n_words * n_partitions``; keep ``n_words`` small
          for large ``n_samples`` and partition counts.
        - A single word per guess is supported. This is an attack only, with no reverse counterpart.
    """

    def compute_intermediate_values(self, metadata):
        """Return the raw partition labels and build the model table on the first batch.

        Unlike the default analysis flow, the leakage model is not applied here. The raw target
        values are returned unchanged as the partition labels; scared's partitioned backend maps them
        to partition indices. On the first batch the model table is built and its structure verified.

        Args:
            metadata (mapping): dict-like object holding the traces metadata.

        Returns:
            numpy.ndarray: partition labels of shape ``(n_traces, n_words)``, integer dtype.
        """
        raw = _np.asarray(metadata[self.selection_function.target_tag])
        selected = self._select_words(raw)

        if self.partitions is None:
            self._set_auto_partitions(selected)

        if not hasattr(self, '_models'):
            self._n_target_words = raw.shape[-1]
            self._build_models(raw)
            self._verify_structure(raw, selected)

        return selected

    def _select_words(self, raw):
        """Apply the selection function word selection to a raw ``(n, n_bytes)`` array."""
        labels = raw.swapaxes(0, -1)[self.selection_function.words].swapaxes(0, -1)
        if labels.ndim == 1:
            labels = labels[:, None]
        return labels

    def _set_auto_partitions(self, selected):
        """Derive the partition class set from the first batch as ``range(max_value + 1)``."""
        mindata = int(_np.min(selected))
        maxdata = int(_np.max(selected))
        if mindata < 0:
            raise _DistinguisherError('Target values are negative; pass partitions explicitly (the raw target class set).')
        if maxdata >= _MAX_PARTITION:
            raise _DistinguisherError(f'Target values reach {maxdata}; class values must be in ]-{_MAX_PARTITION}, {_MAX_PARTITION}[.')
        self.partitions = _np.arange(maxdata + 1, dtype='int32')

    def _class_index(self, values):
        """Map raw target values to their partition index, matching the accumulator lookup."""
        offset = int(_np.min(self.partitions))
        lut = _np.full(int(_np.max(self.partitions)) - offset + 1, -1, dtype=_np.int64)
        lut[_np.asarray(self.partitions) - offset] = _np.arange(len(self.partitions))
        return lut[values.astype(_np.int64) - offset]

    def _build_models(self, raw):
        """Build the per-word model table by evaluating the selection function on the class set.

        For every partition class value ``x`` and guess ``g``, ``M[x, g, word] = model(sf(x, g))``.
        Because the selection function accepts the whole guess space at once, the entire table is
        obtained in a single evaluation on the canonical class values, independent of the trace count.
        """
        guesses = _np.asarray(self.selection_function.guesses)
        if guesses.ndim != 1:
            raise _DistinguisherError('CPAPartitionedAttack supports single-word guesses only. Use scared.CPAAttack for joint guessing.')

        class_values = _np.asarray(self.partitions).astype(_np.int64)
        canonical = _np.tile(class_values[:, None], (1, self._n_target_words))
        models = _np.asarray(self.model(self.selection_function(**{self.selection_function.target_tag: canonical})))
        if models.ndim == 2:
            models = models[:, :, None]
        self._models = models.astype(_np.float64)

    def _verify_structure(self, raw, selected):
        """Verify on a bounded sample that the modeled output depends only on the class and the guess.

        The structure is a property of the selection function, not of the data, so a bounded sample of
        real traces is enough to reject selection functions whose output is not a function of the
        partition class and the guess alone (for instance joint multi-word guessing).
        """
        sample_size = min(_VERIFY_SAMPLE, raw.shape[0])
        sample = _np.asarray(self.model(self.selection_function(**{self.selection_function.target_tag: raw[:sample_size]})))
        if sample.ndim == 2:
            sample = sample[:, :, None]

        index = self._class_index(selected[:sample_size])
        expected = _np.empty_like(sample, dtype=_np.float64)
        for word in range(sample.shape[2]):
            expected[:, :, word] = self._models[index[:, word], :, word]

        if not _np.allclose(sample.astype(_np.float64), expected):
            raise _DistinguisherError(
                'Selection function is not compatible with partitioned CPA: the modeled output is not a function of the partition class and the guess alone. '
                'This happens with joint multi-word guessing or selection functions using metadata beyond the partitioned target. Use scared.CPAAttack instead.'
            )

    def _compute(self):
        """Correlate every guess against the partition statistics with a Pearson coefficient.

        For a guess ``g`` every trace in class ``x`` shares the modeled leakage ``models[x, g, word]``,
        so the Pearson correlation reduces to a weighted combination of the per-class sums, independent
        of the trace count.

        Returns:
            numpy.ndarray: correlations of shape ``(n_guesses, n_words, n_samples)``.
        """
        if bool(((self.counters > 0).sum(axis=1) < 2).any()):
            raise _DistinguisherError('Fewer than two distinct target values were observed: the partition is degenerate. '
                                      'Process more varied traces before computing results.')

        total = float(self.processed_traces)
        n_guesses = self._models.shape[1]
        results = _np.empty((n_guesses, self._data_words, self._trace_length), dtype=self.precision)

        for word in range(self._data_words):
            class_sum = self.sum[:, word, :].astype(_np.float64).T
            class_sum_square = self.sum_square[:, word, :].astype(_np.float64).T
            class_count = self.counters[word].astype(_np.float64)

            trace_sum = class_sum.sum(axis=0)
            trace_variance = class_sum_square.sum(axis=0) - trace_sum ** 2 / total

            models = self._models[:, :, word].T
            weighted_models = models @ class_count
            model_variance = (models ** 2) @ class_count - weighted_models ** 2 / total

            covariance = models @ class_sum - weighted_models[:, None] * trace_sum[None, :] / total
            denominator = _np.sqrt(model_variance)[:, None] * _np.sqrt(trace_variance)[None, :]

            correlation = covariance / denominator
            correlation[_np.isinf(correlation)] = _np.nan
            results[:, word, :] = correlation.astype(self.precision)

        return results

    def __getstate__(self):
        """Return picklable state, dropping the non-picklable numba partition lookup (rebuilt on update)."""
        state = self.__dict__.copy()
        state.pop('_data_to_partition_index', None)
        return state

    def __setstate__(self, state):
        """Restore state, leaving the partition lookup unset until update is called again."""
        self.__dict__.update(state)

    @property
    def _distinguisher_str(self):
        return 'CPA_Partitioned'
