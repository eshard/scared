"""SCALib CPA attack wrapper for scared."""

import logging as _logging

import numpy as _np

from ..analysis.base import BaseAttack as _BaseAttack
from ..distinguishers.base import DistinguisherMixin as _DistinguisherMixin, DistinguisherError as _DistinguisherError
from ..utils.fast_astype import fast_astype as _fast_astype

_logger = _logging.getLogger(__name__)

try:
    from scalib.attacks import Cpa as _Cpa
    SCALIB_AVAILABLE = True
except ImportError:
    SCALIB_AVAILABLE = False
    _logger.warning("SCALib not available. CPAAttackSCALib will not work.")


_N_CLASSES = 256
_VERIFY_SAMPLE = 1024


def _is_full_guess_range(guesses):
    """Return whether ``guesses`` is the identity range ``arange(n_classes)`` in order.

    In that case the correlation reindexing by guess is an identity permutation and can be
    skipped, avoiding a full-array gather copy of the correlations.
    """
    return guesses.shape == (_N_CLASSES,) and bool((guesses == _np.arange(_N_CLASSES)).all())


class CPAAttackSCALib(_BaseAttack, _DistinguisherMixin):
    """CPA attack backed by SCALib's high-performance correlation engine.

    It reproduces :class:`scared.CPAAttack` but relies on a trick that removes the guess
    dimension from the main computation: rather than applying the leakage model once per
    guess, traces are partitioned once by the raw target value and every guess is
    enumerated cheaply at correlation time.

    The trick requires the selection function to combine target and guess through a key
    XOR (a bijective operation), so the modeled intermediate is a function of
    ``target ^ guess``. Selection functions that break this (non-XOR or non-injective, such
    as :class:`scared.aes.selection_functions.encrypt.DeltaRLastRounds`, or joint
    multi-word guessing) are rejected with a :class:`DistinguisherError`; use
    :class:`scared.CPAAttack` instead. This is an attack only, with no reverse counterpart.

    Notes:
        - SCALib must be installed.
        - Target values must be bytes (256 classes), a single word per guess.
        - Results match :class:`scared.CPAAttack` on key ranking; exact correlation values
          differ slightly due to a different model-variance normalization.
    """

    def _check_scalib_available(self):
        """Raise an informative error if SCALib is not installed."""
        if not SCALIB_AVAILABLE:
            raise ImportError("SCALib is not installed. Please install it with: pip install scalib\n"
                              "See https://github.com/simple-crypto/SCALib for more information.")

    def compute_intermediate_values(self, metadata):
        """Return the raw partition labels and keep building the model table.

        Unlike the default analysis flow, the leakage model is not applied here. The
        raw target bytes are returned as the SCALib partition labels. On the first batch
        the model table is built and the XOR structure is verified.

        Args:
            metadata (mapping): dict-like object holding the traces metadata.

        Returns:
            numpy.ndarray: partition labels of shape ``(n_traces, n_words)``, dtype uint16.
        """
        raw = _np.asarray(metadata[self.selection_function.target_tag])
        labels = self._select_words(raw)

        if _np.any(labels < 0) or _np.any(labels >= _N_CLASSES):
            raise _DistinguisherError(f'CPAAttackSCALib supports byte target values in [0, {_N_CLASSES}) only.')

        guesses = _np.asarray(self.selection_function.guesses)
        if _np.any(guesses < 0) or _np.any(guesses >= _N_CLASSES):
            raise _DistinguisherError(f'CPAAttackSCALib supports guess values in [0, {_N_CLASSES}) only.')

        if not hasattr(self, '_model_matrix'):
            self._initialize_model_table(labels.shape[1])
            self._build_model_table(raw)
            self._verify_xor_structure(raw, labels)

        self._record_observed_labels(labels)

        return labels.astype(_np.uint16)

    def _select_words(self, raw):
        """Apply the selection function word selection to a raw ``(n, n_bytes)`` array."""
        labels = raw.swapaxes(0, -1)[self.selection_function.words].swapaxes(0, -1)
        if labels.ndim == 1:
            labels = labels[:, None]
        return labels

    def _initialize_model_table(self, n_words):
        """Allocate the model table and the observed-label tracker."""
        self._nv = n_words
        self._model_matrix = _np.zeros((n_words, _N_CLASSES), dtype=_np.float64)
        self._observed_labels = _np.zeros((n_words, _N_CLASSES), dtype=bool)

    def _build_model_table(self, raw):
        """Fill the whole model table once from the canonical label values.

        For an XOR-structured selection function ``sf(x, g) = phi(x ^ g)``, class ``j`` maps
        to ``model(phi(j))``. Evaluating the selection function on the canonical labels
        ``arange(nc)`` at the reference guess ``guesses[0]`` sweeps every class exactly once,
        so the entire table is built in a single evaluation independent of the trace count.
        """
        guesses = _np.asarray(self.selection_function.guesses)
        if guesses.ndim != 1:
            raise _DistinguisherError('CPAAttackSCALib supports single-word guesses only. Use scared.CPAAttack for joint guessing.')

        canonical = _np.tile(_np.arange(_N_CLASSES, dtype=raw.dtype)[:, None], (1, raw.shape[-1]))
        reference = _np.asarray(self.model(self.selection_function(**{self.selection_function.target_tag: canonical})))
        if reference.ndim == 2:
            reference = reference[:, :, None]
        reference = reference[:, 0, :].astype(_np.float64)

        classes = _np.arange(_N_CLASSES) ^ int(guesses[0])
        for word in range(self._nv):
            self._model_matrix[word, classes] = reference[:, word]

    def _verify_xor_structure(self, raw, labels):
        """Verify on a bounded sample that the modeled intermediate is a function of ``target ^ guess``.

        The XOR structure is a property of the selection function, not of the data, so a bounded
        sample of real traces is enough to reject non-XOR selection functions (for instance
        :class:`DeltaRLastRounds`, non-injective ones, or joint guessing).
        """
        guesses = _np.asarray(self.selection_function.guesses).astype(_np.int64)
        sample_size = min(_VERIFY_SAMPLE, raw.shape[0])

        sample = _np.asarray(self.model(self.selection_function(**{self.selection_function.target_tag: raw[:sample_size]})))
        if sample.ndim == 2:
            sample = sample[:, :, None]
        sample = sample.transpose(0, 2, 1).astype(_np.float64)

        classes = labels[:sample_size].astype(_np.int64)[:, :, None] ^ guesses[None, None, :]
        expected = self._model_matrix[_np.arange(self._nv)[None, :, None], classes]

        if not _np.allclose(sample, expected):
            raise _DistinguisherError(
                'Selection function is not compatible with SCALib CPA: the modeled intermediate is not a function of (target XOR guess). '
                'This happens with non-XOR or non-injective selection functions (e.g. DeltaRLastRounds) or joint guessing. Use scared.CPAAttack instead.'
            )

    def _record_observed_labels(self, labels):
        """Track which raw label values have been observed, for the degenerate-partition guard."""
        self._observed_labels[_np.arange(self._nv)[None, :], labels.astype(_np.int64)] = True

    def _initialize(self, traces, data):
        """Create the SCALib CPA engine on first update."""
        self._check_scalib_available()
        self._trace_length = traces.shape[1]
        self._models = None
        use_64bit = _np.dtype(self.precision) == _np.float64
        self._cpa = _Cpa(_N_CLASSES, _Cpa.Xor, use_64bit)

    def _update(self, traces, data):
        """Partition the traces by their raw labels using SCALib."""
        if self._cpa is None:
            raise RuntimeError('This CPAAttackSCALib instance was restored from serialization and cannot accumulate new traces. Use stored results.')
        if traces.shape[1] != self._trace_length:
            raise ValueError(f'traces has different length {traces.shape[1]} than already processed traces {self._trace_length}.')

        traces_int16 = _fast_astype(traces, dtype='int16', order='C')
        labels_uint16 = _np.ascontiguousarray(data.astype(_np.uint16))
        self._cpa.fit_u(traces_int16, labels_uint16)

    def _correlation_models(self):
        """Return the ``(n_words, n_classes, n_samples)`` model tensor SCALib expects.

        SCALib requires a leakage model per sample, but the analytical model is sample
        independent: the same ``(n_words, n_classes)`` table is broadcast over the samples.
        The materialized tensor is cached because ``_model_matrix`` and the trace length are
        fixed after initialization, so correlation checkpoints reuse it instead of rebuilding.
        """
        if self._models is None:
            self._models = _np.ascontiguousarray(
                _np.broadcast_to(self._model_matrix[:, :, None], (self._nv, _N_CLASSES, self._trace_length)),
                dtype=_np.float64,
            )
        return self._models

    def _compute(self):
        """Compute correlations for every guess and return shape ``(n_guesses, n_words, n_samples)``."""
        if self._cpa is None:
            raise RuntimeError('This CPAAttackSCALib instance was restored from serialization and cannot recompute results. Use stored results.')
        if bool((self._observed_labels.sum(axis=1) < 2).any()):
            raise _DistinguisherError('Fewer than two distinct target values were observed: the partition is degenerate. '
                                      'Process more varied traces before computing results.')

        correlations = self._cpa.get_correlation(self._correlation_models())
        guesses = _np.asarray(self.selection_function.guesses).astype(_np.int64)
        reindexed = correlations.transpose(1, 0, 2)
        if not _is_full_guess_range(guesses):
            reindexed = reindexed[guesses]
        return reindexed.astype(self.precision, copy=False)

    def _memory_usage(self, traces, data):
        """Estimate the memory needed for the correlation computation."""
        dtype_size = _np.dtype(self.precision).itemsize
        return 2 * dtype_size * _N_CLASSES * data.shape[1] * traces.shape[1]

    def __getstate__(self):
        """Return picklable state, dropping the non-picklable engine and the rebuildable model cache."""
        state = self.__dict__.copy()
        state['_cpa'] = None
        state['_models'] = None
        return state

    def __setstate__(self, state):
        """Restore state, leaving the SCALib engine unset until update is called again."""
        self.__dict__.update(state)

    @property
    def _distinguisher_str(self):
        return 'CPA_SCALib'
