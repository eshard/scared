"""SCALib SNR Distinguisher wrapper for scared."""

import numpy as _np
import logging as _logging

from ..distinguishers.base import DistinguisherMixin as _DistinguisherMixin, _StandaloneDistinguisher
from ..distinguishers.partitioned import _set_partitions
from ..utils.fast_astype import fast_astype as _fast_astype

_logger = _logging.getLogger(__name__)

try:
    from scalib.metrics import SNR as _SCALibSNR  # noqa: N811
    SCALIB_AVAILABLE = True
except ImportError:
    SCALIB_AVAILABLE = False
    _logger.warning("SCALib not available. SNRDistinguisherSCALib will not work.")


class SNRDistinguisherSCALibMixin(_DistinguisherMixin):
    """Mixin for SNR distinguisher using SCALib's high-performance implementation.

    This distinguisher wraps SCALib's SNR metric to provide the same interface as
    scared's built-in SNRDistinguisher, while leveraging SCALib's optimized implementation.

    Attributes:
        partitions (numpy.ndarray or range, default=None): partitions used to categorize
            traces according to intermediate data value. If None, it will be automatically
            estimated at first update of distinguisher.

    Notes:
        - SCALib must be installed to use this distinguisher
        - This wrapper handles conversion from arbitrary partitions to SCALib's 0-indexed classes
        - Results should match scared's SNRDistinguisher within numerical precision
    """

    def _check_scalib_available(self):
        """Check if SCALib is available and raise error if not."""
        if not SCALIB_AVAILABLE:
            raise ImportError("SCALib is not installed. Please install it with: pip install scalib\n"
                              "See https://github.com/simple-crypto/SCALib for more information.")

    def _memory_usage(self, traces, data):
        """Estimate memory usage for the distinguisher."""
        self._init_partitions(data)
        dtype_size = _np.dtype(self.precision).itemsize
        return 3 * dtype_size * data.shape[1] * traces.shape[1] * len(self.partitions)

    def _init_partitions(self, data):
        """Initialize partitions from data if not already set."""
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
        """Initialize the SCALib SNR metric on first update call."""
        self._check_scalib_available()

        self._trace_length = traces.shape[1]
        self._data_words = data.shape[1]
        self._build_partition_mapping()

        nc = len(self.partitions)
        use_64bit = self.precision == _np.float64
        self._scalib_snr = _SCALibSNR(nc=nc, use_64bit=use_64bit)

        _logger.info(f"Initialized SCALib SNR with {nc} classes, {self._trace_length} samples, {self._data_words} variables")

    def _build_partition_mapping(self):
        """Build a lookup table to map partition values to indices."""
        self._partition_to_index = _np.full(2**17, -1, dtype=_np.int32)
        for idx, partition_value in enumerate(self.partitions):
            self._partition_to_index[partition_value] = idx

    def _map_data_to_indices(self, data):
        """Convert data from partition values to class indices."""
        return self._partition_to_index[data]

    def _update(self, traces, data):
        """Update the SNR estimation with new traces and data."""
        if traces.shape[1] != self._trace_length:
            raise ValueError(f'traces has different length {traces.shape[1]} than already processed traces {self._trace_length}.')
        if data.shape[1] != self._data_words:
            raise ValueError(f'data has different number of data words {data.shape[1]} than already processed data {self._data_words}.')
        if not _np.issubdtype(data.dtype, _np.integer):
            raise TypeError(f'data dtype for partitioned distinguisher must be an integer dtype, not {data.dtype}.')

        _logger.info('Update of SNRDistinguisherSCALib in progress.')

        mapped_data = self._map_data_to_indices(data)
        traces_int16 = _fast_astype(traces, dtype='int16', order='C')
        mapped_data_uint16 = mapped_data.astype(_np.int32)
        valid_traces_mask = _np.all(mapped_data_uint16 >= 0, axis=1)

        if not _np.any(valid_traces_mask):
            _logger.warning("No valid traces to update (all data values not in partitions)")
            return

        traces_filtered = traces_int16[valid_traces_mask]
        mapped_data_filtered = mapped_data_uint16[valid_traces_mask].astype(_np.uint16)

        self._scalib_snr.fit_u(traces_filtered, mapped_data_filtered)
        _logger.info(f'Updated SCALib SNR with {traces_filtered.shape[0]} valid traces.')

    def _compute(self):
        """Compute and return the SNR values."""
        snr_values = self._scalib_snr.get_snr()
        return snr_values.astype(self.precision)

    @property
    def _distinguisher_str(self):
        return 'SNR_SCALib'


class SNRDistinguisherSCALib(_StandaloneDistinguisher, SNRDistinguisherSCALibMixin):
    """Standalone SNR distinguisher using SCALib's high-performance implementation.

    This class provides the same interface as scared's SNRDistinguisher but uses SCALib's optimized SNR metric internally for better performance.

    Parameters:
        partitions (numpy.ndarray, list, range, or None): Partitions used to categorize traces according to intermediate data values.
            If None, automatically estimated from data at first update (default: None).
        precision (str or numpy.dtype): Precision for computations, should be a float dtype (default: 'float32').

    Examples:
        >>> from scared.scalib import SNRDistinguisherSCALib
        >>> import numpy as np
        >>>
        >>> # Create distinguisher for 256 possible values (0-255)
        >>> snr = SNRDistinguisherSCALib(partitions=range(256))
        >>>
        >>> # Update with traces and labels
        >>> traces = np.random.randn(100, 1000).astype(np.float32)
        >>> labels = np.random.randint(0, 256, (100, 1), dtype=np.uint8)
        >>> snr.update(traces, labels)
        >>>
        >>> # Compute SNR values
        >>> snr_values = snr.compute()
        >>> print(snr_values.shape)  # (1, 1000)

        >>> # Example with non-trivial partitions
        >>> snr2 = SNRDistinguisherSCALib(partitions=range(13, 42))
        >>> labels2 = np.random.randint(13, 42, (100, 2), dtype=np.uint8)
        >>> snr2.update(traces, labels2)
        >>> snr_values2 = snr2.compute()

    Notes:
        - Requires SCALib to be installed: pip install scalib
        - Handles arbitrary partitions by mapping them to 0-indexed classes
        - Results should match SNRDistinguisher within numerical precision
        - SCALib uses optimized C++ implementation for better performance
    """

    def __init__(self, partitions=None, precision='float32'):
        super().__init__(precision=precision)
        _set_partitions(self, partitions=partitions)
