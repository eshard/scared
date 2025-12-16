"""SCALib-based TTest implementation for scared."""

import numpy as _np
from ..ttest import TTestContainer as _TTestContainer
from ..utils.fast_astype import fast_astype as _fast_astype

try:
    from scalib.metrics import Ttest as _SCALibTtest  # noqa: N811
    SCALIB_AVAILABLE = True
except ImportError:
    SCALIB_AVAILABLE = False


class TTestAnalysisSCALib:
    """T-test analysis using SCALib's high-performance implementation.

    This class wraps SCALib's Ttest to provide compatibility with scared's TTestAnalysis API.
    It provides leakage detection using Welch's t-test on two trace sets (e.g., fixed vs random),
    detecting first-order or higher-order leakages without knowledge of the leakage function.

    SCALib's implementation uses optimized C++/Rust backend for significantly faster computation
    compared to the standard scared TTestAnalysis.

    Args:
        order (int, optional): Statistical order of the t-test. Use order=1 for standard first-order
            t-test (Welch's t-test), order=2 for second-order, etc. Higher orders test for higher-order
            statistical moments. Defaults to 1.

    Attributes:
        result (numpy.ndarray): Array containing the t-test statistic for each time sample.
            Shape is (n_samples,) where n_samples is the number of points per trace.
            Available after calling run(). Always float64 dtype (SCALib native precision).
        result_all_orders (numpy.ndarray): Array containing t-test statistics for all orders.
            Shape is (order, n_samples). For example, if order=3, result_all_orders[0] contains
            the 1st order t-test, result_all_orders[1] contains the 2nd order, and result_all_orders[2]
            contains the 3rd order. The result property returns result_all_orders[order-1].
            Available after calling run().
        order (int): The statistical order of the t-test.

    Raises:
        ImportError: If SCALib is not installed.

    Examples:
        Basic usage with two trace sets:

            >>> from scared.scalib import TTestAnalysisSCALib
            >>> from scared import TTestContainer
            >>> container = TTestContainer(ths_1, ths_2)
            >>> ttest = TTestAnalysisSCALib()
            >>> ttest.run(container)
            >>> print(ttest.result)

        Higher-order t-test:

            >>> ttest_2nd = TTestAnalysisSCALib(order=2)
            >>> ttest_2nd.run(container)

    Notes:
        - SCALib requires traces to be int16 dtype. Automatic conversion is performed if needed.
        - SCALib always computes in float64 precision.
        - The implementation processes both trace sets batch-by-batch to avoid loading all
          traces into memory at once.
        - For order > 1, SCALib automatically computes all lower-order t-tests as well,
          but only the highest order result is returned by this wrapper.

    """

    def __init__(self, order=1):
        """Initialize t-test analysis with SCALib backend.

        Args:
            order (int, optional): Statistical order of the t-test. Defaults to 1.

        Raises:
            ImportError: If SCALib is not installed.
            TypeError: If order is not an integer.
            ValueError: If order is not a positive integer.

        """
        if not SCALIB_AVAILABLE:
            raise ImportError('SCALib is not installed. Please install it to use TTestAnalysisSCALib.')

        self._set_order(order)
        self._scalib_ttest = None
        self.result_all_orders = None

    def _set_order(self, order):
        """Validate and set order parameter."""
        if not isinstance(order, (int, _np.integer)):
            raise TypeError(f'order should be an integer, not {type(order)}.')
        if order < 1:
            raise ValueError(f'order should be a positive integer, not {order}.')
        self.order = int(order)

    def run(self, ttest_container):
        """Process traces from ttest_container and compute the t-test statistic.

        This method processes traces from both groups in the TTestContainer batch-by-batch,
        converting them to the format required by SCALib and accumulating statistics.

        Args:
            ttest_container (TTestContainer): Container wrapping two TraceHeaderSet objects.

        Raises:
            TypeError: If ttest_container is not a TTestContainer instance.
            ValueError: If the two trace sets have different trace lengths.

        """
        if not isinstance(ttest_container, _TTestContainer):
            raise TypeError(f'ttest_container should be a TTestContainer, not {type(ttest_container)}.')

        container_1, container_2 = ttest_container.containers
        self._scalib_ttest = _SCALibTtest(d=self.order)

        batches_1 = list(container_1.batches())
        batches_2 = list(container_2.batches())

        for batch_1, batch_2 in zip(batches_1, batches_2):
            traces_1 = batch_1.samples[:]
            traces_2 = batch_2.samples[:]

            traces_1_int16 = _fast_astype(traces_1, dtype='int16', order='C')
            traces_2_int16 = _fast_astype(traces_2, dtype='int16', order='C')

            labels_1 = _np.zeros(traces_1_int16.shape[0], dtype=_np.uint16)
            labels_2 = _np.ones(traces_2_int16.shape[0], dtype=_np.uint16)

            self._scalib_ttest.fit_u(traces_1_int16, labels_1)
            self._scalib_ttest.fit_u(traces_2_int16, labels_2)

        if len(batches_1) < len(batches_2):
            for batch_2 in batches_2[len(batches_1):]:
                traces_2 = batch_2.samples[:]
                traces_2_int16 = _fast_astype(traces_2, dtype='int16', order='C')
                labels_2 = _np.ones(traces_2_int16.shape[0], dtype=_np.uint16)
                self._scalib_ttest.fit_u(traces_2_int16, labels_2)
        elif len(batches_2) < len(batches_1):
            for batch_1 in batches_1[len(batches_2):]:
                traces_1 = batch_1.samples[:]
                traces_1_int16 = _fast_astype(traces_1, dtype='int16', order='C')
                labels_1 = _np.zeros(traces_1_int16.shape[0], dtype=_np.uint16)
                self._scalib_ttest.fit_u(traces_1_int16, labels_1)

        self._compute()

    def _compute(self):
        """Compute and store the final t-test result."""
        self.result_all_orders = self._scalib_ttest.get_ttest()

    @property
    def result(self):
        """Get the t-test statistic result.

        Returns:
            numpy.ndarray: T-test statistic for each time sample. Shape is (n_samples,).

        """
        return self.result_all_orders[self.order - 1]

    def __str__(self):
        return f't-Test analysis (SCALib, order={self.order})'
