from . import container as _container
from .utils.misc import _use_parallel
import threading as _th
import numpy as _np
import numba as _nb
import logging as _logging

logger = _logging.getLogger(__name__)
_parallel = _use_parallel()


class TTestContainer:
    """Wrapper container for trace header sets dedicated to TTest analysis.

    Args:
        ths_1, ths_2 (:class:`TraceHeaderSet`): the two trace header set to use for the TTest.

    Attributes:
        containers (list): list of two Container.

    """

    def __init__(self, ths_1, ths_2, frame=None, preprocesses=[]):
        self.containers = [
            _container.Container(ths=ths_1, frame=frame, preprocesses=preprocesses),
            _container.Container(ths=ths_2, frame=frame, preprocesses=preprocesses)
        ]
        if self.containers[0].trace_size != self.containers[1].trace_size:
            raise ValueError(f"Shape of traces must be the same, "
                             f"found {self.containers[0].trace_size} and {self.containers[1].trace_size}")

    def __str__(self):
        return f'''Container for ths_1:
    {self.containers[0]}
Container for ths_2:
    {self.containers[1]}
        '''


class TTestAnalysis:
    """TTest analysis class.

    It provides the processing of a TTest on two trace header sets.
    Leakage detection using a t-test on two trace sets (e.g. fix vs random).
    It is able to detect any first order leakages, without knowledge of leakage function or model.

    It provides an API similar to :class:Analysis`, but simpler.

    Attributes:
        accumulators (list): list containing two instances of :class:`TTestAccumulator`.
        result (:class:`numpy.ndarray`): array containing the result of the t-test with the current state.

    Examples:
        Create your analysis object and run it on a t-test container:

            container = scared.TTestContainer(ths_1, ths_2)
            ttest = scared.TTestAnalysis()
            ttest.run(container)
            ttest.result

    """

    def __init__(self, precision='float32'):
        """Initialize t-test object.

        Args:
            precision (:class:`numpy.dtype`, default=`float32`): precision which will be used for computations.

        """
        self._set_precision(precision)
        self.accumulators = []

    def _set_precision(self, precision):
        try:
            precision = _np.dtype(precision)
        except TypeError:
            raise TypeError(f'precision should be a valid dtype, not {precision}.')

        if precision.kind != 'f':
            raise ValueError(f'precision should be a float dtype, not {precision.kind}.')
        self.precision = precision

    def run(self, ttest_container):
        """Process traces wrapped by `ttest_container` and compute the result.

        Starting from the current state of this instance, the ttest containers are processed by batch.

        Args:
            ttest_container (:class:`TTestContainer`): a :class:`TTestContainer` instance wrapping the trace header sets.

        """
        if not isinstance(ttest_container, TTestContainer):
            raise TypeError(f'ttest_container should be a type TTestContainer, not {type(ttest_container)}.')

        if len(self.accumulators) == 0:
            self.accumulators = [TTestThreadAccumulator(precision=self.precision), TTestThreadAccumulator(precision=self.precision)]
        nb_iterations = len(ttest_container.containers[0].batches()) + len(ttest_container.containers[1].batches())
        logger.info(f'Start run t-test on container {ttest_container}, with {nb_iterations} iterations', {'nb_iterations': nb_iterations})

        for i in range(2):
            self.accumulators[i].start(ttest_container.containers[i])
        try:
            for accu in self.accumulators:
                accu.join()
                accu.compute()
        finally:
            for accu in self.accumulators:
                accu.stop()
                try:
                    accu.join(timeout=1)
                except Exception:
                    pass

        self._compute()

    def _compute(self):
        accu_1, accu_2 = self.accumulators
        self.result = (accu_1.mean - accu_2.mean) / _np.sqrt(accu_1.var / accu_1.processed_traces + accu_2.var / accu_2.processed_traces)

    def __str__(self):
        return 't-Test analysis'


class TTestThreadAccumulator():
    """Accumulator class used for t-test analysis.

    It is a threaded accumulator that will process a complete container. Allows multiple accumulations to be launched in parallel.

    Args:
        precision (np.dtype or str): Data precision (dtype) to use.

    Attributes:
        processed_traces (int): number of traces processed
        sum (:class:`numpy.ndarray`): array containing sum of traces along first axis
        sum_squared (:class:`numpy.ndarray`): array containing sum of squared traces along first axis
        mean (:class:`numpy.ndarray`): array the mean of traces
        var (:class:`numpy.ndarray`): array containing variance of traces

    Methods:
        run(): launch the accumulation on the given Container, in the main thread.
        start(): launch the accumulation on the given Container, in a separated thread.
        join(): wait end of thread processing and check for exception. Reraise if any.
        compute(): computes and stores the values of mean and var for the current values accumulated.
        update(traces): given a traces array, update the sum and sum_squared attributes.

    """

    def __init__(self, precision):
        """Note: No tests are performed on inputs, delegated to parent TTest."""
        self.processed_traces = 0
        self.precision = precision
        self.container = None
        self._exception = None
        self._stop_loop = _th.Event()
        self._thread = None

    def _initialize(self, traces):
        self.sum = _np.zeros(traces.shape[-1], dtype=self.precision)
        self.sum_squared = _np.zeros(traces.shape[-1], dtype=self.precision)

    @staticmethod
    @_nb.njit(parallel=_parallel, nogil=True)
    def _update_core(traces, self_sum, self_sum_squared, precision):
        for i in _nb.prange(traces.shape[1]):
            # New array allocation is the faster way to change both dtype and data alignment.
            tmp = _np.empty(traces.shape[0], dtype=precision)
            tmp[:] = traces[:, i]
            self_sum[i] += tmp.sum()
            self_sum_squared[i] += tmp.T @ tmp

    def update(self, traces):
        if not isinstance(traces, _np.ndarray):
            raise TypeError(f'traces must be numpy ndarray, not {type(traces)}.')

        try:
            self.sum
        except AttributeError:
            self._initialize(traces)

        self._update_core(traces, self.sum, self.sum_squared, _np.dtype(self.precision))
        self.processed_traces += traces.shape[0]

    def stop(self):
        """Inform the thread to stop at next batch processing."""
        self._stop_loop.set()

    def run(self, container=None):
        """Launch the accumulation on the given Container, in the main thread."""
        self._exception = None
        try:
            self.container = container if container is not None else self.container
            if not isinstance(self.container, _container.Container):
                raise ValueError(f'Please give a Container, {type(self.container)} found.')
            self._stop_loop.clear()
            for batch in self.container.batches():
                if self._stop_loop.is_set():
                    return
                samples = batch.samples[:]
                self.update(samples)
                logger.info('t-test iteration finished.')
        except Exception as e:
            self._exception = e
        finally:
            if self._exception and self._thread is None:
                raise self._exception

    def start(self, container):
        """Launch the accumulation on the given Container, in a separated thread."""
        if self._thread is not None and self._thread._initialized and self._thread.is_alive():
            raise RuntimeError('Thread is already running. Use the `join` method before starting again.')
        self.container = container
        self._thread = _th.Thread(target=self.run, daemon=True)
        self._thread.start()

    def join(self):
        """Wait end of thread processing and check for exception. Reraise if any."""
        if self._thread is None:
            raise RuntimeError('No running thread.')
        self._thread.join()
        self._thread = None
        if self._exception is not None:
            raise self._exception

    def compute(self):
        try:
            assert self.processed_traces > 0
        except (AttributeError, AssertionError):
            raise TTestError('TTestAccumulator has not been initialized, or no traces have been processed.\
                Please initialize and update the accumulator before trying to use compute function.')
        self.mean = self.sum / self.processed_traces
        self.var = self.sum_squared / self.processed_traces - self.mean ** 2


class TTestError(Exception):
    pass
