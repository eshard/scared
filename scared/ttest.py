from . import distinguishers
from . import container as _container
import numpy as _np
import logging

logger = logging.getLogger(__name__)


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
        self.accumulators = [TTestAccumulator(precision=precision), TTestAccumulator(precision=precision)]

    def run(self, ttest_container):
        """Process traces wrapped by `ttest_container` and compute the result.

        Starting from the current state of this instance, the ttest containers are processed by batch.

        Args:
            ttest_container (:class:`TTestContainer`): a :class:`TTestContainer`instance wrapping the trace header sets.

        """
        if not isinstance(ttest_container, TTestContainer):
            raise TypeError(f'ttest_container should be a type TTestContainer, not {type(ttest_container)}.')

        nb_iterations = sum([max(int(len(cont._ths) / cont.batch_size), 1) for cont in ttest_container.containers])
        logger.info(f'Start run t-test on container {ttest_container}, with {nb_iterations} iterations', {'nb_iterations': nb_iterations})
        for i in range(2):
            container = ttest_container.containers[i]
            logger.info(f'Start processing t-test on ths number {i}.')
            for batch in container.batches():
                self.accumulators[i].update(batch.samples)
                logger.info(f't-test iteration finished.')
            self.accumulators[i].compute()

        self._compute()

    def _compute(self):
        accu_1, accu_2 = self.accumulators
        self.result = (accu_1.mean - accu_2.mean) / _np.sqrt(accu_1.var / accu_1.processed_traces + accu_2.var / accu_2.processed_traces)

    def __str__(self):
        return 't-Test analysis'


class TTestAccumulator:
    """Accumulator class used for t-test analysis.

    Attributes:
        - processed_traces (int): number of traces processed
        - sum (:class:`numpy.ndarray): array containing sum of traces along first axis
        - sum_squared (:class:`numpy.ndarray`): array containing sum of squared traces along first axis
        - mean (:class:`numpy.ndarray`): array the mean of traces
        - var (:class:`numpy.ndarray`): array containing variance of traces

    Methods:
        - update(traces): given a traces array, update the sum and sum_squared attributes
        - compute(): computes and stores the values of mean and var for the current values accumulated.

    """

    def __init__(self, precision):
        distinguishers._initialize_distinguisher(self, precision=precision, processed_traces=0)

    def _initialize(self, traces):
        self.sum = _np.zeros(traces.shape[-1], dtype=self.precision)
        self.sum_squared = _np.zeros(traces.shape[-1], dtype=self.precision)

    def update(self, traces):
        if not isinstance(traces, _np.ndarray):
            raise TypeError(f'traces must be numpy ndarray, not {type(traces)}.')

        traces = traces.astype(self.precision)

        try:
            self.sum
        except AttributeError:
            self._initialize(traces)

        self.processed_traces += traces.shape[0]
        self.sum += _np.sum(traces, axis=0)
        self.sum_squared += _np.sum(traces ** 2, axis=0)

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
