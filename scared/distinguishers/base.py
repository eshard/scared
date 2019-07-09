import abc
import numpy as _np
import logging

logger = logging.getLogger(__name__)


class DistinguisherError(Exception):
    pass


class DistinguisherMixin(abc.ABC):
    """Abstract distinguisher mixin class.

    A distinguisher mixin is used either as a component of :class:`BaseAnalysis` child, or as a standalone object by subclassing it directly.
    It defines the distinguisher interface, which has two public methods:

        - `update` takes traces and intermediate values data and update the state of the distinguisher.
        - `compute` returns the result of the distinguisher applied on the current state

    This abstract class defines three private methods that must be implemented by each distinguisher mixin:
        - `_initialize` which is called on first `update` call and can be leverage to complete process init.
        - `_update` which implements the specific of the distinguisher update process.
        - `_compute` which implements the specific of the distinguisher compute process.

    """

    def update(self, traces, data):
        if not isinstance(traces, _np.ndarray):
            raise TypeError(f'traces must be numpy ndarray, not {type(traces)}.')
        if not isinstance(data, _np.ndarray):
            raise TypeError(f'data must be numpy ndarray, not {type(data)}.')

        if traces.shape[0] != data.shape[0]:
            raise ValueError(f'traces and data must have the same first dimension, not {traces.shape[0]} for traces and {data.shape[0]} for data.')

        logger.info(f'Start update of distinguisher {self.__class__.__name__} with traces {traces.shape} and data {data.shape}.')
        o_shape = data.shape
        data = data.reshape((o_shape[0], -1))
        try:
            self._origin_shape
        except AttributeError:
            logger.info(f'Initialize distinguisher state.')
            self._origin_shape = o_shape
            self._initialize(traces=traces, data=data)

        self.processed_traces += traces.shape[0]
        logger.info(f'Will call _update traces.')
        self._update(traces=traces, data=data)

    @abc.abstractmethod
    def _initialize(self, traces, data):
        pass

    @abc.abstractmethod
    def _update(self, traces, data):
        pass

    def compute(self):
        try:
            assert self.processed_traces > 0
        except (AttributeError, AssertionError):
            raise DistinguisherError('Distinguisher has not been initialized, or no traces have been processed.\
                Please initialize and update the distinguisher before trying to use compute function.')
        return self._compute().reshape((self._origin_shape[1:] + (-1, )))

    @abc.abstractmethod
    def _compute(self):
        pass


def _set_precision(obj, precision):
    try:
        precision = _np.dtype(precision)
    except TypeError:
        raise TypeError(f'precision should be a valid dtype, not {precision}.')

    if precision.kind != 'f':
        raise ValueError(f'precision should be a float dtype, not {precision.kind}.')
    obj.precision = precision


def _initialize_distinguisher(obj, precision, processed_traces):
    _set_precision(obj, precision)
    obj.processed_traces = processed_traces


class _StandaloneDistinguisher:

    def __init__(self, precision='float32'):
        _initialize_distinguisher(self, precision, processed_traces=0)


class Distinguisher(_StandaloneDistinguisher, DistinguisherMixin):
    """Abstract stand-alone distinguisher class.

    This class must be inherited and implements the distinguisher methods.
    """
