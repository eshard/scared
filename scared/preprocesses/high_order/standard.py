from .._base import PreprocessError
from .. import first_order
from ._base import _CombinationFrameOnDistance, _CombinationOfTwoFrames, _CombinationPointToPoint
import numpy as _np


def _product(el1, el2):
    return (el1 * el2)


def _difference(el1, el2):
    return (el1 - el2)


def _centered(function, mean):
    def _(traces):
        try:
            _mean = _np.nanmean(traces, axis=0) if mean is None else mean
        except IndexError:
            raise TypeError(f'traces must be a 2 dimensionnal numpy ndarray, not {type(traces)}.')
        return function(first_order._center(traces, _mean))
    return _


def _absolute(function):
    def _(traces):
        return _np.absolute(function(traces))
    return _


def _combination(operation, frame_1, frame_2=None, mode='full', distance=None):
    if mode not in ['same', 'full']:
        raise PreprocessError(f'Only same or full mode are available for combination preprocesses.')
    if distance is not None and (mode == 'same' or frame_2 is not None):
        raise PreprocessError(f'same mode or usage of two frames is incompatible with use of distance.')
    if mode == 'same' and frame_2 is None:
        raise PreprocessError(f'same mode requires two frames.')

    if distance is not None:
        res = _CombinationFrameOnDistance(frame_1=frame_1, distance=distance)
    elif mode == 'same' and frame_2 is not None:
        res = _CombinationPointToPoint(frame_1=frame_1, frame_2=frame_2)
    else:
        res = _CombinationOfTwoFrames(frame_1=frame_1, frame_2=frame_2)
    res._operation = operation
    return res


class Difference:
    """Difference combination preprocess for High Order analysis.

    Args:
        frame_1 (slice or iterable, default=...): first traces frame that will be taken.
        frame_2 (slice or iterable, default=None): second optionnal traces frame that will be taken.
        mode (str, default='full'): Combination mode between `'full'` and `'same'` values.
            In `'same'` mode, each point of `frame_1` will be combined with its corresponding point in `frame_2`.
            The two frames needs to be provided and of the same length when using this mode.
            In `'full'` mode, each point of `frame_1` is combined with full `frame_2` if it provided,
            otherwise with the frame between the current point position in `frame_1` and the end of the frame if `distance` is None,
            else with a subframe starting at the current point position in `frame_1` and of size equals to `distance`.
        dist (integer, default=None): size of the frame to combine with each point of `frame_1`. This parameter is not available if `frame_2` is provided.

    """

    def __new__(cls, frame_1=..., frame_2=None, mode='full', distance=None):
        return _combination(
            _difference, frame_1=frame_1, frame_2=frame_2, mode=mode, distance=distance
        )


class Product:
    """Product combination preprocess for High Order analysis.

    Args:
        frame_1 (slice or iterable, default=...): first traces frame that will be taken.
        frame_2 (slice or iterable, default=None): second optionnal traces frame that will be taken.
        mode (str, default='full'): Combination mode between `'full'` and `'same'` values.
            In `'same'` mode, each point of `frame_1` will be combined with its corresponding point in `frame_2`.
            The two frames needs to be provided and of the same length when using this mode.
            In `'full'` mode, each point of `frame_1` is combined with full `frame_2` if it provided,
            otherwise with the frame between the current point position in `frame_1` and the end of the frame if `distance` is None,
            else with a subframe starting at the current point position in `frame_1` and of size equals to `distance`.
        dist (integer, default=None): size of the frame to combine with each point of `frame_1`. This parameter is not available if `frame_2` is provided.

    """

    def __new__(cls, frame_1=..., frame_2=None, mode='full', distance=None):
        return _combination(
            _product, frame_1=frame_1, frame_2=frame_2, mode=mode, distance=distance
        )


class CenteredProduct(Product):
    """Centered prodiuct combination preprocess for High Order analysis.

    Args:
        frame_1 (slice or iterable, default=...): first traces frame that will be taken.
        frame_2 (slice or iterable, default=None): second optionnal traces frame that will be taken.
        mode (str, default='full'): Combination mode between `'full'` and `'same'` values.
            In `'same'` mode, each point of `frame_1` will be combined with its corresponding point in `frame_2`.
            The two frames needs to be provided and of the same length when using this mode.
            In `'full'` mode, each point of `frame_1` is combined with full `frame_2` if it provided,
            otherwise with the frame between the current point position in `frame_1` and the end of the frame if `distance` is None,
            else with a subframe starting at the current point position in `frame_1` and of size equals to `distance`.
        dist (integer, default=None): size of the frame to combine with each point of `frame_1`. This parameter is not available if `frame_2` is provided.
        mean (numpy.ndarray, default=None): a mean array with compatible size with traces. If it None, the mean of provided traces is computed.
    """

    def __new__(cls, frame_1=..., frame_2=None, mode='full', distance=None, mean=None):
        return _centered(
            _combination(
                _product, frame_1=frame_1, frame_2=frame_2, mode=mode, distance=distance
            ), mean=mean)


class AbsoluteDifference:
    """Absolute difference combination preprocess for High Order analysis.

    Args:
        frame_1 (slice or iterable, default=...): first traces frame that will be taken.
        frame_2 (slice or iterable, default=None): second optionnal traces frame that will be taken.
        mode (str, default='full'): Combination mode between `'full'` and `'same'` values.
            In `'same'` mode, each point of `frame_1` will be combined with its corresponding point in `frame_2`.
            The two frames needs to be provided and of the same length when using this mode.
            In `'full'` mode, each point of `frame_1` is combined with full `frame_2` if it provided,
            otherwise with the frame between the current point position in `frame_1` and the end of the frame if `distance` is None,
            else with a subframe starting at the current point position in `frame_1` and of size equals to `distance`.
        dist (integer, default=None): size of the frame to combine with each point of `frame_1`. This parameter is not available if `frame_2` is provided.

    """

    def __new__(cls, frame_1=..., frame_2=None, mode='full', distance=None):
        return _absolute(
            _combination(_difference, frame_1=frame_1, frame_2=frame_2, mode=mode, distance=distance)
        )
