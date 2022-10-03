import numpy as _np
from ..first_order import CenterOn
from ._base import _combination


def _product(el1, el2):
    return (el1 * el2)


def _difference(el1, el2):
    return (el1 - el2)


def _centered(function, mean, precision):
    return lambda traces: function(CenterOn(mean=mean, precision=precision)(traces))


def _absolute(function):
    return lambda traces: _np.absolute(function(traces))


class Difference:
    """Difference combination preprocess for high-order side-channel analysis.

    Args:
        frame_1 (slice or iterable, default=...): first traces frame that will be taken.
        frame_2 (slice or iterable, default=None): second optional traces frame that will be taken.
        mode (str, default='full'): Combination mode either `'full'` or `'same'`.
            In `'same'` mode, each time-sample of `frame_1` will be combined with the corresponding time-sample in `frame_2`.
            When using this mode, the two frames needs to be provided and of the same length.
            In `'full'` mode, each point of `frame_1` is combined with full `frame_2` if it is provided;
            otherwise, if `distance` is None, each point of `frame_1` is combined with the following points until the end of `frame_1`;
            else with a subframe starting at the current point position in `frame_1` and of size equals to `distance`.
        distance (integer, default=None): size of the frame to combine with each point of `frame_1`. This parameter is not available if `frame_2` is provided.
        precision (numpy.dtype, default='float32'): optional parameter to define minimum numerical precision used to perform computation.
            If input data has higher precision, it will be kept instead.

    """

    def __new__(cls, frame_1=..., frame_2=None, mode='full', distance=None, precision='float32'):
        return _combination(
            _difference, frame_1=frame_1, frame_2=frame_2, mode=mode, distance=distance, precision=precision
        )


class Product:
    """Product combination preprocess for high-order side-channel analysis.

    Args:
        frame_1 (slice or iterable, default=...): first traces frame that will be taken.
        frame_2 (slice or iterable, default=None): second optional traces frame that will be taken.
        mode (str, default='full'): Combination mode either `'full'` or `'same'`.
            In `'same'` mode, each time-sample of `frame_1` will be combined with the corresponding time-sample in `frame_2`.
            When using this mode, the two frames needs to be provided and of the same length.
            In `'full'` mode, each point of `frame_1` is combined with full `frame_2` if it is provided;
            otherwise, if `distance` is None, each point of `frame_1` is combined with the following points until the end of `frame_1`;
            else with a subframe starting at the current point position in `frame_1` and of size equals to `distance`.
        distance (integer, default=None): size of the frame to combine with each point of `frame_1`. This parameter is not available if `frame_2` is provided.
        precision (numpy.dtype, default='float32'): optional parameter to define minimum numerical precision used to perform computation.
            If input data has higher precision, it will be kept instead.

    """

    def __new__(cls, frame_1=..., frame_2=None, mode='full', distance=None, precision='float32'):
        return _combination(
            _product, frame_1=frame_1, frame_2=frame_2, mode=mode, distance=distance, precision=precision
        )


class CenteredProduct(Product):
    """Centered product combination preprocess for high-order side-channel analysis.

    Args:
        frame_1 (slice or iterable, default=...): first traces frame that will be taken.
        frame_2 (slice or iterable, default=None): second optional traces frame that will be taken.
        mode (str, default='full'): Combination mode either `'full'` or `'same'`.
            In `'same'` mode, each time-sample of `frame_1` will be combined with the corresponding time-sample in `frame_2`.
            When using this mode, the two frames needs to be provided and of the same length.
            In `'full'` mode, each point of `frame_1` is combined with full `frame_2` if it is provided;
            otherwise, if `distance` is None, each point of `frame_1` is combined with the following points until the end of `frame_1`;
            else with a subframe starting at the current point position in `frame_1` and of size equals to `distance`.
        distance (integer, default=None): size of the frame to combine with each point of `frame_1`. This parameter is not available if `frame_2` is provided.
        mean (numpy.ndarray, default=None): a mean array with compatible size with traces. If None, the mean of provided batch of traces is computed.
        precision (numpy.dtype, default='float32'): optional parameter to define minimum numerical precision used to perform computation.
            If input data has higher precision, it will be kept instead.

    """

    def __new__(cls, frame_1=..., frame_2=None, mode='full', distance=None, mean=None, precision='float32'):
        return _centered(
            _combination(
                _product, frame_1=frame_1, frame_2=frame_2, mode=mode, distance=distance, precision=precision
            ), mean=mean, precision=precision)


class AbsoluteDifference:
    """Absolute difference combination preprocess for high-order side-channel analysis.

    Args:
        frame_1 (slice or iterable, default=...): first traces frame that will be taken.
        frame_2 (slice or iterable, default=None): second optional traces frame that will be taken.
        mode (str, default='full'): Combination mode either `'full'` or `'same'`.
            In `'same'` mode, each time-sample of `frame_1` will be combined with the corresponding time-sample in `frame_2`.
            When using this mode, the two frames needs to be provided and of the same length.
            In `'full'` mode, each point of `frame_1` is combined with full `frame_2` if it is provided;
            otherwise, if `distance` is None, each point of `frame_1` is combined with the following points until the end of `frame_1`;
            else with a subframe starting at the current point position in `frame_1` and of size equals to `distance`.
        distance (integer, default=None): size of the frame to combine with each point of `frame_1`. This parameter is not available if `frame_2` is provided.
        precision (numpy.dtype, default='float32'): optional parameter to define minimum numerical precision used to perform computation.
            If input data has higher precision, it will be kept instead.

    """

    def __new__(cls, frame_1=..., frame_2=None, mode='full', distance=None, precision='float32'):
        return _absolute(
            _combination(_difference, frame_1=frame_1, frame_2=frame_2, mode=mode, distance=distance, precision=precision)
        )
