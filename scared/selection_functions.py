import logging
import numpy as _np
import inspect
from ._utils import _is_bytes_array

logger = logging.getLogger(__name__)


class SelectionFunction:
    """Base class for selection function used by analysis framework for intermediate value computation.

    This class is a callable class to validate and decorate the function provided by the user.

    In most of the use cases, user should either use selection function decorators or ready-to-use selection function.

    """

    def __init__(self, function, words=None):
        self._set_words(words)
        if not callable(function):
            raise TypeError(f'function must be a callable, not {type(function)}.')
        self._signature = inspect.signature(function)
        self._function = function
        self._base_kwargs = {}
        self._ref_shape = None

    def _set_words(self, words):
        if words is not None and not isinstance(words, (int, slice, type(...), _np.ndarray)):
            raise TypeError(f'words should be instance of int, slice, Ellipsis or ndarray, not {type(words)}.')

        if isinstance(words, _np.ndarray) and words.dtype.kind not in ('u', 'i'):
            raise TypeError(f'words should be an unsigned integer ndarray, not {words.dtype}.')
        self.words = words if words is not None else ...

    def __call__(self, **kwargs):
        for name, arg in self._signature.parameters.items():
            try:
                self._base_kwargs[name] = kwargs[name]
                self._ref_shape = kwargs[name].shape
            except KeyError:
                if not arg.default:
                    logger.info(f'Arg {name} has no value for this selection function.')

        values = self._function(**self._base_kwargs)
        if values.shape[0] != self._ref_shape[0]:
            raise ValueError(f'Shape of selection function output should be ({self._ref_shape}, -1), not {values.shape}.')
        if self.words is not None:
            try:
                values = values.swapaxes(0, -1)[self.words].swapaxes(0, -1)
            except IndexError:
                raise ValueError(f'Words selection {self.words} can\'t be applied for this selection function with shape {values.shape}.')
        return values


class _AttackSelectionFunction(SelectionFunction):

    def __init__(self, function, guesses, words):
        super().__init__(function=function, words=words)
        _is_bytes_array(guesses)
        self.guesses = guesses
        self._base_kwargs['guesses'] = guesses


def _decorated_selection_function(klass, function, **kwargs):
    def decorated(func):
        return klass(function=func, **kwargs)
    if function is None:
        def decorator(func):
            return decorated(func)
        return decorator
    return decorated(function)


def selection_function(function=None, words=None):
    """Decorator that wraps the provided function as a selection function."""
    return _decorated_selection_function(SelectionFunction, function, words=words)


def attack_selection_function(function=None, guesses=None, words=None):
    """Decorator that wraps provided selection function as an attack selection function.

    Attack selection function must accepts a guesses parameter.

    """
    if guesses is None:
        guesses = _np.arange(256, dtype='uint8')
    return _decorated_selection_function(_AttackSelectionFunction, function, words=words, guesses=guesses)


def reverse_selection_function(function=None, words=None):
    """Decorator that wraps provided selection function as a reverse selection function."""
    return selection_function(function, words=words)
