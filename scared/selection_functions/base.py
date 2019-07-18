import logging
import numpy as _np
import inspect
from .._utils import _is_bytes_array

logger = logging.getLogger(__name__)


class SelectionFunctionError(Exception):
    pass


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
        if words is not None and not isinstance(words, (int, slice, type(...), list, _np.ndarray)):
            raise TypeError(f'words should be instance of int, slice, Ellipsis, list or ndarray, not {type(words)}.')

        if isinstance(words, list):
            words = _np.array(words, dtype='uint8')

        if isinstance(words, _np.ndarray) and words.dtype.kind not in ('u', 'i'):
            raise TypeError(f'words should be an unsigned integer ndarray, not {words.dtype}.')
        self.words = words if words is not None else ...

    def __call__(self, **kwargs):
        for name, arg in self._signature.parameters.items():
            try:
                self._base_kwargs[name] = kwargs[name]
                self._ref_shape = kwargs[name].shape
            except KeyError as e:
                if name not in self._base_kwargs:
                    raise SelectionFunctionError(f'Missing values in metadata {list(kwargs.keys())} for expected argument {e} of selection function {self}.')

        values = self._function(**self._base_kwargs)
        if values.shape[0] != self._ref_shape[0]:
            raise SelectionFunctionError(f'Shape of selection function output should begin with {self._ref_shape[0]}, not {values.shape[0]}.')
        if self.words is not None:
            try:
                values = values.swapaxes(0, -1)[self.words].swapaxes(0, -1)
            except IndexError:
                raise SelectionFunctionError(f'Words selection {self.words} can\'t be applied for this selection function with shape {values.shape}.')
        return values

    @property
    def _words_str(self):
        if not isinstance(self.words, Ellipsis.__class__):
            return str(self.words)
        return 'All'

    def __str__(self):
        template_str = f'''Selection function:
        Function             : {self._function.__name__}
        Function args        : {list(self._signature.parameters.keys())}
        Words selection      : {self._words_str}
        '''
        return template_str


class _AttackSelectionFunction(SelectionFunction):

    def __init__(self, function, guesses, words, expected_key_function=None):
        super().__init__(function=function, words=words)
        if isinstance(guesses, range):
            guesses = _np.array(guesses, dtype='uint8')
        _is_bytes_array(guesses)
        self.guesses = guesses
        self._base_kwargs['guesses'] = guesses

        if expected_key_function is not None and not callable(expected_key_function):
            raise TypeError(f'Expected key function must be a callable, not {type(expected_key_function)}.')
        self.expected_key_function = expected_key_function

    def compute_expected_key(self, **kwargs):
        if self.expected_key_function:
            kargs = {}
            for name, arg in inspect.signature(self.expected_key_function).parameters.items():
                try:
                    kargs[name] = kwargs[name]
                except KeyError as e:
                    raise SelectionFunctionError(
                        f'Missing key values in metadata {list(kwargs.keys())} for expected argument {e} of compute expected function {self}.'
                    )

            return self.expected_key_function(**kargs)

    @property
    def _guesses_str(self):
        return f'{str(self.guesses)[:20]} ... {str(self.guesses)[-20:]}'.replace('\n', '')

    def __str__(self):
        res = super().__str__()
        res += f'''Guesses              : {self._guesses_str}
        Expected key function: {self.expected_key_function.__name__ if self.expected_key_function else '-'}
        Expected key args    : {list(inspect.signature(self.expected_key_function).parameters.keys()) if self.expected_key_function else '-'}
        '''
        return res


def _decorated_selection_function(klass, function, **kwargs):
    def decorated(func):
        return klass(function=func, **kwargs)
    if function is None:
        def decorator(func):
            return decorated(func)
        return decorator
    return decorated(function)


def selection_function(function=None, words=None):
    """Decorator that wraps the provided function as a selection function.

    Args:
        function (callable): the attack selection function callable.
        words (ndarray, slice, list, default=None): words subselection used by the selection function.

    """
    return _decorated_selection_function(SelectionFunction, function, words=words)


def attack_selection_function(function=None, guesses=range(256), words=None, expected_key_function=None):
    """Decorator that wraps provided selection function as an attack selection function.

    Attack selection function must accepts a guesses parameter.

    Args:
        function (callable): the attack selection function callable.
        guesses (ndarray or range, default=range(256)): guesses values to be used by the selection function.
        words (ndarray, slice, list, default=None): words subselection used by the selection function.
        expected_key_function (callable, default=None): callable to compute the corresponding expected key value for this selection function.

    Methods:
        compute_expected_key: returns the result of expected_key_function, if available.

    """
    return _decorated_selection_function(_AttackSelectionFunction, function, words=words, guesses=guesses, expected_key_function=expected_key_function)


def reverse_selection_function(function=None, words=None):
    """Decorator that wraps provided selection function as a reverse selection function.

    Args:
        function (callable): the attack selection function callable.
        words (ndarray, slice, list, default=None): words subselection used by the selection function.

    """
    return selection_function(function, words=words)


class _AttackSelectionFunctionWrapped(_AttackSelectionFunction):

    def __init__(self, function, guesses, words, expected_key_function=None, target_tag=None, key_tag=None, target_name='data', key_name='key'):
        super().__init__(function=function, words=words, guesses=guesses, expected_key_function=expected_key_function)
        self.target_tag = target_tag
        self.target_name = target_name
        self.key_name = key_name
        self.key_tag = key_tag

    def __call__(self, **kwargs):
        kwargs[self.target_name] = kwargs[self.target_tag]
        return super().__call__(**kwargs)

    def compute_expected_key(self, **kwargs):
        kwargs[self.key_name] = kwargs[self.key_tag]
        return super().compute_expected_key(**kwargs)
