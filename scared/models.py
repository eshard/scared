import numpy as _np
import abc
import numba

_HW_LUT = _np.array([0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
                     1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
                     1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
                     2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
                     1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
                     2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
                     2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
                     3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8], dtype='uint32')


@numba.vectorize([numba.uint32(numba.uint8)])
def _fhw8(x):
    return _HW_LUT[x]


@numba.vectorize([numba.uint32(numba.uint16)])
def _fhw16(x):
    return _HW_LUT[x & 0x00ff] + _HW_LUT[x >> 8]


@numba.vectorize([numba.uint32(numba.uint32)])
def _fhw32(x):
    r = 0
    for _ in range(4):
        r += _HW_LUT[x & 0x000000ff]
        x >>= 8
    return r


@numba.vectorize([numba.uint32(numba.uint64)])
def _fhw64(x):
    r = 0
    for _ in range(8):
        r += _HW_LUT[x & 0x00000000000000ff]
        x >>= 8
    return r


_hw_functions_list = dict([(1, _fhw8), (2, _fhw16), (4, _fhw32), (8, _fhw64)])


class Model(abc.ABC):
    """Leakage model callable abstract class.

    Use this abstract class to implement your own leakage function. Subclass it and define a _compute method which
    take a data argument.

    _compute function must returns a numpy array with all dimensions preserved except the last.

    See implementations of Value, HammingWeight or Monobit model for examples.

    """

    def __call__(self, data, axis=-1):
        if not isinstance(data, _np.ndarray):
            raise TypeError(f'Model should take ndarray as input data, not {type(data)}.')
        if data.dtype.kind not in ('b', 'i', 'u', 'f', 'c'):
            raise ValueError(f'Model should take numerical ndarray as input data, not {data.dtype}).')

        if axis == -1:
            axis = len(data.shape) - 1

        results = self._compute(data, axis=axis)

        check_shape = [d for i, d in enumerate(results.shape) if i != axis]
        origin_shape = [d for i, d in enumerate(data.shape) if i != axis]
        if check_shape != origin_shape:
            raise ValueError(f'Model instance {self.__class__} does not preserve dimensions of data properly on call.')
        return results

    @abc.abstractmethod
    def _compute(self, data, axis):
        pass

    @property
    @abc.abstractmethod
    def max_data_value(self):
        pass


class Value(Model):
    """Value leakage model class.

    Instances of this class are callables which takes a data numpy array as input and returns it unchanged.

    Args:
        data (numpy.ndarray): numeric numpy ndarray

    Returns:
        (numpy.ndarray): unchanged input data numpy ndarray.

    """

    def _compute(self, data, axis):
        return data

    @property
    def max_data_value(self):
        return 256

    def __str__(self):
        return 'Value'


class Monobit(Model):
    """Monobit model leakage class.

    Instances of this class are callables which takes a data numpy array as input and
    returns the monobit model value computed on last dimension of the array.

    Attributes:
        bit (int): number of the bit targeted. Should be between 0 and 8, otherwise raises a ValueError.

    Args:
        data (numpy.ndarray): a ndarray of numeric type

    Returns:
        (numpy.ndarray) an ndarray of the same shape as data, with the result monobit mask applied.

    """

    def __init__(self, bit):
        if not isinstance(bit, int):
            raise TypeError(f'bit target should be an int, not {type(bit)}.')
        if bit < 0 or bit > 8:
            raise ValueError(f'bit should be between 0 and 8, not {bit}.')
        self.bit = bit

    def _compute(self, data, axis):
        return (_np.bitwise_and(data, 2 ** self.bit) > 0).astype('uint8')

    @property
    def max_data_value(self):
        return 1

    def __str__(self):
        return f'Monobit {self.bit}'


class HammingWeight(Model):
    """Hamming weight leakage model for unsigned integer arrays.

    Instances of this class are callables which takes a data numpy array as input and returns the
    Hamming Weight values computed on the last dimension of the array, and on a number of words
    defined at instantiation.

    Attributes:
        nb_words (int, default=1): number of words on which to compute the hamming weight.
        expected_dtype(numpy.dtype, default='uint8'): expected dtype of input data.

    Args:
        data (numpy.ndarray): a unsigned integer ndarray.

    Returns:
        (numpy.ndarray) an ndarray with hamming weight computed on last dimension.
            Every dimensions but the last are preserved.

    """

    def __init__(self, nb_words=1, expected_dtype='uint8'):
        if not isinstance(nb_words, int):
            raise TypeError(f'nb_words should be an integer, not {nb_words}.')
        if nb_words <= 0:
            raise ValueError(f'nb_words must be strictly greater than 0, not {nb_words}.')
        try:
            expected_dtype = _np.dtype(expected_dtype)
        except TypeError:
            raise ValueError(f'{expected_dtype} is not a valid dtype.')
        if expected_dtype.kind != 'u':
            raise ValueError(f'`expected_dtype` should be an unsigned integer dtype, not {expected_dtype}).')
        self.nb_words = nb_words
        self.expected_dtype = expected_dtype

    def _compute(self, data, axis):
        if data.dtype.kind != 'u':
            raise ValueError(f'HammingWeight should take unsigned integer data as input, not {data.dtype}).')

        if data.dtype != self.expected_dtype:
            raise ValueError(f'Expected dtype for HammingWeight input data is {self.expected_dtype}, not {data.dtype}.')

        if data.shape[axis] < self.nb_words:
            raise ValueError(f'data should have at least {self.nb_words} as dimension with index {axis}, not {data.shape[axis]}.')

        result_data = _hw_functions_list[data.dtype.itemsize](data)
        if self.nb_words > 1:
            final_w_dimension = data.shape[axis] // self.nb_words
            final_shape = [d if i != axis else final_w_dimension for i, d in enumerate(data.shape)]
            result = _np.zeros(final_shape, dtype='uint32').swapaxes(0, axis)
            result_data = _np.swapaxes(result_data, 0, axis)
            for i in range(result.shape[0]):
                slices = result_data[i * self.nb_words: (i + 1) * self.nb_words]
                result[i] = _np.sum(slices, axis=0)
            result_data = result
            result_data = _np.swapaxes(result, 0, axis)

        return result_data

    @property
    def max_data_value(self):
        return self.nb_words * self.expected_dtype.itemsize * 8

    def __str__(self):
        return f'Hamming Weight on {self.nb_words} word(s).'
