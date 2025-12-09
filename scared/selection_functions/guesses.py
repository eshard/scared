from __future__ import annotations
from typing import Union, Callable
import numpy as _np
import copy as _copy
import numba as _nb


class Guesses(_np.ndarray):
    def __new__(cls, guess_list: Union[range, _np.ndarray, Guesses], dtype: Union[None, _np.dtype] = None, nb_words: int = 1) -> Guesses:
        """Instantiates a Guesses object.

        This class checks the type of given guesses. Additionally, if guesses for multiple words must be done (by specifying nb_words > 1),
        it performs a cartesian product between the guesses of each word.

        If no dtype is given, it will be inferred with the smallest possible precision.

        Args:
            guess_list (Union[range, numpy.ndarray, Guesses]): Guesses to validate.
            dtype (Union[None, numpy.dtype], optional): dtype of the guesses. Defaults to None.
            nb_words (int, optional): Number of words for a guess. Defaults to None.

        Raises:
            TypeError: Raised if the type of the inputs is not valid.
            ValueError: If dimensions of a given numpy.ndarray are not valid or nb_words is less than 1.

        Returns:
            Guesses: The instantiated object.

        """
        # Guards nb_words
        if not isinstance(nb_words, int):
            raise TypeError("nb_words must be int.")
        if nb_words < 1:
            raise ValueError("Invalid number of words.")

        obj = guess_list

        if isinstance(obj, Guesses):
            return _copy.deepcopy(obj)

        if isinstance(obj, range):
            obj = _np.array(obj, dtype=dtype)
            cls._verify_type(obj)
        else:
            cls._verify_type(obj)
            if obj.ndim > 2:
                raise ValueError(f"Numpy array given contains more than two dimensions. Current dimensions: {obj.ndim}.")
            if dtype is not None:
                obj = obj.astype(dtype)

        if dtype is None:
            # Here we infer the dtype and we use it for memory efficiency reasons.
            min_val = _np.min(obj)
            max_val = _np.max(obj)

            if min_val >= 0:
                # Unsigned type
                bits = _np.ceil(_np.log2(max_val + 1))
                candidates = [8, 16, 32, 64]
                dtype_inferred = _np.dtype(f'uint{next(b for b in candidates if bits <= b)}')
            else:
                # Signed type
                bound = max(abs(min_val), abs(max_val))
                bits = _np.ceil(_np.log2(bound + 1)) + 1  # +1 for sign bit
                candidates = [8, 16, 32, 64]
                dtype_inferred = _np.dtype(f'int{next(b for b in candidates if bits <= b)}')
            obj = obj.astype(dtype_inferred)

        # Here we expand to multiple words if nb_words is not 1 and current guess array has one dimension.
        if nb_words > 1 and obj.ndim == 1:
            obj = [obj] * nb_words
            obj = _np.stack(_np.meshgrid(*obj, indexing='ij'), axis=-1).reshape(-1, nb_words)

        obj = _np.asarray(obj).view(cls)  # We obtain a view of our object from __array_finalize__

        return obj

    @staticmethod
    def _verify_type(array):
        """Verifies the type of the input array."""
        if not isinstance(array, _np.ndarray):
            raise TypeError(f'array should be a Numpy ndarray instance, not {type(array)}.')
        if not _np.issubdtype(array.dtype, _np.integer):
            raise TypeError(f'array dtype should be integer-like, not {array.dtype}.')

    def expand(self, arr: _np.ndarray, op: Union[_np.ufunc, _nb.np.ufunc.dufunc.DUFunc, Callable]) -> _np.ndarray:
        """Helper method for selection functions. Expands guesses and given array to the correct dimensions to operate on a function.

        When coding custom selection functions, a usual operation is the broadcast of the data and guesses to operate and give a result with the correct shape.
        This method perfoms this broadcast while checking that the shape or the result is correct. That is, that is has 3 dimensions, and that the first one
        is defined by the number of trace-related data rows and the second one by the number of guesses.

        When using this function, you should aim to use a numpy or numba(.vectorize) ufunc as op, since broadcasting will be done automatically. If you input a
        Callable as op, the broadcasting and shapes must be built by yourself. In that case, this method is only useful for shapes and type checking.

        Args:
            arr (numpy.ndarray): Array to operate with the guesses. Should be 2d. If guesses is 2-d, arr.shape[1] should be divisible by guesses.shape[1].
            op (Union[numpy.ufunc, numba.np.ufunc.dufunc.DUFunc, Callable]): Operation between Guesses and arr.

        Raises:
            TypeError: Raised if arr is not a numpy.ndarray
            TypeError: Raised if op is not a numpy or numba (from numba.vectorize) ufunc or a Callable.
            ValueError: Raised if the dimensions of arr are not valid.
            TypeError: Raised if op does not return a numpy.ndarray.
            ValueError: Raised if op returns a numpy.ndarray with incorrect shape.

        Returns:
            numpy.ndarray: The resulting array from op on guesses and arr.

        """
        # Argument validation guards
        if not isinstance(arr, _np.ndarray):
            raise TypeError(f"arr must be numpy.ndarray. Given {type(arr)}.")

        if not isinstance(op, (_np.ufunc, _nb.np.ufunc.dufunc.DUFunc)) and not callable(op):
            raise TypeError(f"op must be callable, a numpy.ufunc or a ufunc generated by numba.vectorize. Given: {type(op)}")

        # Shape validation guard
        if arr.ndim != 2:
            raise ValueError(f"Invalid shape of arr. Should have 2 dimensions. Given: {arr.ndim}.")
        if self.ndim == 2:
            if arr.shape[1] % self.shape[1] != 0:
                raise ValueError("For 2d guesses, the second dimension of arr should be divisible by the second dimension of the guesses.")

        if isinstance(op, (_np.ufunc, _nb.np.ufunc.dufunc.DUFunc)):  # If ufunc-like
            if self.ndim == 1:
                ret_arr: _np.ndarray = op(self[_np.newaxis, :, _np.newaxis], arr[:, _np.newaxis, :])  # type: ignore
            else:
                arr_reshaped = arr.reshape((arr.shape[0], 1, arr.shape[1] // self.shape[1], self.shape[1]))
                ret_arr: _np.ndarray = op(self[_np.newaxis, :, _np.newaxis, :], arr_reshaped)  # type: ignore
                ret_arr = ret_arr.reshape((arr.shape[0], self.shape[0], -1)).copy()  # We infer the last dimension.
        else:  # If callable
            # Here, the only thing that is done is to check that the dimensions are correct, since we do not know the broadcast rules of the callable given.
            ret_arr: _np.ndarray = op(self, arr)  # type: ignore

        # We verify that the object is an ndarray
        if not isinstance(ret_arr, _np.ndarray):
            raise TypeError(f"op does not return a numpy.ndarray. Given: {type(ret_arr)}")
        # We ensure first two dimensions are consistent and that it has three dimensions
        if ret_arr.shape[0:2] != (arr.shape[0], self.shape[0]) or ret_arr.ndim != 3:
            raise ValueError(f"The returned numpy.ndarray does not have a correct shape. Its shape is: {ret_arr.shape}")

        return ret_arr

    def __array_finalize__(self, obj):
        return  # Currently no additional attribute is added to our class.

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        args = tuple(x.view(_np.ndarray) if isinstance(x, Guesses) else x for x in inputs)  # Casting Guesses obj to ndarray
        result = getattr(ufunc, method)(*args, **kwargs)
        return result

    def __repr__(self) -> str:
        return f"Guesses object with shape {self.shape}"

    def __str__(self) -> str:
        arr = self.view(_np.ndarray)
        if len(self) < 20:
            return f'{str(arr)[:20]}'.replace('\n', '')
        else:
            return f'{str(arr)[:20]} ... {str(arr)[-20:]}'.replace('\n', '')
