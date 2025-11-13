from __future__ import annotations
from typing import Union
import numpy as _np


class Guesses(_np.ndarray):
    def __new__(cls, guess_list: Union[range, _np.ndarray, list[_np.ndarray], list[range]], dtype: Union[None, _np.dtype] = None) -> Guesses:
        obj = guess_list
        if isinstance(obj, range):
            obj = _np.array(obj, dtype=dtype)
            cls._verify_type(obj)
        elif isinstance(obj, list):
            if len(obj) == 0:
                raise ValueError("List of guesses per word given was empty.")
            obj = [_np.array(element, dtype=dtype) if isinstance(element, range) else element for element in obj]  # Convert ranges to numpy arrays
            # We verify the type of every element
            for element in obj:
                cls._verify_type(element)
            # We do the crossing of the guesses
            obj = _np.stack(_np.meshgrid(*obj, indexing='ij'), axis=-1).reshape(-1, len(obj))
        else:
            cls._verify_type(obj)
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

        obj = _np.asarray(obj).view(cls)  # We obtain a view of our object from __array_finalize__

        return obj

    @staticmethod
    def _verify_type(array):
        if not isinstance(array, _np.ndarray):
            raise TypeError(f'array should be a Numpy ndarray instance, not {type(array)}.')
        if not _np.issubdtype(array.dtype, _np.integer):
            raise TypeError(f'array dtype should be integer-like, not {array.dtype}.')
        return True

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
