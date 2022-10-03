from .._base import PreprocessError, Preprocess
import numpy as _np


class _BaseCombination(Preprocess):

    def __init__(self, precision='float32', **kwargs):
        super().__init__()
        self.precision = _np.dtype(precision)
        self._set_frames(**kwargs)

    def _set_frame(self, name, frame):
        if isinstance(frame, slice):
            setattr(self, name, range(frame.start if frame.start else 0, frame.stop, frame.step if frame.step else 1))
        elif isinstance(frame, int):
            setattr(self, name, [frame])
        else:
            setattr(self, name, frame)

    def _set_frames(self, frame_1, frame_2):
        self._set_frame('frame_1', frame_1)
        self._set_frame('frame_2', frame_2)


class _CombinationPointToPoint(_BaseCombination):

    def _set_frames(self, frame_1, frame_2):
        super()._set_frames(frame_1, frame_2)
        if self.frame_1 is not None and self.frame_2 is not None and len(self.frame_1) != len(self.frame_2):
            raise PreprocessError('This combination mode needs frame 1 and frame 2 to be provided and of the same length.')

    def __call__(self, traces):
        dtype = max(traces.dtype, self.precision)
        frame_1 = ... if self.frame_1 is None else self.frame_1
        frame_2 = ... if self.frame_2 is None else self.frame_2
        return self._operation(traces[:, frame_1].astype(dtype),
                               traces[:, frame_2].astype(dtype))


class _CombinationOfTwoFrames(_BaseCombination):

    def _set_frames(self, frame_1, frame_2):
        self._frame_2_was_none = frame_2 is None
        if frame_2 is None:
            frame_2 = frame_1
        super()._set_frames(frame_1, frame_2)

    def __call__(self, traces):
        dtype = max(traces.dtype, self.precision)
        chunk_1 = traces[:, self.frame_1].astype(dtype)
        chunk_2 = traces[:, self.frame_2].astype(dtype)

        if self._frame_2_was_none:
            result_size = sum(range(chunk_1.shape[1] + 1))
        else:
            result_size = chunk_1.shape[1] * chunk_2.shape[1]
        result = _np.empty((traces.shape[0], result_size), dtype=dtype)

        cnt = 0
        for i in range(chunk_1.shape[1]):
            if self._frame_2_was_none:
                tmp2 = chunk_2[:, i:]
            else:
                tmp2 = chunk_2
            tmp1 = chunk_1[:, i]
            result[:, cnt: cnt + tmp2.shape[1]] = self._operation(tmp1, tmp2.T).T
            cnt += tmp2.shape[1]
        return result


class _CombinationFrameOnDistance(_BaseCombination):

    def _set_frames(self, frame_1, distance):
        super()._set_frames(frame_1, None)
        if not isinstance(distance, int) or distance < 1:
            raise ValueError(f'distance must be a positive integer, not {distance}.')
        self.distance = distance

    def _execute(self, chunk_1, chunk_2, result=None):
        cnt = 0
        for i in range(chunk_1.shape[1]):
            end = min(i + self.distance + 1, chunk_2.shape[1])
            tmp2 = chunk_2[:, i:end]
            tmp1 = chunk_1[:, i]
            if result is not None:
                result[:, cnt: cnt + tmp2.shape[1]] = self._operation(tmp1, tmp2.T).T
            cnt += tmp2.shape[1]
        return cnt, result

    def __call__(self, traces):
        dtype = max(traces.dtype, self.precision)
        chunk_1 = chunk_2 = traces[:, self.frame_1].astype(dtype)
        result_size, _ = self._execute(chunk_1, chunk_2)

        result = _np.empty((traces.shape[0], result_size), dtype=dtype)
        _, result = self._execute(chunk_1, chunk_2, result=result)
        return result


def _combination(operation, frame_1, frame_2=None, mode='full', distance=None, precision='float32'):
    if mode not in ['same', 'full']:
        raise PreprocessError('Only same or full mode are available for combination preprocesses.')
    if distance is not None and (mode == 'same' or frame_2 is not None):
        raise PreprocessError('same mode or usage of two frames is incompatible with use of distance.')
    if mode == 'same' and frame_2 is None:
        raise PreprocessError('same mode requires two frames.')

    if distance is not None:
        res = _CombinationFrameOnDistance(frame_1=frame_1, distance=distance, precision=precision)
    elif mode == 'same' and frame_2 is not None:
        res = _CombinationPointToPoint(frame_1=frame_1, frame_2=frame_2, precision=precision)
    else:
        res = _CombinationOfTwoFrames(frame_1=frame_1, frame_2=frame_2, precision=precision)
    res._operation = operation
    return res
