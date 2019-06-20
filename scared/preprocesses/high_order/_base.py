from .._base import PreprocessError, Preprocess
import numpy as _np


def _set_frame(obj, name, frame):
    if isinstance(frame, slice):
        setattr(obj, name, range(frame.start if frame.start else 0, frame.stop, frame.step if frame.step else 1))
    elif isinstance(frame, int):
        setattr(obj, name, [frame])
    else:
        setattr(obj, name, frame)


def _set_frames(obj, frame_1, frame_2):
    _set_frame(obj, 'frame_1', frame_1)
    _set_frame(obj, 'frame_2', frame_2)


def _check_same_length_frames(frame_1, frame_2):
    if frame_1 is not None and frame_2 is not None and len(frame_1) != len(frame_2):
        raise PreprocessError(f'This combination mode needs frame 1 and frame 2 to be provided and of the same length.')


def _point_to_point_frames_check(obj, frame_1, frame_2):
    _set_frames(obj, frame_1, frame_2)
    _check_same_length_frames(obj.frame_1, obj.frame_2)


class _BaseCombination:

    def __init__(self, **kwargs):
        self._set_frames(**kwargs)


class _BasicCombination(Preprocess, _BaseCombination):

    def __call__(self, traces):
        frame_1 = ... if self.frame_1 is None else self.frame_1
        frame_2 = ... if self.frame_2 is None else self.frame_2
        return self._operation(traces[:, frame_1], traces[:, frame_2])


class _CombinationPointToPoint(_BasicCombination):

    def _set_frames(self, frame_1, frame_2):
        _point_to_point_frames_check(self, frame_1, frame_2)


class _CombinationOfTwoFrames(Preprocess, _BaseCombination):

    def _set_frames(self, frame_1, frame_2):
        self._frame_2_was_none = frame_2 is None
        if frame_2 is None:
            frame_2 = frame_1
        _set_frames(self, frame_1, frame_2)

    def __call__(self, traces):
        chunk_1 = traces[:, self.frame_1]
        chunk_2 = traces[:, self.frame_2]

        if self._frame_2_was_none:
            result_size = sum(range(chunk_1.shape[1] + 1))
        else:
            result_size = chunk_1.shape[1] * chunk_2.shape[1]
        result = _np.empty((traces.shape[0], result_size), dtype=traces.dtype)
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


class _CombinationFrameOnDistance(Preprocess, _BaseCombination):

    def _set_frames(self, frame_1, distance):
        _set_frames(self, frame_1, None)
        if not isinstance(distance, int) or distance < 1:
            raise ValueError(f'distance must be a positive integer, not {distance}.')
        self.distance = distance

    def _execute(self, chunk_1, chunk_2, result=None):
        cnt = 0
        for i in range(chunk_1.shape[1]):
            end = _np.min([i + self.distance + 1, chunk_2.shape[1]]).astype('uint32')
            tmp2 = chunk_2[:, i:end]
            tmp1 = chunk_1[:, i]
            if result is not None:
                result[:, cnt: cnt + tmp2.shape[1]] = self._operation(tmp1, tmp2.T).T
            cnt += tmp2.shape[1]
        return cnt, result

    def __call__(self, traces):
        chunk_1 = chunk_2 = traces[:, self.frame_1]

        result_size, _ = self._execute(chunk_1, chunk_2)
        result = _np.empty((traces.shape[0], result_size), dtype=traces.dtype)
        _, result = self._execute(chunk_1, chunk_2, result=result)
        return result
