from ._base import _BaseCombination, _point_to_point_frames_check, _set_frames
from .. import center, standardize, Preprocess
import numpy as _np


def _process_frames(frame_1, frame_2):
    if (frame_1 is None) ^ (frame_2 is None):
        frame_1 = frame_2 = frame_2 if frame_1 is None else frame_1
    return frame_1, frame_2


def _fht(el):
    f = _np.fft.rfft(el, axis=1)
    return _np.real(f) - _np.imag(f)


class _TimeFrequenceCombination(_BaseCombination, Preprocess):
    def __init__(self, frame_1=None, frame_2=None, mode='raw'):
        super().__init__(frame_1=frame_1, frame_2=frame_2)
        if mode == 'centered':
            self._preprocess = center
        elif mode == 'standardized':
            self._preprocess = standardize
        else:
            self._preprocess = lambda traces: traces

    def __call__(self, traces):
        frame_1 = ... if self.frame_1 is None else self.frame_1
        frame_2 = ... if self.frame_2 is None else self.frame_2
        t_1 = self._preprocess(traces[:, frame_1])
        t_2 = self._preprocess(traces[:, frame_2])
        return self._operation(t_1, t_2)


class _TimeFrequencePointToPointMixin:

    def _set_frames(self, frame_1=None, frame_2=None):
        frame_1, frame_2 = _process_frames(frame_1, frame_2)
        _point_to_point_frames_check(self, frame_1, frame_2)


class _TimeFrequenceDifferentFramesMixin:

    def _set_frames(self, frame_1=None, frame_2=None):
        frame_1, frame_2 = _process_frames(frame_1, frame_2)
        _set_frames(self, frame_1, frame_2)


class Xcorr(_TimeFrequenceCombination, _TimeFrequencePointToPointMixin):
    """Circular Cross-Correlation combination function for second-order analysis.

    As described in paper: "Time-Frequency Analysis for Second-Order Attacks" (DOI:10.1007/978-3-319-08302-5_8)

    Args:
        frame_1 (slice or iterable, default=None): first traces frame that will be taken.
        frame_2 (slice or iterable, default=None): second optionnal traces frame that will be taken. Must be of the same length than frame_1.
        mode (str, default='raw'): a value in 'raw', 'centered' and 'standardized'.
            In centered mode, a centering preprocess is applied to traces before computation.
            In standardized mode, a standardized preprocess is applied to traces before computation.

    If frame_1 and frame_2 are both None, computed on the entire traces.
    If one of frame_1 or frame_2 is None, the not None value is used for both.

    """

    def _operation(self, el1, el2):
        return _np.real(
            _np.fft.irfft(
                _np.conjugate(_np.fft.rfft(el1, axis=1)) * _np.fft.rfft(el2, axis=1)
            )
        )


class WindowFFT(_TimeFrequenceCombination, _TimeFrequencePointToPointMixin):
    """Window-FFT combination function for second-order analysis.

    As described in paper: "Time-Frequency Analysis for Second-Order Attacks" (DOI:10.1007/978-3-319-08302-5_8)

    Computes Fast Fourier Transform (FFT) of the two frames and returns the absolute product of one and the complex conjugate of the other.

    Args:
        frame_1 (slice or iterable, default=None): first traces frame that will be taken.
        frame_2 (slice or iterable, default=None): second optionnal traces frame that will be taken. Must be of the same length than frame_1.
        mode (str, default='raw'): a value in 'raw', 'centered' and 'standardized'.
            In centered mode, a centering preprocess is applied to traces before computation.
            In standardized mode, a standardized preprocess is applied to traces before computation.

    If frame_1 and frame_2 are both None, computed on the entire traces.
    If one of frame_1 or frame_2 is None, the not None value is used for both.

    """

    def _operation(self, el1, el2):
        return _np.abs(
            _np.conjugate(
                _np.fft.rfft(el1, axis=1)) * _np.fft.rfft(el2, axis=1)
        )


class WindowFHT(_TimeFrequenceCombination, _TimeFrequencePointToPointMixin):
    """Window-FHT combination function for second-order analysis.

    As described in paper: "Time-Frequency Analysis for Second-Order Attacks" (DOI:10.1007/978-3-319-08302-5_8)

    Computes Fast Hartley Transform (FHT) of the two frames and returns the product of FHTs.
    This combination function requires equal length frames.

    Args:
        frame_1 (slice or iterable, default=None): first traces frame that will be taken.
        frame_2 (slice or iterable, default=None): second optionnal traces frame that will be taken. Must be of the same length than frame_1.
        mode (str, default='raw'): a value in 'raw', 'centered' and 'standardized'.
            In centered mode, a centering preprocess is applied to traces before computation.
            In standardized mode, a standardized preprocess is applied to traces before computation.

    If frame_1 and frame_2 are both None, computed on the entire traces.
    If one of frame_1 or frame_2 is None, the not None value is used for both.


    """

    def _operation(self, el1, el2):
        return _fht(el1) * _fht(el2)


class MaxCorr(_TimeFrequenceCombination, _TimeFrequenceDifferentFramesMixin):
    """MaxCorr combination function for second-order analysis.

    As described in paper: "Time-Frequency Analysis for Second-Order Attacks" (DOI:10.1007/978-3-319-08302-5_8)

    Concatenates the two frames and computes its Fast Fourier Transform (FFT) and returns the concatenation of real part, imaginary part and modulus of the FFT.

    Args:
        frame_1 (slice or iterable, default=None): first traces frame that will be taken.
        frame_2 (slice or iterable, default=None): second optionnal traces frame that will be taken.
        mode (str, default='raw'): a value in 'raw', 'centered' and 'standardized'.
            In centered mode, a centering preprocess is applied to traces before computation.
            In standardized mode, a standardized preprocess is applied to traces before computation.

    If frame_1 and frame_2 are both None, computed on the entire traces.
    If one of frame_1 or frame_2 is None, the not None value is used for both.

    """

    def _operation(self, el1, el2):
        f = _np.fft.rfft(_np.hstack([el1, el2]), axis=1)
        return _np.hstack([_np.real(f), _np.imag(f), _np.abs(f)])


class ConcatFFT(_TimeFrequenceCombination, _TimeFrequenceDifferentFramesMixin):
    """Concat-FFT combination function for second-order analysis.

    As described in paper: "Time-Frequency Analysis for Second-Order Attacks" (DOI:10.1007/978-3-319-08302-5_8)

    Concatenates the two frames and returns the squared modulus of its Fast Fourier Transform (FFT).

    Args:
        frame_1 (slice or iterable, default=None): first traces frame that will be taken.
        frame_2 (slice or iterable, default=None): second optionnal traces frame that will be taken.
        mode (str, default='raw'): a value in 'raw', 'centered' and 'standardized'.
            In centered mode, a centering preprocess is applied to traces before computation.
            In standardized mode, a standardized preprocess is applied to traces before computation.

    If frame_1 and frame_2 are both None, computed on the entire traces.
    If one of frame_1 or frame_2 is None, the not None value is used for both.

    """

    def _operation(self, el1, el2):
        return _np.abs(_np.fft.rfft(_np.hstack([el1, el2]), axis=1))**2


class ConcatFHT(_TimeFrequenceCombination, _TimeFrequenceDifferentFramesMixin):
    """Concat-FHT combination function for second-order analysis.

    As described in paper: "Time-Frequency Analysis for Second-Order Attacks" (DOI:10.1007/978-3-319-08302-5_8)

    Concatenates the two frames and returns its Fast Hartley Transform (FHT).

    Args:
        frame_1 (slice or iterable, default=None): first traces frame that will be taken.
        frame_2 (slice or iterable, default=None): second optionnal traces frame that will be taken.
        mode (str, default='raw'): a value in 'raw', 'centered' and 'standardized'.
            In centered mode, a centering preprocess is applied to traces before computation.
            In standardized mode, a standardized preprocess is applied to traces before computation.

    If frame_1 and frame_2 are both None, computed on the entire traces.
    If one of frame_1 or frame_2 is None, the not None value is used for both.

    """

    def _operation(self, el1, el2):
        return _fht(_np.hstack([el1, el2]))**2
