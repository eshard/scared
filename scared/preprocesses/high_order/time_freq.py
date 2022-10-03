from ._base import _BaseCombination
from .. import center as _center, standardize as _standardize
from .._base import PreprocessError
import numpy as _np


def _fht(el):
    f = _np.fft.rfft(el, axis=1)
    return _np.real(f) - _np.imag(f)


class _TimeFrequencyCombination(_BaseCombination):
    def __init__(self, frame_1=None, frame_2=None, mode='raw'):
        super().__init__(frame_1=frame_1, frame_2=frame_2)

        self._preprocess = {'centered': _center,
                            'standardized': _standardize
                            }.get(mode, lambda traces: traces)  # Fallback to identity by default

    def _handle_none_frame(self, frame_1, frame_2):
        if (frame_1 is None) or (frame_2 is None):
            frame_1 = frame_2 = frame_2 if frame_1 is None else frame_1
        return frame_1, frame_2

    def __call__(self, traces):
        frame_1 = ... if self.frame_1 is None else self.frame_1
        frame_2 = ... if self.frame_2 is None else self.frame_2
        t_1 = self._preprocess(traces[:, frame_1])
        t_2 = self._preprocess(traces[:, frame_2])
        return self._operation(t_1, t_2)


class _TimeFrequencyPointToPointMixin:

    def _set_frames(self, frame_1=None, frame_2=None):
        frame_1, frame_2 = super()._handle_none_frame(frame_1, frame_2)
        super()._set_frames(frame_1, frame_2)  # Uses _BaseCombination._set_frames
        if self.frame_1 is not None and self.frame_2 is not None and len(self.frame_1) != len(self.frame_2):
            raise PreprocessError('This combination mode needs frame 1 and 2 to be of the same length.')


class _TimeFrequencyDifferentFramesMixin:

    def _set_frames(self, frame_1=None, frame_2=None):
        frame_1, frame_2 = super()._handle_none_frame(frame_1, frame_2)
        super()._set_frames(frame_1, frame_2)  # Uses _BaseCombination._set_frames


class Xcorr(_TimeFrequencyPointToPointMixin, _TimeFrequencyCombination):
    """Circular Cross-Correlation combination function for second-order analysis.

    As described in paper: "Time-Frequency Analysis for Second-Order Attacks" (DOI:10.1007/978-3-319-08302-5_8)

    Args:
        frame_1 (slice or iterable, default=None): first traces frame that will be taken.
        frame_2 (slice or iterable, default=None): second optional traces frame that will be taken. Must be of the same length than frame_1.
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


class WindowFFT(_TimeFrequencyPointToPointMixin, _TimeFrequencyCombination):
    """Window-FFT combination function for second-order analysis.

    As described in paper: "Time-Frequency Analysis for Second-Order Attacks" (DOI:10.1007/978-3-319-08302-5_8)

    Computes Fast Fourier Transform (FFT) of the two frames and returns the absolute product of one and the complex conjugate of the other.

    Args:
        frame_1 (slice or iterable, default=None): first traces frame that will be taken.
        frame_2 (slice or iterable, default=None): second optional traces frame that will be taken. Must be of the same length than frame_1.
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


class WindowFHT(_TimeFrequencyPointToPointMixin, _TimeFrequencyCombination):
    """Window-FHT combination function for second-order analysis.

    As described in paper: "Time-Frequency Analysis for Second-Order Attacks" (DOI:10.1007/978-3-319-08302-5_8)

    Computes Fast Hartley Transform (FHT) of the two frames and returns the product of FHTs.
    This combination function requires equal length frames.

    Args:
        frame_1 (slice or iterable, default=None): first traces frame that will be taken.
        frame_2 (slice or iterable, default=None): second optional traces frame that will be taken. Must be of the same length than frame_1.
        mode (str, default='raw'): a value in 'raw', 'centered' and 'standardized'.
            In centered mode, a centering preprocess is applied to traces before computation.
            In standardized mode, a standardized preprocess is applied to traces before computation.

    If frame_1 and frame_2 are both None, computed on the entire traces.
    If one of frame_1 or frame_2 is None, the not None value is used for both.


    """

    def _operation(self, el1, el2):
        return _fht(el1) * _fht(el2)


class MaxCorr(_TimeFrequencyDifferentFramesMixin, _TimeFrequencyCombination):
    """MaxCorr combination function for second-order analysis.

    As described in paper: "Time-Frequency Analysis for Second-Order Attacks" (DOI:10.1007/978-3-319-08302-5_8)

    Concatenates the two frames and computes its Fast Fourier Transform (FFT) and returns the concatenation of real part, imaginary part and modulus of the FFT.

    Args:
        frame_1 (slice or iterable, default=None): first traces frame that will be taken.
        frame_2 (slice or iterable, default=None): second optional traces frame that will be taken.
        mode (str, default='raw'): a value in 'raw', 'centered' and 'standardized'.
            In centered mode, a centering preprocess is applied to traces before computation.
            In standardized mode, a standardized preprocess is applied to traces before computation.

    If frame_1 and frame_2 are both None, computed on the entire traces.
    If one of frame_1 or frame_2 is None, the not None value is used for both.

    """

    def _operation(self, el1, el2):
        f = _np.fft.rfft(_np.hstack([el1, el2]), axis=1)
        return _np.hstack([_np.real(f), _np.imag(f), _np.abs(f)])


class ConcatFFT(_TimeFrequencyDifferentFramesMixin, _TimeFrequencyCombination):
    """Concat-FFT combination function for second-order analysis.

    As described in paper: "Time-Frequency Analysis for Second-Order Attacks" (DOI:10.1007/978-3-319-08302-5_8)

    Concatenates the two frames and returns the squared modulus of its Fast Fourier Transform (FFT).

    Args:
        frame_1 (slice or iterable, default=None): first traces frame that will be taken.
        frame_2 (slice or iterable, default=None): second optional traces frame that will be taken.
        mode (str, default='raw'): a value in 'raw', 'centered' and 'standardized'.
            In centered mode, a centering preprocess is applied to traces before computation.
            In standardized mode, a standardized preprocess is applied to traces before computation.

    If frame_1 and frame_2 are both None, computed on the entire traces.
    If one of frame_1 or frame_2 is None, the not None value is used for both.

    """

    def _operation(self, el1, el2):
        return _np.abs(_np.fft.rfft(_np.hstack([el1, el2]), axis=1))**2


class ConcatFHT(_TimeFrequencyDifferentFramesMixin, _TimeFrequencyCombination):
    """Concat-FHT combination function for second-order analysis.

    As described in paper: "Time-Frequency Analysis for Second-Order Attacks" (DOI:10.1007/978-3-319-08302-5_8)

    Concatenates the two frames and returns its Fast Hartley Transform (FHT).

    Args:
        frame_1 (slice or iterable, default=None): first traces frame that will be taken.
        frame_2 (slice or iterable, default=None): second optional traces frame that will be taken.
        mode (str, default='raw'): a value in 'raw', 'centered' and 'standardized'.
            In centered mode, a centering preprocess is applied to traces before computation.
            In standardized mode, a standardized preprocess is applied to traces before computation.

    If frame_1 and frame_2 are both None, computed on the entire traces.
    If one of frame_1 or frame_2 is None, the not None value is used for both.

    """

    def _operation(self, el1, el2):
        return _fht(_np.hstack([el1, el2]))**2
