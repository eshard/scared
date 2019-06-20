from ..context import scared  # noqa: F401
import pytest
import numpy as np

traces = np.random.randint(0, 256, (10, 100), dtype='uint8')
frame1 = range(5, 10)
frame2 = range(15, 20)


def test_xcorr_with_frame_none_returns_equal_shape():
    assert scared.preprocesses.high_order.Xcorr()(traces).shape == traces.shape


def test_xcorr_raises_exception_if_frames_are_not_equal_len():
    with pytest.raises(scared.PreprocessError):
        scared.preprocesses.high_order.Xcorr(
            frame_1=slice(None, 50),
            frame_2=slice(None, 10)
        )


def test_xcorr_with_not_none_frames():
    xc = scared.preprocesses.high_order.Xcorr(frame_1=frame1, frame_2=frame2)
    out = xc(traces)
    t_1 = traces[:, frame1]
    t_2 = traces[:, frame2]
    t_1 = np.fft.rfft(t_1, axis=1)
    t_2 = np.fft.rfft(t_2, axis=1)
    expected = np.fft.irfft(np.conjugate(t_1) * t_2)
    assert np.allclose(out, expected)


def test_xcorr_with_none_frames():
    xc = scared.preprocesses.high_order.Xcorr()
    out = xc(traces)
    t_1 = np.fft.rfft(traces, axis=1)
    expected = np.fft.irfft(np.conjugate(t_1) * t_1)
    assert np.allclose(out, expected)


def test_xcorr_with_one_none_frame():
    xc = scared.preprocesses.high_order.Xcorr(frame_1=frame1)
    out = xc(traces)
    t_1 = traces[:, frame1]
    t_1 = np.fft.rfft(t_1, axis=1)
    expected = np.fft.irfft(np.conjugate(t_1) * t_1)
    assert np.allclose(out, expected)

    xc = scared.preprocesses.high_order.Xcorr(frame_2=frame2)
    out = xc(traces)
    t_1 = traces[:, frame2]
    t_1 = np.fft.rfft(t_1, axis=1)
    expected = np.fft.irfft(np.conjugate(t_1) * t_1)
    assert np.allclose(out, expected)


def test_window_fft_raises_exception_if_frames_are_not_equal_len():
    with pytest.raises(scared.PreprocessError):
        scared.preprocesses.high_order.WindowFFT(
            frame_1=slice(None, 50),
            frame_2=slice(None, 10)
        )


def test_window_fft_with_not_none_frames():
    xc = scared.preprocesses.high_order.WindowFFT(frame_1=frame1, frame_2=frame2)
    out = xc(traces)
    t_1 = traces[:, frame1]
    t_2 = traces[:, frame2]
    t_1 = np.fft.rfft(t_1, axis=1)
    t_2 = np.fft.rfft(t_2, axis=1)
    expected = np.abs(np.conjugate(t_1) * t_2)
    assert np.allclose(out, expected)


def test_window_fft_with_none_frames():
    xc = scared.preprocesses.high_order.WindowFFT()
    out = xc(traces)
    t_1 = np.fft.rfft(traces, axis=1)
    expected = np.abs(np.conjugate(t_1) * t_1)
    assert np.allclose(out, expected)


def test_window_fft_with_one_none_frame():
    xc = scared.preprocesses.high_order.WindowFFT(frame_1=frame1)
    out = xc(traces)
    t_1 = traces[:, frame1]
    t_1 = np.fft.rfft(t_1, axis=1)
    expected = np.abs(np.conjugate(t_1) * t_1)
    assert np.allclose(out, expected)

    xc = scared.preprocesses.high_order.WindowFFT(frame_2=frame2)
    out = xc(traces)
    t_1 = traces[:, frame2]
    t_1 = np.fft.rfft(t_1, axis=1)
    expected = np.abs(np.conjugate(t_1) * t_1)
    assert np.allclose(out, expected)


def test_window_fht_raises_exception_if_frames_are_not_equal_len():
    with pytest.raises(scared.PreprocessError):
        scared.preprocesses.high_order.WindowFHT(
            frame_1=slice(None, 50),
            frame_2=slice(None, 10)
        )


def test_window_fht_with_not_none_frames():
    xc = scared.preprocesses.high_order.WindowFHT(frame_1=frame1, frame_2=frame2)
    out = xc(traces)
    t_1 = traces[:, frame1]
    t_2 = traces[:, frame2]
    t_1 = np.fft.rfft(t_1, axis=1)
    t_2 = np.fft.rfft(t_2, axis=1)
    t_1 = np.real(t_1) - np.imag(t_1)
    t_2 = np.real(t_2) - np.imag(t_2)
    expected = t_1 * t_2
    assert np.allclose(out, expected)


def test_window_fht_with_none_frames():
    xc = scared.preprocesses.high_order.WindowFHT()
    out = xc(traces)
    t_1 = np.fft.rfft(traces, axis=1)
    t_1 = np.real(t_1) - np.imag(t_1)
    expected = t_1 * t_1
    assert np.allclose(out, expected)


def test_window_fht_with_one_none_frame():
    xc = scared.preprocesses.high_order.WindowFHT(frame_1=frame1)
    out = xc(traces)
    t_1 = traces[:, frame1]
    t_1 = np.fft.rfft(t_1, axis=1)
    t_1 = np.real(t_1) - np.imag(t_1)
    expected = t_1 * t_1
    assert np.allclose(out, expected)

    xc = scared.preprocesses.high_order.WindowFHT(frame_2=frame2)
    out = xc(traces)
    t_1 = traces[:, frame2]
    t_1 = np.fft.rfft(t_1, axis=1)
    t_1 = np.real(t_1) - np.imag(t_1)
    expected = t_1 * t_1
    assert np.allclose(out, expected)


def test_concat_fxt_and_max_corr_accepts_different_lengths_frames():
    frame = range(3)
    other_len = range(10)

    assert scared.preprocesses.high_order.ConcatFFT(frame_1=frame, frame_2=other_len)
    assert scared.preprocesses.high_order.ConcatFHT(frame_1=frame, frame_2=other_len)
    assert scared.preprocesses.high_order.MaxCorr(frame_1=frame, frame_2=other_len)


def test_concat_fft():
    p = scared.preprocesses.high_order.ConcatFFT(frame_1=frame1, frame_2=frame2)
    out = p(traces)

    t_1 = traces[:, frame1]
    t_2 = traces[:, frame2]
    t = np.hstack([t_1, t_2])

    t = np.fft.rfft(t, axis=1)
    expected = np.abs(t)**2

    assert np.allclose(out, expected)


def test_concat_fht():
    p = scared.preprocesses.high_order.ConcatFHT(frame_1=frame1, frame_2=frame2)
    out = p(traces)

    t_1 = traces[:, frame1]
    t_2 = traces[:, frame2]
    t = np.hstack([t_1, t_2])

    t = np.fft.rfft(t, axis=1)
    expected = (np.real(t) - np.imag(t))**2

    assert np.allclose(out, expected)


def test_maxcorr():
    p = scared.preprocesses.high_order.MaxCorr(frame_1=frame1, frame_2=frame2)
    out = p(traces)

    t_1 = traces[:, frame1]
    t_2 = traces[:, frame2]
    t = np.hstack([t_1, t_2])

    t = np.fft.rfft(t, axis=1)
    expected = np.hstack([np.real(t), np.imag(t), np.abs(t)])

    assert np.allclose(out, expected)


def test_xcorr_centered_mode():
    xc = scared.preprocesses.high_order.Xcorr(mode='centered')
    out = xc(traces)
    t_1 = np.fft.rfft(scared.preprocesses.center(traces), axis=1)
    expected = np.fft.irfft(np.conjugate(t_1) * t_1)
    assert np.allclose(out, expected)


def test_window_fft_centered_mode():
    xc = scared.preprocesses.high_order.WindowFFT(mode='centered')
    out = xc(traces)
    t_1 = np.fft.rfft(scared.preprocesses.center(traces), axis=1)
    expected = np.abs(np.conjugate(t_1) * t_1)
    assert np.allclose(out, expected)


def test_window_fht_centered_mode():
    xc = scared.preprocesses.high_order.WindowFHT(mode='centered')
    out = xc(traces)
    t_1 = np.fft.rfft(scared.preprocesses.center(traces), axis=1)
    t_1 = np.real(t_1) - np.imag(t_1)
    expected = t_1 * t_1
    assert np.allclose(out, expected)


def test_max_corr_centered_mode():
    xc = scared.preprocesses.high_order.MaxCorr(mode='centered')
    out = xc(traces)
    expected = scared.preprocesses.high_order.MaxCorr()(scared.preprocesses.center(traces))
    assert np.allclose(out, expected)


def test_concat_fft_centered_mode():
    xc = scared.preprocesses.high_order.ConcatFFT(mode='centered')
    out = xc(traces)
    expected = scared.preprocesses.high_order.ConcatFFT()(scared.preprocesses.center(traces))
    assert np.allclose(out, expected)


def test_concat_fht_centered_mode():
    xc = scared.preprocesses.high_order.ConcatFHT(mode='centered')
    out = xc(traces)
    expected = scared.preprocesses.high_order.ConcatFHT()(scared.preprocesses.center(traces))
    assert np.allclose(out, expected)


def test_xcorr_std_mode():
    xc = scared.preprocesses.high_order.Xcorr(mode='standardized')
    out = xc(traces)
    t_1 = np.fft.rfft(scared.preprocesses.standardize(traces), axis=1)
    expected = np.fft.irfft(np.conjugate(t_1) * t_1)
    assert np.allclose(out, expected)


def test_window_fft_std_mode():
    xc = scared.preprocesses.high_order.WindowFFT(mode='standardized')
    out = xc(traces)
    t_1 = np.fft.rfft(scared.preprocesses.standardize(traces), axis=1)
    expected = np.abs(np.conjugate(t_1) * t_1)
    assert np.allclose(out, expected)


def test_window_fht_std_mode():
    xc = scared.preprocesses.high_order.WindowFHT(mode='standardized')
    out = xc(traces)
    t_1 = np.fft.rfft(scared.preprocesses.standardize(traces), axis=1)
    t_1 = np.real(t_1) - np.imag(t_1)
    expected = t_1 * t_1
    assert np.allclose(out, expected)


def test_max_corr_std_mode():
    xc = scared.preprocesses.high_order.MaxCorr(mode='standardized')
    out = xc(traces)
    expected = scared.preprocesses.high_order.MaxCorr()(scared.preprocesses.standardize(traces))
    assert np.allclose(out, expected)


def test_concat_fft_std_mode():
    xc = scared.preprocesses.high_order.ConcatFFT(mode='standardized')
    out = xc(traces)
    expected = scared.preprocesses.high_order.ConcatFFT()(scared.preprocesses.standardize(traces))
    assert np.allclose(out, expected)


def test_concat_fht_std_mode():
    xc = scared.preprocesses.high_order.ConcatFHT(mode='standardized')
    out = xc(traces)
    expected = scared.preprocesses.high_order.ConcatFHT()(scared.preprocesses.standardize(traces))
    assert np.allclose(out, expected)
