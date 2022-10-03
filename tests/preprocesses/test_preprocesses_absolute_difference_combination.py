from ..context import scared  # noqa: F401
import pytest
import numpy as np


def _read(filename):
    for v in np.load(f'tests/preprocesses/samples/{filename}').values():
        return v


@pytest.fixture
def traces():
    return _read('absolute_difference_input.npz')


@pytest.fixture(params=['int16', 'uint8', 'float32', 'float64'])
def traces_dtypes(request):
    if request.param == 'int16':
        return np.random.randint(-10000, 10000, (500, 2001), dtype='int16')
    elif request.param == 'uint8':
        return np.random.randint(0, 256, (500, 2001), dtype='uint8')
    elif request.param == 'float32':
        return np.random.random((500, 2001)).astype('float32')
    elif request.param == 'float64':
        return np.random.random((500, 2001)).astype('float64')


def test_absolute_difference_with_no_frame(traces):
    result = scared.preprocesses.high_order.AbsoluteDifference()(traces)
    assert (500, 20100) == result.shape


def test_absolute_difference_with_one_frame_and_default_values(traces):
    expected = _read('absolute_difference_result.npz')
    frame = slice(None, 50)
    result = scared.preprocesses.high_order.AbsoluteDifference(frame_1=frame)(traces)
    assert np.array_equal(expected, result)


def test_absolute_difference_with_two_frames_and_default_values(traces):
    expected = _read('absolute_difference_result_frame2_10.npz')
    frame = slice(None, 50)
    frame_2 = slice(None, 10)
    result = scared.preprocesses.high_order.AbsoluteDifference(frame_1=frame, frame_2=frame_2)(traces)
    assert np.array_equal(expected, result)


def test_absolute_difference_with_two_frames_of_same_length_and_default_values(traces):
    expected = _read('absolute_difference_result_two_frames_same_length.npz')
    frame = slice(None, 50)
    frame_2 = slice(None, 50)
    result = scared.preprocesses.high_order.AbsoluteDifference(frame_1=frame, frame_2=frame_2)(traces)
    assert np.array_equal(expected, result)


def test_absolute_difference_with_one_frame_of_one_point_and_default_values(traces):
    expected = _read('absolute_difference_result_frame_one_point.npz')
    frame = 50
    result = scared.preprocesses.high_order.AbsoluteDifference(frame_1=frame)(traces)
    assert np.array_equal(expected, result)


def test_absolute_difference_with_two_frames_of_one_point_and_default_values(traces):
    expected = _read('absolute_difference_result_two_frames_of_one_point.npz')
    frame = 50
    frame_2 = 10
    result = scared.preprocesses.high_order.AbsoluteDifference(frame_1=frame, frame_2=frame_2)(traces)
    assert np.array_equal(expected, result)


def test_absolute_difference_same_mode_raises_exceptions_if_frame_1_frame_2_different_lengths():
    with pytest.raises(scared.PreprocessError):
        scared.preprocesses.high_order.AbsoluteDifference(frame_1=range(60), mode='same')
    with pytest.raises(scared.PreprocessError):
        scared.preprocesses.high_order.AbsoluteDifference(frame_1=range(60), frame_2=range(10), mode='same')
    with pytest.raises(scared.PreprocessError):
        scared.preprocesses.high_order.AbsoluteDifference(frame_1=60, frame_2=range(10), mode='same')
    with pytest.raises(scared.PreprocessError):
        scared.preprocesses.high_order.AbsoluteDifference(frame_1=range(60), frame_2=10, mode='same')


def test_absolute_difference_same_mode(traces):
    expected = _read('absolute_difference_result_frame1_same_mode.npz')
    frame = slice(None, 50)
    frame_2 = slice(None, 50)
    result = scared.preprocesses.high_order.AbsoluteDifference(frame_1=frame, frame_2=frame_2, mode='same')(traces)
    assert np.array_equal(expected, result)

    expected = _read('absolute_difference_result_frame1_same_mode_2.npz')
    frame = slice(None, 10)
    frame_2 = slice(50, 60)
    result = scared.preprocesses.high_order.AbsoluteDifference(frame_1=frame, frame_2=frame_2, mode='same')(traces)
    assert np.array_equal(expected, result)


def test_absolute_difference_raises_exceptions_if_distance_is_used_with_incompatible_values():
    with pytest.raises(scared.PreprocessError):
        scared.preprocesses.high_order.AbsoluteDifference(frame_1=range(60), frame_2=range(10), distance=12)
    with pytest.raises(ValueError):
        scared.preprocesses.high_order.AbsoluteDifference(frame_1=range(60), distance=0)
    with pytest.raises(ValueError):
        scared.preprocesses.high_order.AbsoluteDifference(frame_1=60, distance='foo')
    with pytest.raises(ValueError):
        scared.preprocesses.high_order.AbsoluteDifference(frame_1=range(60), distance=1.2)
    with pytest.raises(ValueError):
        scared.preprocesses.high_order.AbsoluteDifference(frame_1=range(60), distance=-2)
    with pytest.raises(scared.PreprocessError):
        scared.preprocesses.high_order.AbsoluteDifference(frame_1=range(60), distance=2, mode='same')


def test_absolute_difference_full_mode_with_distance(traces):
    expected = _read('absolute_difference_result_frame1_dist_5.npz')
    frame = slice(None, 50)
    result = scared.preprocesses.high_order.AbsoluteDifference(frame_1=frame, distance=5)(traces)
    assert np.array_equal(expected, result)

    expected = _read('absolute_difference_result_frame1_dist_10.npz')
    frame = slice(None, 50)
    result = scared.preprocesses.high_order.AbsoluteDifference(frame_1=frame, distance=10)(traces)
    assert np.array_equal(expected, result)


def test_absolute_difference_is_preprocess():
    p = scared.preprocesses.high_order.AbsoluteDifference(frame_1=50, distance=5)
    with pytest.raises(TypeError):
        p('foo')


def test_raises_exception_with_improper_mode():
    with pytest.raises(scared.PreprocessError):
        scared.preprocesses.high_order.AbsoluteDifference(mode='wfor')


def test_absolute_difference_various_dtypes(traces_dtypes):
    # Mode PointToPoint
    result = scared.preprocesses.high_order.AbsoluteDifference(frame_1=slice(0, 50), frame_2=slice(50, 100), mode='same')(traces_dtypes)
    assert result.dtype == max(traces_dtypes.dtype, 'float32')
    assert result.min() >= 0
    assert result.max() <= np.abs(traces_dtypes[:, slice(0, 50)] - traces_dtypes[:, slice(50, 100)]).max()

    # Mode TwoFrames
    result = scared.preprocesses.high_order.AbsoluteDifference(slice(None, 50))(traces_dtypes)
    assert result.dtype == max(traces_dtypes.dtype, 'float32')
    assert result.min() >= 0
    assert result.max() <= traces_dtypes[:, slice(None, 50)].max() - traces_dtypes[:, slice(None, 50)].min()

    # Mode FrameOnDistance
    result = scared.preprocesses.high_order.AbsoluteDifference(slice(None, 50), distance=5)(traces_dtypes)
    assert result.dtype == max(traces_dtypes.dtype, 'float32')
    assert result.min() >= 0
    assert result.max() <= traces_dtypes[:, slice(None, 50)].max() - traces_dtypes[:, slice(None, 50)].min()


def test_absolute_difference_given_precision(traces_dtypes):
    # Mode PointToPoint
    result = scared.preprocesses.high_order.AbsoluteDifference(frame_1=slice(0, 50), frame_2=slice(50, 100), mode='same', precision='float64')(traces_dtypes)
    expected = scared.preprocesses.high_order.AbsoluteDifference(frame_1=slice(0, 50), frame_2=slice(50, 100), mode='same')(traces_dtypes.astype('float64'))
    assert result.dtype == 'float64'
    assert np.array_equal(result, expected)

    # Mode TwoFrames
    result = scared.preprocesses.high_order.AbsoluteDifference(slice(None, 50), precision='float64')(traces_dtypes)
    expected = scared.preprocesses.high_order.AbsoluteDifference(slice(None, 50))(traces_dtypes.astype('float64'))
    assert result.dtype == 'float64'
    assert np.array_equal(result, expected)

    # Mode FrameOnDistance
    result = scared.preprocesses.high_order.AbsoluteDifference(slice(None, 50), distance=5, precision='float64')(traces_dtypes)
    expected = scared.preprocesses.high_order.AbsoluteDifference(slice(None, 50), distance=5)(traces_dtypes.astype('float64'))
    assert result.dtype == 'float64'
    assert np.array_equal(result, expected)
