from .context import scared  # noqa: F401
import pytest
import numpy as np
import time
from collections.abc import Iterable


SIZES_TO_TEST = [
    (1000, 25000),
    (1001, 5000),
    (5001, 2500),
    (10001, 1000),
    (50001, 250),
    (100001, 100)
]

BATCH_NUMBER = 10
LAST_BATCH_SIZE = 100


@pytest.fixture(params=SIZES_TO_TEST)
def thss(request):
    sizes = request.param
    shape = (BATCH_NUMBER * sizes[1] + LAST_BATCH_SIZE, sizes[0])
    samples = np.random.randint(0, 255, shape, dtype='uint8')
    plaintext = np.random.randint(0, 255, (shape[0], 16), dtype='uint8')
    return scared.traces.formats.read_ths_from_ram(samples=samples, plaintext=plaintext), sizes


@pytest.fixture
def ths():
    shape = (2000, 1000)
    samples = np.random.randint(0, 255, shape, dtype='uint8')
    plaintext = np.random.randint(0, 255, (shape[0], 16), dtype='uint8')
    return scared.traces.formats.read_ths_from_ram(samples=samples, plaintext=plaintext)


@pytest.fixture
def moderate_ths():
    shape = (2000, 10000)
    samples = np.random.randint(0, 255, shape, dtype='uint8')
    plaintext = np.random.randint(0, 255, (shape[0], 16), dtype='uint8')
    return scared.traces.formats.read_ths_from_ram(samples=samples, plaintext=plaintext)


@pytest.fixture
def preprocess_half():
    @scared.preprocess
    def half(traces):
        new_shape = list(traces.shape)
        new_shape[1] = int(new_shape[1] // 2)
        new_traces = np.random.randint(0, 255, new_shape, dtype='uint8')
        return new_traces
    return half


@pytest.fixture
def preprocess_double():
    @scared.preprocess
    def double(traces):
        new_shape = list(traces.shape)
        new_shape[1] = int(new_shape[1] * 2)
        return np.random.randint(0, 255, new_shape, dtype='uint8')
    return double


def test_container_raises_exception_if_ths_is_not_trace_header_set_compatible():
    with pytest.raises(TypeError):
        scared.Container(ths='foo')
    with pytest.raises(TypeError):
        scared.Container(ths=1235)
    with pytest.raises(TypeError):
        scared.Container(ths={})


def test_container_provides_iterator_on_traces_objects(thss):
    ths, sizes = thss
    container = scared.Container(ths)
    assert isinstance(container.batches(), Iterable)
    assert isinstance(str(container), str)
    for batch in container.batches():
        assert batch.samples is not None
        assert batch.metadatas is not None
        assert len(batch) in (sizes[1], LAST_BATCH_SIZE)


def test_container_provides_trace_size_attribute(thss):
    ths, sizes = thss
    container = scared.Container(ths)
    assert isinstance(str(container), str)
    assert container.trace_size == len(ths.samples[0, :])


def test_container_batches_accept_batch_size(thss):
    ths, sizes = thss
    container = scared.Container(ths)
    n_batch = len(ths) // 2000
    last_batch = len(ths) - 2000 * n_batch
    assert isinstance(str(container), str)

    for batch in container.batches(batch_size=2000):
        if len(ths) < 2000:
            assert len(batch) == len(ths)
        else:
            assert len(batch) in (2000, last_batch)


def test_container_batches_raises_exception_if_batch_size_is_incorrect(ths):
    container = scared.Container(ths)
    with pytest.raises(TypeError):
        container.batches(batch_size='foo')
    with pytest.raises(ValueError):
        container.batches(batch_size=-12)


def test_container_batch_size_with_frame(moderate_ths):
    container = scared.Container(moderate_ths, frame=range(10))
    assert container.trace_size == 10
    assert container.batch_size == 25000


def test_container_batch_size_with_preprocess(moderate_ths):
    @scared.preprocess
    def prep_reducing_size(traces):
        return traces[:, :10]

    container = scared.Container(moderate_ths, preprocesses=[prep_reducing_size])
    assert container.trace_size == 10
    assert container.batch_size == 2500

    @scared.preprocess
    def prep_expanding_size(traces):
        return np.tile(traces, (20,))

    container = scared.Container(moderate_ths, preprocesses=[prep_expanding_size])
    assert container.trace_size == 20 * len(moderate_ths.samples[0])
    assert container.batch_size == 100


def test_container_raises_exception_if_frame_param_has_improper_type(ths):
    with pytest.raises(TypeError):
        scared.Container(ths, frame='foo')
    with pytest.raises(TypeError):
        scared.Container(ths, frame=2121.1)
    with pytest.raises(TypeError):
        scared.Container(ths, frame={})


def test_container_trace_size_use_frame(ths):
    cont = scared.Container(ths, frame=slice(None, 10))
    assert cont.trace_size == 10
    assert isinstance(str(cont), str)

    cont = scared.Container(ths, frame=slice(None, 2000))
    assert cont.trace_size == 1000
    assert isinstance(str(cont), str)

    cont = scared.Container(ths, frame=1)
    assert cont.trace_size == 1
    assert isinstance(str(cont), str)


def test_container_raises_error_if_bad_preprocesses(ths):
    with pytest.raises(TypeError):
        scared.Container(ths, preprocesses='foo')
    with pytest.raises(TypeError):
        scared.Container(ths, preprocesses=['foo', 123])
    with pytest.raises(TypeError):
        scared.Container(ths, preprocesses=134)


def test_container_with_one_preprocess(ths):
    @scared.preprocess
    def square(traces):
        return traces ** 2

    c = scared.Container(ths, preprocesses=square)
    b = c.batches(batch_size=10)[0]
    assert np.array_equal(b.samples, square(ths.samples[:10, :]))
    assert isinstance(str(c), str)


def test_container_with_multiple_preprocess(ths):
    @scared.preprocess
    def square(traces):
        return traces ** 2

    @scared.preprocess
    def minus_2(traces):
        return (traces - 2).astype(traces.dtype)

    c = scared.Container(ths, preprocesses=[square, minus_2])
    b = c.batches(batch_size=10)[0]
    assert np.array_equal(b.samples, minus_2(square(ths.samples[:10, :])))
    assert isinstance(str(c), str)


def test_container_with_frame(ths):
    c = scared.Container(ths, frame=slice(None, 20))
    b = c.batches(batch_size=10)[0]
    assert np.array_equal(b.samples, ths.samples[:10, :20])
    assert isinstance(str(c), str)


def test_container_with_frame_compute_batch_size(ths):
    c = scared.Container(ths, frame=slice(None, 20))
    s = c._compute_batch_size(trace_size=len(ths.samples[0, :20]))
    assert isinstance(s, int)
    b = c.batches(batch_size=s)[0]
    assert np.array_equal(b.samples, ths.samples[:s, :20])


def test_container_compute_batch_size_static_call(ths):
    s = scared.Container._compute_batch_size({}, trace_size=len(ths.samples[0, :20]))
    assert isinstance(s, int)
    c = scared.Container(ths, frame=slice(None, 20))
    b = c.batches(batch_size=s)[0]
    assert np.array_equal(b.samples, ths.samples[:s, :20])


def test_container_with_multiple_preprocess_and_frame(ths):
    @scared.preprocess
    def square(traces):
        return traces ** 2

    @scared.preprocess
    def minus_2(traces):
        return (traces - 2).astype(traces.dtype)

    c = scared.Container(ths, preprocesses=[square, minus_2], frame=slice(10, 30))
    b = c.batches(batch_size=10)[2]
    assert np.array_equal(b.samples, minus_2(square(ths.samples[20:30, 10:30])))
    assert isinstance(str(c), str)


def test_container_str_with_preprocesses(ths):
    # Test for issue https://gitlab.com/eshard/scared/issues/26
    c = scared.Container(ths, preprocesses=scared.preprocesses.serialize_bit)
    assert isinstance(str(c), str)

    c = scared.Container(ths, preprocesses=scared.preprocesses.high_order.WindowFFT())
    assert isinstance(str(c), str)

    @scared.preprocess
    def square(traces):
        return traces ** 2

    @scared.preprocess
    def minus_2(traces):
        return (traces - 2).astype(traces.dtype)

    c = scared.Container(ths, preprocesses=[square, minus_2])
    assert isinstance(str(c), str)

    class Pow(scared.Preprocess):

        def __init__(self, powe=1):
            super().__init__()
            self.pow = powe

        def __call__(self, traces):
            return traces ** self.pow

        def __str__(self):
            return f'Power {self.pow}'

    power = Pow(4)

    c = scared.Container(ths, preprocesses=[square, minus_2, power])
    assert 'Power 4' == str(power)
    assert isinstance(str(c), str)


def test_container_batches_last_batch_length_is_1():
    shape = (42, 100)
    samples = np.random.randint(0, 255, shape, dtype='uint8')
    plaintext = np.random.randint(0, 255, (shape[0], 16), dtype='uint8')
    ths = scared.traces.formats.read_ths_from_ram(samples=samples, plaintext=plaintext)
    container = scared.Container(ths)
    batches = list(container.batches(41))
    assert len(batches[0]) == 41
    assert len(batches[1]) == 1
    assert batches[1].samples[:].shape == (1, 100)


def test_performance_container_str():
    shapes = [(100_000, 10_000), (10_000, 10_000), (1_000, 10_000), (100_000, 1_000), (10_000, 1_000), (1_000, 1_000)]
    times = []
    for shape in shapes:
        samples = np.empty(shape, dtype='uint8')
        plaintext = np.empty((shape[0], 16), dtype='uint8')
        ths = scared.traces.formats.read_ths_from_ram(samples=samples, plaintext=plaintext)
        c = scared.Container(ths)

        tic = time.time()
        str(c)
        toc = time.time()
        times.append(toc - tic)

    assert max(times) / min(times) < 10, 'str representation of container takes too much time'


def test_performance_container_batch_size(moderate_ths):
    shapes = [(100_000, 10_000), (10_000, 10_000), (1_000, 10_000)]
    times = []
    for shape in shapes:
        samples = np.empty(shape, dtype='uint8')
        plaintext = np.empty((shape[0], 16), dtype='uint8')
        ths = scared.traces.formats.read_ths_from_ram(samples=samples, plaintext=plaintext)
        c = scared.Container(ths)

        tic = time.time()
        c.batch_size
        toc = time.time()
        times.append(toc - tic)

    assert max(times) / min(times) < 5, 'batch_size computation takes too much time'


######################
#   Set batch size   #
######################

def test_floor_to_most_significant_digit():
    for value, expected in [(0, 0), (0.1, 0), (1, 1), (10, 10), (2000, 2000), (11, 10), (5999, 5000)]:
        assert scared.container._floor_to_most_significant_digit(value) == expected


def test_set_batch_size_list(ths):
    scared.set_batch_size([(0, 500), (1000, 100), (2000, 50)])
    c = scared.Container(ths)
    assert c.batch_size == 100


def test_set_batch_size_int(ths):
    scared.set_batch_size(42)
    c = scared.Container(ths)
    assert c.batch_size == 42


def test_set_batch_size_float_1(ths):
    # raw batch size 262, floored to 200
    scared.set_batch_size(0.25)
    c = scared.Container(ths)
    assert c.batch_size == 200


def test_set_batch_size_float_2(ths):
    # raw batch size 786, floored to 700
    scared.set_batch_size(0.75)
    c = scared.Container(ths)
    assert c.batch_size == 700


def test_set_batch_size_list_with_preprocess(ths, preprocess_half, preprocess_double):
    scared.set_batch_size([(0, 500), (1000, 100), (2000, 50)])
    c = scared.Container(ths, preprocesses=preprocess_half)
    assert c.batch_size == 100
    c = scared.Container(ths, preprocesses=preprocess_double)
    assert c.batch_size == 50


def test_set_batch_size_int_with_preprocess(ths, preprocess_half, preprocess_double):
    scared.set_batch_size(42)
    c = scared.Container(ths, preprocesses=preprocess_half)
    assert c.batch_size == 42
    c = scared.Container(ths, preprocesses=preprocess_double)
    assert c.batch_size == 42


def test_set_batch_size_float_1_with_preprocess(ths, preprocess_half, preprocess_double):
    # raw batch size 262, floored to 200
    scared.set_batch_size(0.25)
    c = scared.Container(ths, preprocesses=preprocess_half)
    assert c.batch_size == 200
    c = scared.Container(ths, preprocesses=preprocess_double)
    assert c.batch_size == 100


def test_set_batch_size_float_minimum_batch_size_is_10(ths):
    scared.set_batch_size(0.000001)
    c = scared.Container(ths)
    assert c.batch_size == 10


def test_set_batch_size_big_int(ths):
    scared.set_batch_size(42_000)
    c = scared.Container(ths)
    batches = list(c.batches())
    assert len(batches) == 1
    for batch in batches:
        assert len(batch) == 2000


def test_set_batch_size_none_restore():
    scared.set_batch_size(42)
    assert scared.Container._BATCH_SIZE == 42
    scared.set_batch_size()
    assert scared.Container._BATCH_SIZE == scared.container._ORIGINAL_BATCH_SIZES


def test_set_batch_size_raises_if_incorrect_type():
    with pytest.raises(TypeError, match='batch_size must be an integer, a float or a list, but'):
        scared.set_batch_size('foo')


def test_set_batch_size_raises_if_value_lower_or_equal_zero():
    with pytest.raises(ValueError, match='batch_size must be strictly positive, but'):
        scared.set_batch_size(0)
    with pytest.raises(ValueError, match='batch_size must be strictly positive, but'):
        scared.set_batch_size(-1.1)


def test_set_batch_size_raises_if_not_list_of_couples():
    with pytest.raises(ValueError, match='Items of a batch sizes list must be couples, but'):
        scared.set_batch_size([(1, 2), 'foo'])
    with pytest.raises(ValueError, match='Items of a batch sizes list must be couples, but'):
        scared.set_batch_size([(1, 2), (2, 3, 4)])


def test_set_batch_size_raises_if_not_integers_in_list():
    with pytest.raises(ValueError, match='Values in batch sizes list must be integers, but '):
        scared.set_batch_size([(1, 2), (1.0, 1)])
