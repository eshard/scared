from .context import scared  # noqa: F401
import pytest
import numpy as np


@pytest.fixture
def ths_1():
    shape = (2000, 1000)
    sample = np.random.randint(0, 255, (1000,), dtype='uint8')
    plain = np.random.randint(0, 255, (16), dtype='uint8')
    samples = np.array([sample for i in range(shape[0])], dtype='uint8')
    plaintext = np.array([plain for i in range(shape[0])], dtype='uint8')
    return scared.traces.formats.read_ths_from_ram(samples=samples, plaintext=plaintext)


@pytest.fixture
def ths_2():
    shape = (2000, 1000)
    samples = np.random.randint(0, 255, shape, dtype='uint8')
    plaintext = np.random.randint(0, 255, (shape[0], 16), dtype='uint8')
    return scared.traces.formats.read_ths_from_ram(samples=samples, plaintext=plaintext)


@pytest.fixture
def ths_3():
    shape = (2000, 1001)
    samples = np.random.randint(0, 256, shape, dtype='uint8')
    plaintext = np.random.randint(0, 256, (shape[0], 16), dtype='uint8')
    return scared.traces.formats.read_ths_from_ram(samples=samples, plaintext=plaintext)


def test_ttest_container_raises_exception_if_incorrect_ths_provided(ths_1, ths_2):
    with pytest.raises(TypeError):
        scared.TTestContainer(ths_1='foo', ths_2='bar')
    with pytest.raises(TypeError):
        scared.TTestContainer(ths_1=ths_1, ths_2='bar')
    with pytest.raises(TypeError):
        scared.TTestContainer(ths_1='foo', ths_2=ths_2)


def test_ttest_container_raises_exception_if_incorrect_frame_provided(ths_1, ths_2):
    with pytest.raises(TypeError):
        scared.TTestContainer(ths_1=ths_1, ths_2=ths_2, frame='foo')
    with pytest.raises(TypeError):
        scared.TTestContainer(ths_1, ths_2, frame=2121.1)
    with pytest.raises(TypeError):
        scared.TTestContainer(ths_1, ths_2, frame={})


def test_container_raises_error_if_bad_preprocesses(ths_1, ths_2):
    with pytest.raises(TypeError):
        scared.TTestContainer(ths_1, ths_2, preprocesses='foo')
    with pytest.raises(TypeError):
        scared.TTestContainer(ths_1, ths_2, preprocesses=['foo', 123])
    with pytest.raises(TypeError):
        scared.TTestContainer(ths_1, ths_2, preprocesses=134)


def test_container_raises_error_if_traces_not_same_size(ths_1, ths_3):
    with pytest.raises(ValueError,
                       match=r"Shape of traces must be the same, found 1000 and 1001"):
        scared.TTestContainer(ths_1, ths_3)
