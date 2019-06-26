from ..context import scared
import pytest
import numpy as np


@pytest.fixture
def dumb_preprocess():
    @scared.preprocess
    def _(traces):
        return traces
    return _


def test_preprocess_decorator_raises_exception_if_function_is_not_provided():
    with pytest.raises(TypeError):
        scared.preprocess()


def test_preprocess_call_raise_exception_if_traces_have_incorrect_type(dumb_preprocess):
    @scared.preprocess
    def dumb_preprocess(traces):
        return traces

    with pytest.raises(TypeError):
        dumb_preprocess(traces='foo')

    with pytest.raises(TypeError):
        dumb_preprocess(traces=123)


def test_preprocess_call_raise_exception_if_traces_have_incorrect_shape(dumb_preprocess):
    with pytest.raises(ValueError):
        dumb_preprocess(traces=np.array([[[1], [2]], [[1], [2]]]))
    with pytest.raises(ValueError):
        dumb_preprocess(traces=np.array([1]))


def test_preprocess_call_raise_exception_if_preprocess_output_have_incorrect_type():
    @scared.preprocess
    def wrong(traces):
        return "foo"

    with pytest.raises(scared.PreprocessError):
        wrong(np.array([[1, 2], [3, 4]]))

    @scared.preprocess
    def wrong(traces):
        return 123

    with pytest.raises(scared.PreprocessError):
        wrong(np.array([[1, 2], [3, 4]]))

    @scared.preprocess
    def wrong(traces):
        return traces.sum(axis=0)

    with pytest.raises(scared.PreprocessError):
        wrong(np.array([[1, 2], [3, 4]]))

    @scared.preprocess
    def wrong(traces):
        return traces.swapaxes(0, 1)

    with pytest.raises(scared.PreprocessError):
        wrong(np.array([[1, 2, 3], [3, 4, 5]]))


def test_preprocess_applied():
    @scared.preprocess
    def square(traces):
        return traces ** 2
    data = np.random.randint(0, 255, (500, 2000), dtype='uint8')
    res = square(data)
    assert np.array_equal(res, data ** 2)


def test_preprocess_subclass_raises_exception_if_callable_missing():
    class TPrepro(scared.Preprocess):
        pass
    with pytest.raises(TypeError):
        TPrepro()


def test_preprocess_subclass_applied():
    class Square(scared.Preprocess):

        def __call__(self, traces):
            return traces ** 2
    square = Square()
    data = np.random.randint(0, 255, (500, 2000), dtype='uint8')
    res = square(data)
    assert np.array_equal(res, data ** 2)


def test_preprocess_subclass_call_raise_exception_if_preprocess_output_have_incorrect_type():
    class Wrong(scared.Preprocess):
        def __call__(self, traces):
            return "foo"
    wrong = Wrong()
    with pytest.raises(scared.PreprocessError):
        wrong(np.array([[1, 2], [3, 4]]))

    class Wrong(scared.Preprocess):
        def __call__(self, traces):
            return 123
    wrong = Wrong()

    with pytest.raises(scared.PreprocessError):
        wrong(np.array([[1, 2], [3, 4]]))

    class Wrong(scared.Preprocess):
        def __call__(self, traces):
            return traces.sum(axis=0)
    wrong = Wrong()

    with pytest.raises(scared.PreprocessError):
        wrong(np.array([[1, 2], [3, 4]]))

    class Wrong(scared.Preprocess):
        def __call__(self, traces):
            return traces.swapaxes(0, 1)
    wrong = Wrong()

    with pytest.raises(scared.PreprocessError):
        wrong(np.array([[1, 2, 3], [3, 4, 5]]))
