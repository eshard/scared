from .context import scared  # noqa: F401
import pytest
import numpy as np
import warnings


@pytest.fixture
def ths():
    shape = (200, 33)
    samples = np.random.randint(0, 255, shape, dtype='uint8')
    plaintext = np.random.randint(0, 255, (shape[0], 16), dtype='uint8')
    return scared.traces.formats.read_ths_from_ram(samples=samples, plaintext=plaintext)


@pytest.fixture(params=[scared.CPAAnalysis, scared.DPAAnalysis, scared.ANOVAAnalysis, scared.NICVAnalysis, scared.SNRAnalysis])
def analysis_class(request):
    return request.param


@pytest.fixture(params=[scared.ANOVAAnalysis, scared.NICVAnalysis, scared.SNRAnalysis])
def partitioned_klass(request):
    return request.param


@pytest.fixture
def sf():
    def _sf(guesses, plaintext):
        result = np.empty((plaintext.shape[0], len(guesses), 16), dtype='uint8')
        for guess in guesses:
            for byte in range(16):
                result[:, guess, byte] = np.sum(plaintext, axis=1)
        return result
    return scared.attack_selection_function(_sf)


@pytest.fixture
def container(ths):
    return scared.Container(ths)


class DumbDistinguisherMixin(scared.DistinguisherMixin):

    def _initialize(self, traces, data):
        self.traces = []
        self.data = []

    def _update(self, traces, data):
        self.traces.append(traces)
        self.data.append(data)

    def _compute(self):
        try:
            return np.sum(np.array(self.data), axis=0)
        except Exception:
            return np.sum(np.array(self.data[:-1]), axis=0) + np.sum(np.array(self.data[-1]), axis=0)


class DumbAnalysis(scared.BaseAnalysis, DumbDistinguisherMixin):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.traces = []
        self.data = []
        self.processed_traces = 0


def test_analysis_run_raises_exceptions_if_ths_container_is_not_a_container(analysis_class, sf):
    with pytest.raises(TypeError):
        a = analysis_class(
            selection_function=sf,
            model=scared.Monobit(4),
            discriminant=scared.maxabs)
        a.run('foo')


def test_analysis_object_raises_exceptions_if_sf_is_not_a_selection_function(analysis_class):
    with pytest.raises(TypeError):
        analysis_class(
            selection_function='foo',
            model=scared.HammingWeight(),
            discriminant=scared.maxabs)


def test_analysis_object_raises_exceptions_if_model_is_not_a_proper_model_instance(analysis_class, sf):
    with pytest.raises(TypeError):
        analysis_class(
            selection_function=sf,
            model='foo',
            discriminant=scared.maxabs)


def test_analysis_object_raises_exceptions_if_discriminant_is_not_a_callable(analysis_class, sf):
    with pytest.raises(TypeError):
        analysis_class(
            selection_function=sf,
            model=scared.HammingWeight(),
            discriminant='foof')


def test_base_analysis_cant_be_used_without_subclassing_a_distinguisher_mixin(sf):
    with pytest.raises(NotImplementedError):
        scared.BaseAnalysis(
            selection_function=sf,
            model=scared.HammingWeight(),
            discriminant=scared.maxabs)


def test_analysis_object_compute_intermediate_values(sf, container):
    analysis = DumbAnalysis(
        selection_function=sf,
        model=scared.HammingWeight(),
        discriminant=scared.nanmax
    )
    int_val = analysis.compute_intermediate_values(container._ths.metadatas)
    assert (len(container._ths), 256, 16) == int_val.shape
    expected = scared.HammingWeight()(sf(**container._ths.metadatas))
    int_val = int_val.reshape((len(container._ths), 256, 16))
    assert np.array_equal(
        expected, int_val
    )


def test_analysis_object_process_traces_batch(sf, container):
    analysis = DumbAnalysis(
        selection_function=sf,
        model=scared.Monobit(5),
        discriminant=scared.nanmax
    )
    batches = list(container.batches())
    analysis.process(batches[0])
    assert np.array_equal(analysis.traces[0], batches[0].samples[:])
    assert np.array_equal(
        analysis.data[0].reshape(-1, 256, 16),
        analysis.compute_intermediate_values(batches[0].metadatas)
    )


def test_analysis_object_compute_results(sf, container):
    analysis = DumbAnalysis(
        selection_function=sf,
        model=scared.Monobit(5),
        discriminant=scared.nanmax
    )
    batches = list(container.batches())
    analysis.process(batches[0])
    analysis.compute_results()

    assert (256, 16, 200) == analysis.results.shape
    assert (256, 16) == analysis.scores.shape
    assert np.array_equal(np.nanmax(analysis.results, axis=-1), analysis.scores)


def test_analysis_object_run_method(sf, container):
    analysis = DumbAnalysis(
        selection_function=sf,
        model=scared.Monobit(5),
        discriminant=scared.nanmax
    )
    analysis.run(container)
    assert (256, 16, 200) == analysis.results.shape
    assert (256, 16) == analysis.scores.shape
    assert np.array_equal(np.nanmax(analysis.results, axis=-1), analysis.scores)


def test_dpa_analysis_raise_exception_if_init_with_not_monobit_model(sf):
    with pytest.raises(scared.distinguishers.DistinguisherError):
        scared.DPAAnalysis(
            model=scared.Value(),
            selection_function=sf,
            discriminant=scared.nanmax
        )


def test_analysis_object_run_method_with_frame(sf, ths):
    analysis = DumbAnalysis(
        selection_function=sf,
        model=scared.Monobit(5),
        discriminant=scared.nanmax
    )

    container = scared.Container(ths, frame=slice(None, 10))

    analysis.run(container)
    assert (1, 200, 10) == np.array(analysis.traces).shape

    container.frame = slice(20, 30)
    analysis.run(container)
    assert (2, 200, 10) == np.array(analysis.traces).shape


def test_analysis_run_raise_exceptions_if_inconsistent_traces_size_are_used_between_two_process(sf, container):
    analysis = scared.DPAAnalysis(
        selection_function=sf,
        model=scared.Monobit(3),
        discriminant=scared.nanmax
    )

    container.frame = slice(20, 30)
    analysis.run(container)
    container.frame = slice(20, None)
    with pytest.raises(scared.DistinguisherError):
        analysis.run(container)

    analysis = scared.CPAAnalysis(
        selection_function=sf,
        model=scared.HammingWeight(),
        discriminant=scared.nanmax
    )

    warnings.simplefilter('ignore', RuntimeWarning)
    container.frame = slice(10, 20)
    analysis.run(container)
    container.frame = slice(20, None)
    with pytest.raises(scared.DistinguisherError):
        analysis.run(container)


def test_analysis_object_run_method_with_convergence_traces(sf, container):
    analysis = DumbAnalysis(
        selection_function=sf,
        model=scared.Monobit(5),
        discriminant=scared.nanmax,
        convergence_step=50
    )

    analysis.run(container)
    assert (256, 16, 4) == analysis.convergence_traces.shape
    assert (256, 16) == analysis.results.shape[0:2]
    assert (256, 16) == analysis.scores.shape
    assert np.array_equal(analysis.scores, analysis.convergence_traces[:, :, -1])

    analysis = DumbAnalysis(
        selection_function=sf,
        model=scared.Monobit(5),
        discriminant=scared.nanmax,
        convergence_step=30
    )

    analysis.run(container)
    assert (256, 16, 7) == analysis.convergence_traces.shape
    assert (256, 16) == analysis.results.shape[0:2]
    assert (256, 16) == analysis.scores.shape
    assert np.array_equal(analysis.scores, analysis.convergence_traces[:, :, -1])


def test_analysis_object_run_method_with_convergence_step_higher_than_number_of_traces(sf, container):
    analysis = DumbAnalysis(
        selection_function=sf,
        model=scared.Monobit(5),
        discriminant=scared.nanmax,
        convergence_step=300
    )

    analysis.run(container)
    assert (256, 16, 1) == analysis.convergence_traces.shape
    assert (256, 16) == analysis.results.shape[0:2]
    assert (256, 16) == analysis.scores.shape
    assert np.array_equal(analysis.scores, analysis.convergence_traces[:, :, -1])


def test_analysis_object_run_method_with_convergence_step_larger_than_batch_size(sf):
    analysis = DumbAnalysis(
        selection_function=sf,
        model=scared.Monobit(5),
        discriminant=scared.nanmax,
        convergence_step=5000
    )
    samples = np.random.randint(0, 255, (10000, 6000), dtype='uint8')
    plaintext = np.random.randint(0, 255, (10000, 16), dtype='uint8')
    ths = scared.traces.formats.read_ths_from_ram(samples=samples, plaintext=plaintext)
    container = scared.Container(ths)

    analysis.run(container)
    assert (256, 16, 2) == analysis.convergence_traces.shape
    assert (256, 16) == analysis.results.shape[0:2]
    assert (256, 16) == analysis.scores.shape
    assert np.array_equal(analysis.scores, analysis.convergence_traces[:, :, -1])

    analysis = DumbAnalysis(
        selection_function=sf,
        model=scared.Monobit(5),
        discriminant=scared.nanmax,
        convergence_step=6000
    )
    analysis.run(container)
    assert (256, 16, 2) == analysis.convergence_traces.shape
    assert (256, 16) == analysis.results.shape[0:2]
    assert (256, 16) == analysis.scores.shape
    assert np.array_equal(analysis.scores, analysis.convergence_traces[:, :, -1])

    analysis = DumbAnalysis(
        selection_function=sf,
        model=scared.Monobit(5),
        discriminant=scared.nanmax,
        convergence_step=3000
    )
    analysis.run(container)
    assert (256, 16, 4) == analysis.convergence_traces.shape
    assert (256, 16) == analysis.results.shape[0:2]
    assert (256, 16) == analysis.scores.shape
    assert np.array_equal(analysis.scores, analysis.convergence_traces[:, :, -1])

    analysis = DumbAnalysis(
        selection_function=sf,
        model=scared.Monobit(5),
        discriminant=scared.nanmax,
        convergence_step=4250
    )
    analysis.run(container)
    assert (256, 16, 3) == analysis.convergence_traces.shape
    assert (256, 16) == analysis.results.shape[0:2]
    assert (256, 16) == analysis.scores.shape
    assert np.array_equal(analysis.scores, analysis.convergence_traces[:, :, -1])


def test_analysis_raise_exception_if_convergence_step_is_not_positive_integer(sf, analysis_class):
    with pytest.raises(TypeError):
        analysis_class(
            selection_function=sf,
            model=scared.Monobit(5),
            discriminant=scared.nanmax,
            convergence_step='foo'
        )
    with pytest.raises(ValueError):
        analysis_class(
            selection_function=sf,
            model=scared.Monobit(5),
            discriminant=scared.nanmax,
            convergence_step=0
        )
    with pytest.raises(ValueError):
        analysis_class(
            selection_function=sf,
            model=scared.Monobit(5),
            discriminant=scared.nanmax,
            convergence_step=-12
        )


def test_partitioned_analysis_raises_exception_if_incorrect_partition(sf, partitioned_klass):
    with pytest.raises(TypeError):
        partitioned_klass(selection_function=sf, model=scared.HammingWeight(), discriminant=scared.maxabs, partitions='foo')
    with pytest.raises(TypeError):
        partitioned_klass(selection_function=sf, model=scared.HammingWeight(), discriminant=scared.maxabs, partitions={})
    with pytest.raises(TypeError):
        partitioned_klass(selection_function=sf, model=scared.HammingWeight(), discriminant=scared.maxabs, partitions=[1, 23])
    with pytest.raises(ValueError):
        partitioned_klass(selection_function=sf, model=scared.HammingWeight(), discriminant=scared.maxabs, partitions=np.array([1.2, 3]))


def test_partitioned_analysis_set_partition(sf, partitioned_klass):
    a = partitioned_klass(selection_function=sf, model=scared.HammingWeight(), discriminant=scared.maxabs, partitions=np.arange(9))
    assert np.array_equal(np.arange(9), a.partitions)
