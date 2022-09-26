from .context import scared  # noqa: F401
import pytest
import numpy as np
import warnings
import psutil


@pytest.fixture
def ths():
    shape = (200, 33)
    samples = np.random.randint(0, 255, shape, dtype='uint8')
    plaintext = np.random.randint(0, 255, (shape[0], 16), dtype='uint8')
    return scared.traces.formats.read_ths_from_ram(samples=samples, plaintext=plaintext)


@pytest.fixture(params=[scared.CPAAttack, scared.DPAAttack, scared.ANOVAAttack, scared.NICVAttack, scared.SNRAttack, scared.MIAAttack])
def attack_class(request):
    return request.param


@pytest.fixture(params=[scared.ANOVAAttack, scared.NICVAttack, scared.SNRAttack, scared.MIAAttack])
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

    @property
    def _distinguisher_str(self):
        return 'Dumb'


class DumbAttack(scared.BaseAttack, DumbDistinguisherMixin):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.traces = []
        self.data = []
        self.processed_traces = 0


def test_analysis_run_raises_exceptions_if_ths_container_is_not_a_container(attack_class, sf):
    with pytest.raises(TypeError):
        a = attack_class(
            selection_function=sf,
            model=scared.Monobit(4),
            discriminant=scared.maxabs)
        assert isinstance(str(a), str)
        a.run('foo')


def test_analysis_object_raises_exceptions_if_sf_is_not_a_selection_function(attack_class):
    with pytest.raises(TypeError):
        attack_class(
            selection_function='foo',
            model=scared.HammingWeight(),
            discriminant=scared.maxabs)


def test_analysis_object_raises_exceptions_if_model_is_not_a_proper_model_instance(attack_class, sf):
    with pytest.raises(TypeError):
        attack_class(
            selection_function=sf,
            model='foo',
            discriminant=scared.maxabs)


def test_analysis_object_raises_exceptions_if_discriminant_is_not_a_callable(attack_class, sf):
    with pytest.raises(TypeError):
        attack_class(
            selection_function=sf,
            model=scared.HammingWeight(),
            discriminant='foof')


def test_base_analysis_cant_be_used_without_subclassing_a_distinguisher_mixin(sf):
    with pytest.raises(NotImplementedError):
        scared.BaseAttack(
            selection_function=sf,
            model=scared.HammingWeight(),
            discriminant=scared.maxabs)


def test_analysis_object_compute_intermediate_values(sf, container):
    analysis = DumbAttack(
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
    assert isinstance(str(analysis), str)


def test_analysis_object_process_traces_batch(sf, container):
    analysis = DumbAttack(
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
    assert isinstance(str(analysis), str)


def test_analysis_object_compute_results(sf, container):
    analysis = DumbAttack(
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
    assert isinstance(str(analysis), str)


def test_cpa_analysis_raises_exception_if_estimated_memory_usage_is_90_percent_of_available_memory(sf):
    analysis = scared.CPAAttack(
        selection_function=sf,
        model=scared.Monobit(5),
        discriminant=scared.nanmax
    )
    available_memory = psutil.virtual_memory().available
    trace_size = int((available_memory * 0.95) / (analysis.precision.itemsize * 4096 * 2))
    samples = np.random.randint(0, 255, (200, trace_size), dtype='uint8')
    plaintext = np.random.randint(0, 255, (200, 16), dtype='uint8')
    ths = scared.traces.formats.read_ths_from_ram(samples=samples, plaintext=plaintext)
    container = scared.Container(ths)
    assert isinstance(str(analysis), str)
    with pytest.raises(scared.DistinguisherError):
        analysis.run(container)


def test_dpa_analysis_raises_exception_if_estimated_memory_usage_is_90_percent_of_available_memory(sf):
    analysis = scared.DPAAttack(
        selection_function=sf,
        model=scared.Monobit(5),
        discriminant=scared.nanmax
    )
    available_memory = psutil.virtual_memory().available
    trace_size = int((available_memory * 0.95) / (analysis.precision.itemsize * 4096 * 2))
    samples = np.random.randint(0, 255, (200, trace_size), dtype='uint8')
    plaintext = np.random.randint(0, 255, (200, 16), dtype='uint8')
    ths = scared.traces.formats.read_ths_from_ram(samples=samples, plaintext=plaintext)
    container = scared.Container(ths)
    assert isinstance(str(analysis), str)
    with pytest.raises(scared.DistinguisherError):
        analysis.run(container)


def test_anova_analysis_raises_exception_if_estimated_memory_usage_is_90_percent_of_available_memory(sf):
    analysis = scared.ANOVAAttack(
        selection_function=sf,
        model=scared.Monobit(5),
        discriminant=scared.nanmax
    )
    available_memory = psutil.virtual_memory().available
    trace_size = int((available_memory * 0.95) / (analysis.precision.itemsize * 4096 * 3 * 2))
    samples = np.random.randint(0, 255, (200, trace_size), dtype='uint8')
    plaintext = np.random.randint(0, 255, (200, 16), dtype='uint8')
    ths = scared.traces.formats.read_ths_from_ram(samples=samples, plaintext=plaintext)
    container = scared.Container(ths)
    assert isinstance(str(analysis), str)
    with pytest.raises(scared.DistinguisherError):
        analysis.run(container)


def test_nicv_analysis_raises_exception_if_estimated_memory_usage_is_90_percent_of_available_memory(sf):
    analysis = scared.NICVAttack(
        selection_function=sf,
        model=scared.Monobit(5),
        discriminant=scared.nanmax
    )
    available_memory = psutil.virtual_memory().available
    trace_size = int((available_memory * 0.95) / (analysis.precision.itemsize * 4096 * 3 * 2))
    samples = np.random.randint(0, 255, (200, trace_size), dtype='uint8')
    plaintext = np.random.randint(0, 255, (200, 16), dtype='uint8')
    ths = scared.traces.formats.read_ths_from_ram(samples=samples, plaintext=plaintext)
    container = scared.Container(ths)
    assert isinstance(str(analysis), str)
    with pytest.raises(scared.DistinguisherError):
        analysis.run(container)


def test_snr_analysis_raises_exception_if_estimated_memory_usage_is_90_percent_of_available_memory(sf):
    analysis = scared.SNRAttack(
        selection_function=sf,
        model=scared.Monobit(5),
        discriminant=scared.nanmax
    )
    available_memory = psutil.virtual_memory().available
    trace_size = int((available_memory * 0.95) / (analysis.precision.itemsize * 4096 * 3 * 2))
    samples = np.random.randint(0, 255, (200, trace_size), dtype='uint8')
    plaintext = np.random.randint(0, 255, (200, 16), dtype='uint8')
    ths = scared.traces.formats.read_ths_from_ram(samples=samples, plaintext=plaintext)
    container = scared.Container(ths)
    assert isinstance(str(analysis), str)
    with pytest.raises(scared.DistinguisherError):
        analysis.run(container)


def test_mia_analysis_raises_exception_if_estimated_memory_usage_is_90_percent_of_available_memory(sf):
    analysis = scared.MIAAttack(
        selection_function=sf,
        model=scared.HammingWeight(),
        discriminant=scared.nanmax,
    )
    available_memory = psutil.virtual_memory().available
    trace_size = int((available_memory * 0.95) / (analysis.precision.itemsize * 4096 * 3 * 9 * 128))
    samples = np.random.randint(0, 255, (200, trace_size), dtype='uint8')
    plaintext = np.random.randint(0, 255, (200, 16), dtype='uint8')
    ths = scared.traces.formats.read_ths_from_ram(samples=samples, plaintext=plaintext)
    container = scared.Container(ths)
    assert isinstance(str(analysis), str)
    with pytest.raises(scared.DistinguisherError):
        analysis.run(container)


def test_analysis_object_run_method(sf, container):
    analysis = DumbAttack(
        selection_function=sf,
        model=scared.Monobit(5),
        discriminant=scared.nanmax
    )
    analysis.run(container)
    assert (256, 16, 200) == analysis.results.shape
    assert (256, 16) == analysis.scores.shape
    assert np.array_equal(np.nanmax(analysis.results, axis=-1), analysis.scores)
    assert isinstance(str(analysis), str)


def test_dpa_analysis_raise_exception_if_init_with_not_monobit_model(sf):
    with pytest.raises(scared.distinguishers.DistinguisherError):
        scared.DPAAttack(
            model=scared.Value(),
            selection_function=sf,
            discriminant=scared.nanmax
        )


def test_analysis_object_run_method_with_frame(sf, ths):
    analysis = DumbAttack(
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
    assert isinstance(str(analysis), str)


def test_analysis_run_raise_exceptions_if_inconsistent_traces_size_are_used_between_two_process(sf, container):
    analysis = scared.DPAAttack(
        selection_function=sf,
        model=scared.Monobit(3),
        discriminant=scared.nanmax
    )

    container.frame = slice(20, 30)
    analysis.run(container)
    container.frame = slice(20, None)
    assert isinstance(str(analysis), str)
    with pytest.raises(scared.DistinguisherError):
        analysis.run(container)

    analysis = scared.CPAAttack(
        selection_function=sf,
        model=scared.HammingWeight(),
        discriminant=scared.nanmax
    )

    warnings.simplefilter('ignore', RuntimeWarning)
    container.frame = slice(10, 20)
    analysis.run(container)
    container.frame = slice(20, None)
    assert isinstance(str(analysis), str)
    with pytest.raises(scared.DistinguisherError):
        analysis.run(container)


def test_analysis_object_run_method_with_convergence_traces(sf, container):
    analysis = DumbAttack(
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
    assert isinstance(str(analysis), str)

    analysis = DumbAttack(
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
    assert isinstance(str(analysis), str)


def test_analysis_object_run_method_with_convergence_step_higher_than_number_of_traces(sf, container):
    analysis = DumbAttack(
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
    assert isinstance(str(analysis), str)


def test_analysis_object_run_method_with_convergence_step_larger_than_batch_size(sf):
    analysis = DumbAttack(
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
    assert isinstance(str(analysis), str)

    analysis = DumbAttack(
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
    assert isinstance(str(analysis), str)

    analysis = DumbAttack(
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
    assert isinstance(str(analysis), str)

    analysis = DumbAttack(
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
    assert isinstance(str(analysis), str)


def test_analysis_raise_exception_if_convergence_step_is_not_positive_integer(sf, attack_class):
    with pytest.raises(TypeError):
        attack_class(
            selection_function=sf,
            model=scared.Monobit(5),
            discriminant=scared.nanmax,
            convergence_step='foo'
        )
    with pytest.raises(ValueError):
        attack_class(
            selection_function=sf,
            model=scared.Monobit(5),
            discriminant=scared.nanmax,
            convergence_step=0
        )
    with pytest.raises(ValueError):
        attack_class(
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
    with pytest.raises(ValueError):
        partitioned_klass(selection_function=sf, model=scared.HammingWeight(), discriminant=scared.maxabs, partitions=np.array([1.2, 3]))


def test_partitioned_analysis_set_partition(sf, partitioned_klass):
    a = partitioned_klass(selection_function=sf, model=scared.HammingWeight(), discriminant=scared.maxabs, partitions=np.arange(9))
    assert np.array_equal(np.arange(9), a.partitions)
    assert isinstance(str(a), str)


def test_mia_analysis_raises_excerptions_if_incorrect_histos_parameters(sf):
    with pytest.raises(TypeError):
        scared.MIAAttack(selection_function=sf, model=scared.HammingWeight(), discriminant=scared.maxabs, bins_number='foo')
    with pytest.raises(TypeError):
        scared.MIAAttack(selection_function=sf, model=scared.HammingWeight(), discriminant=scared.maxabs, bins_number={})
    with pytest.raises(TypeError):
        scared.MIAAttack(selection_function=sf, model=scared.HammingWeight(), discriminant=scared.maxabs, bins_number=[1, 23])
    with pytest.raises(TypeError):
        scared.MIAAttack(selection_function=sf, model=scared.HammingWeight(), discriminant=scared.maxabs, bins_number=np.array([1.2, 3]))
    with pytest.raises(TypeError):
        scared.MIAAttack(selection_function=sf, model=scared.HammingWeight(), discriminant=scared.maxabs, bins_number=38.45)


_bin_edges_fail = {
    'bin number': 12,
    'np method': 'auto',
    'No edges': None
}


@pytest.fixture(params=_bin_edges_fail.keys())
def bin_edges_fail_key(request):
    return request.param


def test_mia_with_invalid_bin_edges_raises_exception(sf, bin_edges_fail_key):
    bin_edges = _bin_edges_fail[bin_edges_fail_key]
    with pytest.raises(TypeError):
        d = scared.MIAAttack(selection_function=sf, model=scared.HammingWeight(), discriminant=scared.maxabs)
        d.bin_edges = bin_edges
        assert isinstance(str(d), str)

    with pytest.raises(TypeError):
        d = scared.MIAAttack(bin_edges=bin_edges, selection_function=sf, model=scared.HammingWeight(), discriminant=scared.maxabs)
        d.bin_edges = bin_edges
        assert isinstance(str(d), str)


def test_mia_bin_edges_init(sf):
    a = scared.MIAAttack(bin_edges=np.arange(258), selection_function=sf, model=scared.HammingWeight(), discriminant=scared.abssum)
    assert np.array_equal(a.bin_edges, np.arange(258))
    assert isinstance(str(a), str)
