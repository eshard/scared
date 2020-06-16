from .context import scared
import pytest
import numpy as np


@pytest.fixture
def template_datas():
    datas = np.load('tests/samples/tests_samples_template.npz')
    for k, v in datas.items():
        setattr(datas, k, v)
    return datas


@pytest.fixture
def dpa_template_datas():
    datas = np.load('tests/samples/tests_samples_dpa_template.npz')
    for k, v in datas.items():
        setattr(datas, k, v)
    return datas


@pytest.fixture
def ths():
    shape = (1000, 500)
    samples = np.random.randint(0, 255, shape, dtype='uint8')
    plaintext = np.random.randint(0, 255, (shape[0], 16), dtype='uint8')
    key = np.random.randint(0, 255, (shape[0], 16), dtype='uint8')
    return scared.traces.formats.read_ths_from_ram(samples=samples, plaintext=plaintext, key=key)


@pytest.fixture
def long_ths():
    shape = (25000, 3)
    samples = np.random.randint(0, 256, shape, dtype='uint8')
    plaintext = np.random.randint(0, 256, (shape[0], 1), dtype='uint8')
    return scared.traces.formats.read_ths_from_ram(samples=np.vstack((samples, samples)),
                                                   plaintext=np.vstack((plaintext, plaintext)))


@pytest.fixture
def container(ths):
    return scared.Container(ths)


@pytest.fixture
def building_container(ths):
    return scared.Container(ths)


@pytest.fixture
def sf():
    def _sf(key, plaintext):
        return scared.aes.encrypt(plaintext=plaintext, key=key, at_round=1, after_step=scared.aes.Steps.SUB_BYTES)
    return scared.selection_function(_sf, words=0)


@pytest.fixture(params=[scared.TemplateAttack, scared.TemplateDPAAttack])
def template_klass(request):
    return request.param


def test_template_raises_exception_if_incorrect_container_build_provided(template_klass, sf):
    with pytest.raises(TypeError):
        template_klass(
            container_building='foo',
            selection_function=sf,
            reverse_selection_function=sf,
            model=scared.HammingWeight()

        )
    with pytest.raises(TypeError):
        template_klass(
            container_building={1: 23},
            selection_function=sf,
            reverse_selection_function=sf,
            model=scared.HammingWeight()
        )
    with pytest.raises(TypeError):
        template_klass(
            container_building=12,
            selection_function=sf,
            reverse_selection_function=sf,
            model=scared.HammingWeight()
        )


def test_template_raises_exceptions_if_sf_is_not_a_selection_function(sf, template_klass, building_container):
    with pytest.raises(TypeError):
        template_klass(
            container_building=building_container,
            reverse_selection_function='foo',
            model=scared.HammingWeight(),
            selection_function=sf
        )


def test_template_raises_exceptions_if_model_is_not_a_proper_model_instance(sf, template_klass, building_container):
    with pytest.raises(TypeError):
        template_klass(
            container_building=building_container,
            reverse_selection_function=sf,
            model='foo',
            selection_function=sf
        )


def test_analysis_raise_exception_if_convergence_step_is_not_positive_integer(template_klass, sf, building_container):
    with pytest.raises(TypeError):
        template_klass(
            container_building=building_container,
            reverse_selection_function=sf,
            model=scared.Monobit(5),
            convergence_step='foo',
            selection_function=sf
        )
    with pytest.raises(ValueError):
        template_klass(
            container_building=building_container,
            reverse_selection_function=sf,
            model=scared.Monobit(5),
            convergence_step=0,
            selection_function=sf
        )
    with pytest.raises(ValueError):
        template_klass(
            container_building=building_container,
            reverse_selection_function=sf,
            model=scared.Monobit(5),
            convergence_step=-12,
            selection_function=sf
        )


def test_template_analysis_raises_exception_if_incorrect_partition(template_klass, sf, building_container):
    with pytest.raises(TypeError):
        template_klass(
            container_building=building_container,
            reverse_selection_function=sf,
            model=scared.HammingWeight(),
            partitions='foo',
            selection_function=sf
        )
    with pytest.raises(TypeError):
        template_klass(
            container_building=building_container,
            reverse_selection_function=sf,
            model=scared.HammingWeight(),
            partitions={},
            selection_function=sf
        )
    with pytest.raises(TypeError):
        template_klass(
            container_building=building_container,
            reverse_selection_function=sf,
            model=scared.HammingWeight(),
            partitions=[1, 23],
            selection_function=sf
        )
    with pytest.raises(ValueError):
        template_klass(
            container_building=building_container,
            reverse_selection_function=sf,
            model=scared.HammingWeight(),
            partitions=np.array([1.2, 3]),
            selection_function=sf
        )


def test_template_set_partition(template_klass, sf, building_container):
    a = template_klass(
        container_building=building_container,
        reverse_selection_function=sf,
        model=scared.HammingWeight(),
        partitions=np.arange(9),
        selection_function=sf
    )
    assert np.array_equal(np.arange(9), a.partitions)
    assert isinstance(str(a), str)


def test_template_raises_exception_when_passing_a_non_consistent_precision(template_klass, sf, building_container):

    with pytest.raises(TypeError):
        template_klass(
            container_building=building_container,
            reverse_selection_function=sf,
            model=scared.Value(),
            precision='foo',
            selection_function=sf
        )

    with pytest.raises(ValueError):
        template_klass(
            container_building=building_container,
            reverse_selection_function=sf,
            model=scared.Value(),
            precision='uint8',
            selection_function=sf
        )


def test_template_has_float32_dtype_default_precision(template_klass, sf, building_container):
    s = template_klass(
        container_building=building_container,
        reverse_selection_function=sf,
        model=scared.Value(),
        selection_function=sf
    )
    assert s.precision == np.dtype('float32')


def test_template_raises_exception_if_selection_function_returns_more_than_one_word(template_klass, building_container):
    @scared.selection_function
    def first_sub(key, data):
        return scared.aes.encrypt(plaintext=data, key=key, at_round=1, after_step=scared.aes.Steps.SUB_BYTES)

    with pytest.raises(scared.SelectionFunctionError):
        template_klass(
            container_building=building_container,
            reverse_selection_function=first_sub,
            model=scared.Value(),
            selection_function=scared.aes.selection_functions.encrypt.FirstSubBytes()
        )


def test_template_run_raises_exception_if_building_not_done(template_klass, sf, building_container, container):
    template = template_klass(
        container_building=building_container,
        reverse_selection_function=sf,
        model=scared.Value(),
        selection_function=sf
    )
    with pytest.raises(scared.DistinguisherError):
        template.run(container)


def test_template_build_phase(sf, template_datas):
    ths_building = scared.traces.formats.read_ths_from_ram(
        template_datas.building_samples,
        plaintext=template_datas.building_plaintext,
        key=np.array(
            [
                template_datas.building_key for i in range(
                    len(template_datas.building_samples)
                )
            ]
        )
    )
    building_cont = scared.Container(
        ths=ths_building
    )
    template = scared.TemplateAttack(
        container_building=building_cont,
        reverse_selection_function=sf,
        model=scared.Value()
    )
    template.build()
    assert np.array_equal(template._build_analysis._exi, template_datas.exi)
    assert np.array_equal(template._build_analysis._counters, template_datas.counters)
    assert np.array_equal(template._build_analysis._exxi, template_datas.exxi)
    assert np.allclose(template._build_analysis.pooled_covariance, template_datas.pooled_cov)
    assert np.allclose(template._build_analysis.pooled_covariance_inv, template_datas.pooled_cov_inv, )
    assert np.array_equal(template.templates, template_datas.templates)


def test_template_matching_phase(sf, template_datas):
    ths_building = scared.traces.formats.read_ths_from_ram(
        template_datas.building_samples,
        plaintext=template_datas.building_plaintext,
        key=np.array(
            [
                template_datas.building_key for i in range(
                    len(template_datas.building_samples)
                )
            ]
        )
    )
    building_cont = scared.Container(
        ths=ths_building
    )
    template = scared.TemplateAttack(
        container_building=building_cont,
        reverse_selection_function=sf,
        model=scared.Value(),
        convergence_step=1
    )
    template.build()
    ths_matching = scared.traces.formats.read_ths_from_ram(
        template_datas.matching_samples,
        plaintext=template_datas.matching_plaintext,
        key=np.array(
            [
                template_datas.matching_key for i in range(
                    len(template_datas.matching_samples)
                )
            ]
        )
    )
    matching_cont = scared.Container(
        ths=ths_matching
    )
    template.run(matching_cont)
    assert np.allclose(template.results, template_datas.scores)
    assert np.allclose(template.convergence_traces, template_datas.conv_traces)


def test_template_matching_phase_raises_exception_if_incorrect_trace_size(sf, template_datas):
    ths_building = scared.traces.formats.read_ths_from_ram(
        template_datas.building_samples,
        plaintext=template_datas.building_plaintext,
        key=np.array(
            [
                template_datas.building_key for i in range(
                    len(template_datas.building_samples)
                )
            ]
        )
    )
    building_cont = scared.Container(
        ths=ths_building
    )
    template = scared.TemplateAttack(
        container_building=building_cont,
        reverse_selection_function=sf,
        model=scared.Value(),
        convergence_step=1
    )
    template.build()
    ths_matching = scared.traces.formats.read_ths_from_ram(
        template_datas.matching_samples,
        plaintext=template_datas.matching_plaintext,
        key=np.array(
            [
                template_datas.matching_key for i in range(
                    len(template_datas.matching_samples)
                )
            ]
        )
    )
    matching_cont = scared.Container(
        ths=ths_matching,
        frame=slice(0, 10)
    )
    with pytest.raises(scared.DistinguisherError):
        template.run(matching_cont)


def test_template_dpa_raises_exceptions_if_matching_sf_is_not_an_attack_selection_function(sf, building_container):
    with pytest.raises(TypeError):
        scared.TemplateDPAAttack(
            container_building=building_container,
            reverse_selection_function=sf,
            selection_function='foo',
            model=scared.HammingWeight()
        )


def test_dpa_template_build_phase(sf, dpa_template_datas):
    ths_building = scared.traces.formats.read_ths_from_ram(
        dpa_template_datas.building_samples,
        plaintext=dpa_template_datas.building_plaintext,
        key=np.array(
            [
                dpa_template_datas.building_key for i in range(
                    len(dpa_template_datas.building_samples)
                )
            ]
        )
    )
    building_cont = scared.Container(
        ths=ths_building
    )
    template = scared.TemplateDPAAttack(
        container_building=building_cont,
        reverse_selection_function=sf,
        selection_function=scared.aes.selection_functions.encrypt.FirstSubBytes(words=0),
        model=scared.HammingWeight()
    )
    template.build()
    assert np.array_equal(template._build_analysis._exi, dpa_template_datas.exi)
    assert np.array_equal(template._build_analysis._counters, dpa_template_datas.counters)
    assert np.array_equal(template._build_analysis._exxi, dpa_template_datas.exxi)
    assert np.allclose(template._build_analysis.pooled_covariance, dpa_template_datas.pooled_cov)
    assert np.allclose(template._build_analysis.pooled_covariance_inv, dpa_template_datas.pooled_cov_inv, )
    assert np.array_equal(template.templates, dpa_template_datas.templates)


def test_dpa_template_matching_phase(sf, dpa_template_datas):
    ths_building = scared.traces.formats.read_ths_from_ram(
        dpa_template_datas.building_samples,
        plaintext=dpa_template_datas.building_plaintext,
        key=np.array(
            [
                dpa_template_datas.building_key for i in range(
                    len(dpa_template_datas.building_samples)
                )
            ]
        )
    )
    building_cont = scared.Container(
        ths=ths_building
    )
    template = scared.TemplateDPAAttack(
        container_building=building_cont,
        reverse_selection_function=sf,
        selection_function=scared.aes.selection_functions.encrypt.FirstSubBytes(words=0),
        model=scared.HammingWeight(),
        convergence_step=1
    )
    template.build()
    ths_matching = scared.traces.formats.read_ths_from_ram(
        dpa_template_datas.matching_samples,
        plaintext=dpa_template_datas.matching_plaintext,
        key=np.array(
            [
                dpa_template_datas.matching_key for i in range(
                    len(dpa_template_datas.matching_samples)
                )
            ]
        )
    )
    matching_cont = scared.Container(
        ths=ths_matching
    )
    template.run(matching_cont)
    assert np.allclose(template.results, dpa_template_datas.scores)
    assert np.allclose(template.convergence_traces, dpa_template_datas.conv_traces)


def test_dpa_template_matching_phase_raises_exception_if_incorrect_trace_size(sf, dpa_template_datas):
    ths_building = scared.traces.formats.read_ths_from_ram(
        dpa_template_datas.building_samples,
        plaintext=dpa_template_datas.building_plaintext,
        key=np.array(
            [
                dpa_template_datas.building_key for i in range(
                    len(dpa_template_datas.building_samples)
                )
            ]
        )
    )
    building_cont = scared.Container(
        ths=ths_building
    )
    template = scared.TemplateDPAAttack(
        container_building=building_cont,
        reverse_selection_function=sf,
        selection_function=scared.aes.selection_functions.encrypt.FirstSubBytes(words=0),
        model=scared.HammingWeight(),
        convergence_step=1
    )
    template.build()
    ths_matching = scared.traces.formats.read_ths_from_ram(
        dpa_template_datas.matching_samples,
        plaintext=dpa_template_datas.matching_plaintext,
        key=np.array(
            [
                dpa_template_datas.matching_key for i in range(
                    len(dpa_template_datas.matching_samples)
                )
            ]
        )
    )
    matching_cont = scared.Container(
        ths=ths_matching,
        frame=slice(0, 10)
    )
    with pytest.raises(scared.DistinguisherError):
        template.run(matching_cont)


def test_templates_correct_with_two_batches(long_ths):
    # The ths is composed of two identical halves thus the templates should be the same
    # with the whole ths or just the first half
    sf = scared.selection_function(lambda plaintext: plaintext, words=0)
    container1 = scared.Container(ths=long_ths[:25000])
    template1 = scared.TemplateAttack(container_building=container1,
                                      reverse_selection_function=sf,
                                      model=scared.Value(),
                                      convergence_step=1)
    template1.build()
    container2 = scared.Container(ths=long_ths)
    template2 = scared.TemplateAttack(container_building=container2,
                                      reverse_selection_function=sf,
                                      model=scared.Value(),
                                      convergence_step=1)
    template2.build()

    assert np.array_equiv(template1.templates, template2.templates)
