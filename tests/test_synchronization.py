import os
import warnings
import shutil

import estraces
import numpy as np
from pathlib import Path
import pytest

from .context import scared


WORKING_DIRECTORY = 'tests/samples_workdir/'
SAMPLE_DIRECTORY = 'tests/samples/'


def test_error_counter_warns_every_consecutive_error_with_limit_doubling_each_time():
    warn_counter = 0
    with warnings.catch_warnings():
        warnings.filterwarnings(action='error')
        ec = scared.synchronization._ErrorCounter()

        for i in range(100):
            if i != 50:
                try:
                    ec.error_occur(i)
                except UserWarning:
                    warn_counter += 1

            if i < 7:
                assert warn_counter == 0
            elif i < 15:
                assert warn_counter == 1
            elif i < 31:
                assert warn_counter == 2
            else:
                assert warn_counter == 3


@pytest.fixture
def sample_directory():
    shutil.copytree(SAMPLE_DIRECTORY, WORKING_DIRECTORY)
    yield WORKING_DIRECTORY
    shutil.rmtree(WORKING_DIRECTORY)


@pytest.fixture
def output_filename():
    file_name = "tests/samples/synchronization/synced.ets"
    if os.path.exists(file_name):
        os.remove(file_name)
    yield file_name
    if os.path.exists(file_name):
        os.remove(file_name)


def test_synchronizer_run_with_ets_file_applies_given_function_and_returns_correct_ths(sample_directory, output_filename):
    def sync_function(trace_object):
        return trace_object.samples.array + 0.5

    input_filename = f"{sample_directory}/synchronization/ets_file.ets"
    ths = estraces.read_ths_from_ets_file(input_filename)[:10]

    synchronizer = scared.Synchronizer(ths, output_filename, sync_function)
    out_ths = synchronizer.run()
    assert len(out_ths) == len(ths)
    for i in range(len(ths)):
        trace = ths[i]
        output_trace = out_ths[i]
        assert np.array_equal(trace.samples, output_trace.samples.array - 0.5)
        assert np.array_equal(trace.plaintext, output_trace.plaintext)
        assert np.array_equal(trace.ciphertext, output_trace.ciphertext)
        assert np.array_equal(trace.foo_bar, output_trace.foo_bar)
    ths.close()


def test_synchronizer_run_with_binary_file_applies_given_function_and_returns_correct_ths(sample_directory, output_filename):
    def sync_function(trace_object):
        return trace_object.samples.array + 0.5

    input_text_filename = f"{sample_directory}/synchronization/binary_text_file.txt"
    text_128 = estraces.formats.bin_extractor.FilePatternExtractor(input_text_filename, r"([a-fA-F0-9]{128})", num=0)
    text_64 = estraces.formats.bin_extractor.FilePatternExtractor(input_text_filename, r"([a-fA-F0-9]{64})", num=2)
    input_binary_filenames = f"{sample_directory}/synchronization/aes_binary.*"
    ths = estraces.read_ths_from_bin_filenames_pattern(input_binary_filenames, dtype='uint8', metadatas_parsers={"text1": text_128, "text2": text_64})

    synchronizer = scared.Synchronizer(ths, output_filename, sync_function)
    out_ths = synchronizer.run()
    assert len(out_ths) == len(ths)
    for i in range(len(ths)):
        trace = ths[i]
        output_trace = out_ths[i]
        assert np.array_equal(trace.samples, output_trace.samples.array - 0.5)
        assert np.array_equal(trace.text1, output_trace.text1)
        assert np.array_equal(trace.text2, output_trace.text2)
    ths.close()


def test_synchronizer_run_returns_only_values_that_are_returned_by_the_given_function_without_errors(sample_directory, output_filename):
    def sync_function(trace_object):
        if trace_object.name == 'Trace n°0' or trace_object.name == 'Trace n°1' or trace_object.name == 'Trace n°3':
            return trace_object.samples.array
        if trace_object.name == 'Trace n°2':
            raise scared.ResynchroError('Error.')

    input_filename = f"{sample_directory}/synchronization/ets_file.ets"
    ths = estraces.read_ths_from_ets_file(input_filename)[:10]

    synchronizer = scared.Synchronizer(ths, output_filename, sync_function)
    out_ths = synchronizer.run()
    assert len(out_ths) == 3
    ths.close()


def test_synchronizer_run_returns_traces_preserving_its_added_attributes(sample_directory, output_filename):
    def sync_function(trace_object):
        trace_object.name2 = trace_object.name
        return trace_object.samples.array

    input_filename = f"{sample_directory}/synchronization/ets_file.ets"
    ths = estraces.read_ths_from_ets_file(input_filename)[:10]

    synchronizer = scared.Synchronizer(ths, output_filename, sync_function)
    out_ths = synchronizer.run()
    for i in range(len(ths)):
        trace = ths[i]
        output_trace = out_ths[i]
        assert trace.name == output_trace.name2
    ths.close()


def test_synchronizer_run_passes_kwargs_to_the_synchronization_function(sample_directory, output_filename):
    def sync_function(trace_object, foo, bar):
        trace_object.foo = foo
        trace_object.bar = bar
        return trace_object.samples.array

    input_filename = f"{sample_directory}/synchronization/ets_file.ets"
    ths = estraces.read_ths_from_ets_file(input_filename)[:10]

    synchronizer = scared.Synchronizer(ths, output_filename, sync_function, foo=10, bar=20)
    out_ths = synchronizer.run()
    for i in range(len(ths)):
        output_trace = out_ths[i]
        assert output_trace.foo == 10
        assert output_trace.bar == 20
    ths.close()


def test_synchronizer_check_prints_exception_with_function_raising_exception_and_catch_exceptions_to_true(sample_directory, capsys, output_filename):
    def sync_function(trace_object):
        raise scared.ResynchroError('Error.')

    input_filename = f"{sample_directory}/synchronization/ets_file.ets"
    ths = estraces.read_ths_from_ets_file(input_filename)[:10]

    synchronizer = scared.Synchronizer(ths, output_filename, sync_function)
    synchronizer.check(nb_traces=1, catch_exceptions=True)
    captured = capsys.readouterr()
    assert captured.out.startswith("Raised scared.synchronization.ResynchroError: Error. in sync_function line")
    ths.close()


def test_synchronizer_check_raises_exception_with_function_raising_exception_and_catch_exceptions_to_false(sample_directory, capsys, output_filename):
    def sync_function(trace_object):
        raise scared.ResynchroError('Error.')

    input_filename = f"{sample_directory}/synchronization/ets_file.ets"
    ths = estraces.read_ths_from_ets_file(input_filename)[:10]

    synchronizer = scared.Synchronizer(ths, output_filename, sync_function)
    with pytest.raises(scared.ResynchroError):
        synchronizer.check(nb_traces=1, catch_exceptions=False)
    captured = capsys.readouterr()
    assert captured.out == ""
    ths.close()


def test_synchronizer_run_raises_exception_with_already_existing_result_file_and_overwrite_to_false(sample_directory, output_filename):
    def sync_function(trace_object):
        return trace_object.samples.array

    input_filename = f"{sample_directory}/synchronization/ets_file.ets"
    ths = estraces.read_ths_from_ets_file(input_filename)[:10]

    synchronizer = scared.Synchronizer(ths, output_filename, sync_function)
    synchronizer.run()
    with pytest.raises(FileExistsError, match=f'File "{output_filename}" already exists'):
        synchronizer = scared.Synchronizer(ths, output_filename, sync_function, overwrite=False)
    ths.close()


def test_synchronizer_run_does_not_raise_exception_with_already_existing_result_file_and_overwrite_to_true(sample_directory, output_filename):
    def sync_function(trace_object):
        return trace_object.samples.array

    input_filename = f"{sample_directory}/synchronization/ets_file.ets"
    ths = estraces.read_ths_from_ets_file(input_filename)[:10]

    synchronizer = scared.Synchronizer(ths, output_filename, sync_function)
    synchronizer.run()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        synchronizer = scared.Synchronizer(ths, output_filename, sync_function, overwrite=True)
    synchronizer.run()
    ths.close()


def test_synchronizer_run_raises_exception_with_run_called_twice(sample_directory, output_filename):
    def sync_function(trace_object):
        return trace_object.samples.array

    input_filename = f"{sample_directory}/synchronization/ets_file.ets"
    ths = estraces.read_ths_from_ets_file(input_filename)[:10]

    synchronizer = scared.Synchronizer(ths, output_filename, sync_function)
    synchronizer.run()
    with pytest.raises(scared.SynchronizerError):
        synchronizer.run()
    ths.close()


def test_synchronizer_raises_exception_with_input_ths_of_wrong_type(output_filename):
    def sync_function(trace_object):
        return trace_object.samples.array

    with pytest.raises(TypeError):
        scared.Synchronizer("foo", output_filename, sync_function)


def test_synchronizer_accept_path_as_output(sample_directory, output_filename):
    def sync_function(trace_object):
        return trace_object.samples.array

    input_filename = f"{sample_directory}/synchronization/ets_file.ets"
    ths = estraces.read_ths_from_ets_file(input_filename)[:10]
    scared.Synchronizer(ths, Path(output_filename), sync_function)


def test_synchronizer_raises_exception_with_output_of_wrong_type(sample_directory):
    def sync_function(trace_object):
        return trace_object.samples.array

    input_filename = f"{sample_directory}/synchronization/ets_file.ets"
    ths = estraces.read_ths_from_ets_file(input_filename)[:10]

    with pytest.raises(TypeError, match='output must be an instance of TraceHeaderSet, str or Path, not'):
        scared.Synchronizer(ths, 3, sync_function)


def test_synchronizer_raises_exception_with_function_of_wrong_type(sample_directory, output_filename):

    input_filename = f"{sample_directory}/synchronization/ets_file.ets"
    ths = estraces.read_ths_from_ets_file(input_filename)[:10]

    with pytest.raises(TypeError, match='function attribute should be callable, but'):
        scared.Synchronizer(ths, output_filename, "foo")


def test_synchronizer_copy_headers(sample_directory, output_filename):
    def sync_function(trace_object):
        return trace_object.samples.array

    input_filename = f"{sample_directory}/synchronization/ets_file.ets"
    ths = estraces.read_ths_from_ets_file(input_filename)[:10]
    synchronizer = scared.Synchronizer(ths, output_filename, sync_function)
    out_ths = synchronizer.run()
    assert ths.headers == out_ths.headers
    ths.close()


def test_synchronizer_run_with_ets_file_report_prints_correct_string(sample_directory, capsys, output_filename):
    def sync_function(trace_object):
        return trace_object.samples.array + 0.5

    input_filename = f"{sample_directory}/synchronization/ets_file.ets"
    ths = estraces.read_ths_from_ets_file(input_filename)[:10]

    synchronizer = scared.Synchronizer(ths, output_filename, sync_function)
    synchronizer.run()
    synchronizer.report()
    captured = capsys.readouterr()
    assert captured.out == str("Processed traces....: 10\n"
                               "Synchronized traces.: 10\n"
                               "Success rate........: 100.0%\n")
    ths.close()


def test_synchronizer_run_with_binary_file__str__prints_correct_string(sample_directory, capsys, output_filename):
    def sync_function(trace_object):
        return trace_object.samples.array + 0.5

    input_text_filename = f"{sample_directory}/synchronization/binary_text_file.txt"
    text_128 = estraces.formats.bin_extractor.FilePatternExtractor(input_text_filename, r"([a-fA-F0-9]{128})", num=0)
    text_64 = estraces.formats.bin_extractor.FilePatternExtractor(input_text_filename, r"([a-fA-F0-9]{64})", num=2)
    input_binary_filenames = f"{sample_directory}/synchronization/aes_binary.*"
    ths = estraces.read_ths_from_bin_filenames_pattern(input_binary_filenames, dtype='uint8', metadatas_parsers={"text1": text_128, "text2": text_64})

    synchronizer = scared.Synchronizer(ths, output_filename, sync_function)
    synchronizer.run()
    print(synchronizer)
    captured = capsys.readouterr()
    assert captured.out == str("Processed traces....: 100\n"
                               "Synchronized traces.: 100\n"
                               "Success rate........: 100.0%\n")
    ths.close()
