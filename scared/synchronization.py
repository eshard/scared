"""Provides higher-level python objects to easily apply synchronization method to ths objects."""

from contextlib import contextmanager as _contextmanager
import copy as _copy
import logging as _logging
import os as _os
import traceback as _traceback
import warnings as _warn
import sys as _sys

import estraces as _estraces
from estraces.formats.ets_writer import ETSWriter as _ETSWriter
import numpy as _np


logger = _logging.getLogger(__name__)


@_contextmanager
def _no_stdout():
    with open(_os.devnull, "w") as devnull:
        old_stdout = _sys.stdout
        _sys.stdout = devnull
        try:
            yield
        finally:
            _sys.stdout = old_stdout


class ResynchroError(Exception):
    """Error to raise in resynchronization function to reject trace."""


class SynchronizerError(Exception):
    """Synchronizer Error Exception."""


class _ErrorCounter:
    """Object used to count errors during resync and raise warning if too much consecutive errors occur.

    Attributes:
        last_errro_id: Last error identifier.
        counter: Consecutive errors counter.

    Usage:
        >>> _ErrorCounter.error_occur(error_id)

    """

    def __init__(self):
        self.limit = 8
        self.last_error_id = 0
        self.counter = 0

    def error_occur(self, error_id):
        # test if error is considered 'consecutive' to the last one
        if error_id == self.last_error_id + 1:
            self.counter += 1  # if yes, increment error counter
        else:
            self.counter = 1
        self.last_error_id = error_id
        # if counter over limit, print Warning and double limit
        if self.counter >= self.limit:
            self.limit *= 2
            _warn.warn(f"Exception raised on {self.counter} consecutive traces during synchronization.", UserWarning)


class Synchronizer:
    r"""Higher-level python objects to apply resynchronization method to ths objects.

    Attributes:
        input_ths (TraceHeaderSet): Input ths that contain campaign to synchronize.
        output (str or TraceHeaderSet): If filename, an ETSWriter object is build with this filename.
        function (function): Function to use for synchronization.
        synchronized_counter: Number of synchronized traces.
        overwrite (bool): If True, reset and overwrite output file (default: False).
        kwargs: Used to pass extra named arguments to the synchronization function.

    Note:
        * The following arguments are passed to the synchronization function:
            - trace_object: trace object from trace set
            - \*\*kwargs directly from Synchronizer
        * stdout is disabled during Synchronizer.run().

    Warning:
        All plots in synchronization function must be disabled before launching Synchronizer.run().

    """

    def __init__(self, input_ths, output, function, overwrite=False, **kwargs):
        self.input_ths = self._check_input_ths(input_ths)
        self.output = self._check_output(output, overwrite)
        self.function = self._check_function(function)
        self.kwargs = kwargs
        self.processed_counter = 0
        self.synchronized_counter = 0
        self._err_counter = None

    def _check_input_ths(self, input_ths):
        if not isinstance(input_ths, _estraces.TraceHeaderSet):
            raise TypeError(f'input_ths must be an instance of TraceHeaderSet, not {type(input_ths)}.')
        return input_ths

    def _check_output(self, output, overwrite):
        if isinstance(output, str):
            return _ETSWriter(filename=output, overwrite=overwrite)
        elif isinstance(output, _estraces.TraceHeaderSet):
            return output
        else:
            raise TypeError(f'output must be an instance of TraceHeaderSet or str, not {type(output)}.')

    def _check_function(self, function):
        if not callable(function):
            raise TypeError(f"function attribute should be callable, but it is of type {type(function)}.")
        return function

    def check(self, nb_traces=5, catch_exceptions=True):
        """Test synchronization function on traces picked randomly in trace set.

        Args:
            nb_traces (int): Number of traces to test on.
            catch_exceptions (bool): If True, exceptions are catched and just printed. You can disable it to view the full traceback.

        Returns:
            (list): Trace arrays.

        """
        result = []
        random_indexes = _np.random.choice(_np.arange(len(self.input_ths)), nb_traces)
        tmp_ths = self.input_ths[random_indexes.tolist()]
        for trace_object in tmp_ths:
            try:
                synchronized = self.function(trace_object=trace_object, **self.kwargs)
                result.append(_copy.deepcopy(synchronized))
                if synchronized is None:
                    raise SynchronizerError("Synchronization function returns None.")
            except KeyboardInterrupt as exception:
                raise exception
            except Exception as exception:
                if catch_exceptions:
                    exc_type, exc_value, exc_traceback = _sys.exc_info()
                    last_traceback = _traceback.extract_tb(exc_traceback)[-1]
                    formated = _traceback.format_exception(exc_type, exc_value, exc_traceback)
                    raised = formated[-1][:-1]
                    print(f"Raised {raised} in {last_traceback.name} line {last_traceback.lineno}.")
                else:
                    raise exception
        return result

    def run(self):
        """Perform synchronization.

        Returns:
            (TraceHeaderSet): Ths object of synchronized campaign.

        Note:
            This function is callable only one time.

        """
        if self._err_counter is not None:
            raise SynchronizerError("'run()' method of Synchronizer object was already called. Rebuild Synchronizer object.")
        self._err_counter = _ErrorCounter()

        with _no_stdout():
            logger.info(f'Perform synchronization started.')
            logger.info(f'Number of iterations for the synchronization: {len(self.input_ths)}', {'nb_iterations': len(self.input_ths)})
            for i, trace_object in enumerate(self.input_ths):
                logger.info(f'Start processing synchronization for trace number {i}.')
                try:
                    synchronized_data = None
                    synchronized_data = self.function(trace_object=trace_object, **self.kwargs)
                    if synchronized_data is not None:
                        self.synchronized_counter += 1
                    else:
                        raise SynchronizerError("Synchronization function returns None.")
                except KeyboardInterrupt as exception:
                    raise exception
                except Exception:
                    self._err_counter.error_occur(self.processed_counter)
                finally:
                    self.processed_counter += 1
                if synchronized_data is not None:
                    self.output.write_trace_object_and_points(trace_object=trace_object, points=synchronized_data, index=self.synchronized_counter - 1)
                logger.info(f'processing synchronization for trace number {i}: iteration finished.')

        self.output.close()
        return self.output.get_reader()

    def report(self):
        """Print some statistics about synchronization."""
        print(self)

    def __str__(self):
        """Print some statistics about synchronization."""
        out_str = f"Processed traces....: {self.processed_counter}\n"
        out_str += f"Synchronized traces.: {self.synchronized_counter}\n"
        out_str += f"Success rate........: {_np.round(self.synchronized_counter / self.processed_counter * 100, decimals=2)}%"
        return out_str
