from scared import traces
import numpy as _np
import math as _math

_ORIGINAL_BATCH_SIZES = [(0, 25000), (1001, 5000), (5001, 2500), (10001, 1000), (50001, 250), (100001, 100)]


def set_batch_size(batch_size=None):
    """Set the batch size used by scared analyses.

    Args:
        batch_size (int, float or list): Batch size to use. If None, the default batch size is restored.
            The type of the input defines its meaning. If the input is:
                - an integer, the batch size is set to the given number of traces,
                - a float, the number of traces is computed using the given value as a batch size in MB (see Notes below),
                - a list, the input is used as a list of sizes (see Example below).

    Notes:
        The trace length used to compute the final batch size is the maximum of the following: the raw trace length, the trace length after preprocessing.

        If a size in MB is given, the batch size is floored to the most significant digit (2542 -> 2000).
        If a size in MB is given, the minimum possible batch size is hardcoded to 10 traces.

    Examples:
        The sizes list [(0, 25000), (1001, 5000), (5001, 2500)] means that:
            - if the trace length is in [0, 1000], the batch size is set to 25_000,
            - if the trace length is in [1001, 5000], the batch size is set to 5000,
            - if the trace length is in [5001, +∞[, the batch size is set to 2500.

    """
    if batch_size is None:
        Container._BATCH_SIZE = _ORIGINAL_BATCH_SIZES
        return
    if isinstance(batch_size, (list, tuple)):
        for bs in batch_size:
            if not isinstance(bs, (list, tuple)) or len(bs) != 2:
                raise ValueError(f'Items of a batch sizes list must be couples, but {bs} found.')
            for val in bs:
                if not isinstance(val, int):
                    raise ValueError(f'Values in batch sizes list must be integers, but {val} found.')
        Container._BATCH_SIZE = list(batch_size) if isinstance(batch_size, tuple) else batch_size.copy()
        return
    if isinstance(batch_size, (int, float)):
        if batch_size <= 0:
            raise ValueError(f'batch_size must be strictly positive, but {batch_size} found.')
        Container._BATCH_SIZE = batch_size
        return
    raise TypeError(f'batch_size must be an integer, a float or a list, but {type(batch_size)} found.')


def _floor_to_most_significant_digit(x):
    if x < 1:
        return 0
    digits = int(_math.log10(x))
    multiplier = int(10**digits)
    return int(_math.floor(x / multiplier)) * multiplier


class Container:
    """Provides a wrapper object around a trace header set and a frame targeted by an analysis.

    A container must be provided to :class:`Analysis` run method. This wrapper provides helpers
    for batch processing.

    Attributes:
        frame (:class:`scared.traces.Samples.SUPPORTED_INDICES_TYPES`, default=None): optional frame
            to be used by the analysis.
        preprocesses (callable or list of callable, default=[]): list of callable preprocess function
            which will be applied on samples when access to batches is invoked. Each preprocess should
            be decorated with :func:`scared.preprocess`, as it add some basic dimensions and shape verifications.

    """

    _BATCH_SIZE = _ORIGINAL_BATCH_SIZES

    def __init__(self, ths, frame=None, preprocesses=[]):
        """Initialize the container instance.

        Args:
            ths (:class:`estraces.TraceHeaderSet`): a :class:`TraceHeaderSet` instance.
            frame (:class:`estraces.Samples.SUPPORTED_INDICES_TYPES`, default=None): optional frame
            to be used by the analysis.
            preprocesses (callable or list of callable, default=[]): list of callable preprocess function
                which will be applied on samples when access to batches is invoked. Each preprocess should
                be decorated with :func:`scared.preprocess`, as it add some basic dimensions and shape verifications.

        """
        self._set_ths(ths)
        self._set_frame(frame)
        self._set_preprocesses(preprocesses)
        self._trace_size = None

    @property
    def trace_size(self):
        """Effective trace size after all preprocessed applied."""
        if self._trace_size is None:
            try:
                wrapper = _TracesBatchWrapper(self._ths[0:1], self.frame, self.preprocesses)
                self._trace_size = len(wrapper.samples[0])
            except TypeError:
                self._trace_size = 1
        return self._trace_size

    def _set_preprocesses(self, preprocesses):
        if (not isinstance(preprocesses, list) or len([p for p in preprocesses if not callable(p)]) > 0) and not callable(preprocesses):
            raise TypeError(f'preprocesses should be a list of preprocess or a single preprocess function, not {type(preprocesses)}.')
        self.preprocesses = [preprocesses] if not isinstance(preprocesses, list) else preprocesses

    def _set_ths(self, ths):
        if not isinstance(ths, traces.TraceHeaderSet):
            raise TypeError(f'ths must be an instance of TraceHeaderSet, not {type(ths)}')
        self._ths = ths

    def _set_frame(self, frame):
        if frame is not None and not isinstance(frame, traces.Samples.SUPPORTED_INDICES_TYPES):
            raise TypeError(f'frame should be of type {traces.Samples.SUPPORTED_INDICES_TYPES}, not {type(frame)}.')
        self.frame = frame if frame is not None else ...

    def _compute_batch_size(self, trace_size):

        try:
            tmp = self._ths[0].samples[self.frame]
            input_size = len(tmp)
            input_dtype = tmp.dtype
            del tmp
        except AttributeError:
            input_size = 0
            input_dtype = _np.dtype('float32')
        max_size = max(trace_size, input_size)
        class_batch_size = Container._BATCH_SIZE  # to avoid race conditions in case of threading
        if isinstance(class_batch_size, int):
            return class_batch_size
        if isinstance(class_batch_size, list):
            for i in range(len(class_batch_size)):
                try:
                    if max_size >= class_batch_size[i][0] and max_size < class_batch_size[i + 1][0]:
                        return class_batch_size[i][1]
                except IndexError:
                    return class_batch_size[-1][1]
        if isinstance(class_batch_size, float):
            batch_size_in_bytes = int(class_batch_size * 2**20)
            batch_size = batch_size_in_bytes / (max_size * input_dtype.itemsize)
            batch_size = _floor_to_most_significant_digit(batch_size)
            return max(batch_size, 10)

    @property
    def batch_size(self):
        """Default size of sub-ths provided by `batches` method."""
        return self._compute_batch_size(self.trace_size)

    def batches(self, batch_size=None):
        """Provides an iterable of wrapper class around :class:`TraceHeaderSet` of size `batch_size`.

        The wrapper provides samples and metadatas properties. Container and frame preprocesses are applied
        when access to this properties are made.

        If `batch_size` is not provided, it is computed based on a simple mapping.

        Args:
            batch_size (int, default=None): size of sub ths provided.

        """
        if batch_size and batch_size < 0:
            raise ValueError(f'batch_size must be a positive integer, not {batch_size}.')
        batch_size = batch_size if batch_size else self.batch_size
        return _TracesBatchIterable(
            ths=self._ths,
            batch_size=batch_size,
            frame=self.frame,
            preprocesses=self.preprocesses
        )

    @property
    def _frame_str(self):
        if isinstance(self.frame, _np.ndarray):
            return f'{str(self.frame)[:20]} ... {str(self.frame)[-20:]}'.replace('\n', '')
        elif self.frame == ...:
            return 'All'
        else:
            return str(self.frame)

    def __str__(self):
        template_str = f'''Traces container:
    Number of traces: {len(self._ths)}
    Traces size     : {len(self._ths.samples[0])}
    Metadata        : {list(self._ths.metadatas.keys())}
    Frame           : {self._frame_str}
    Preprocesses    : {[p.__name__ for p in self.preprocesses] if len(self.preprocesses) > 0 else 'None'}
        '''
        return template_str


class _TracesBatchWrapper:

    def __init__(self, ths, frame, preprocesses):
        self.ths = ths
        self.frame = frame
        self.preprocesses = preprocesses

    @property
    def samples(self):
        samples = self.ths.samples[:, self.frame]
        for preprocess in self.preprocesses:
            samples = preprocess(samples)
        return samples

    @property
    def metadatas(self):
        return self.ths.metadatas

    def __len__(self):
        return len(self.ths)


class _TracesBatchIterable:
    def __init__(self, ths, batch_size, frame, preprocesses):
        self._ths = ths
        self._slices = [
            slice(start * batch_size, (start + 1) * batch_size, 1)
            for start in range(len(ths) // batch_size)
        ]
        if len(ths) % batch_size != 0:
            self._slices.append(
                slice(len(ths) // batch_size * batch_size, None, 1)
            )
        self.frame = frame
        self.preprocesses = preprocesses

    def __iter__(self):
        for sl in self._slices:
            yield _TracesBatchWrapper(self._ths[sl], frame=self.frame, preprocesses=self.preprocesses)

    def __getitem__(self, key):
        return _TracesBatchWrapper(self._ths[self._slices[key]], frame=self.frame, preprocesses=self.preprocesses)

    def __len__(self):
        return len(self._slices)
