from scared import traces
import numpy as _np


class Container:
    """Provides a wrapper object around a trace header set and a frame targeted by an analysis.

    A container must be provided to :class:`Analysis` run method. This wrapper provides helpers
    for batch processing.

    Attributes:
        frame (:class:`scared.traces.Samples.SUPPORTED_INDICES_TYPES`, default=None): optional frame
            to be used by the analysis.
        preprocesses (callable or list of callable, default=[]): list of callable preprocess function
            which will be applied on samples when access to batches is invoked. Each preprocess should
            be decored with :func:`scared.preprocess`, as it add some basic dimensions and shape verifications.

    """

    def __init__(self, ths, frame=None, preprocesses=[]):
        """Initialize the container instance.

        Args:
            ths (:class:`estraces.TraceHeaderSet`): a :class:`TraceHeaderSet` instance.
            frame (:class:`estraces.Samples.SUPPORTED_INDICES_TYPES`, default=None): optional frame
            to be used by the analysis.
            preprocesses (callable or list of callable, default=[]): list of callable preprocess function
                which will be applied on samples when access to batches is invoked. Each preprocess should
                be decored with :func:`scared.preprocess`, as it add some basic dimensions and shape verifications.

        """
        self._set_ths(ths)
        self._set_frame(frame)
        self._set_preprocesses(preprocesses)
        self._trace_size = None

    @property
    def trace_size(self):
        """Effective trace size that will be analysed for this container instance."""
        if self._trace_size is None:
            try:
                self._trace_size = len(self._ths[0].samples[self.frame])
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
        ref_sizes = [
            (0, 25000),
            (1001, 5000),
            (5001, 2500),
            (10001, 1000),
            (50001, 250),
            (100001, 100)
        ]
        for i in range(len(ref_sizes)):
            try:
                if trace_size >= ref_sizes[i][0] and trace_size < ref_sizes[i + 1][0]:
                    return ref_sizes[i][1]
            except IndexError:
                return ref_sizes[-1][1]

    @property
    def batch_size(self):
        """Default size of sub-ths provided by `batches` method."""
        return self._compute_batch_size(self.trace_size)

    def batches(self, batch_size=None):
        """Provides an iterable of wrapper class around :class:`TraceHeaderSet` of size `batch_size`.

        The wrapper provides samples and metadatas properties. Container and frame preprocesses are applied
        when acces to this properties are made.

        If `batch_size` is not provided, it is computed based on a simple mapping.

        Args:
            batch_size (int, default=None): size of sub ths provided.
        """
        if batch_size and batch_size < 0:
            raise ValueError(f'batch_size must be a positive integer, not {batch_size}.')
        batch_size = batch_size if batch_size else self._compute_batch_size(self.trace_size)
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
    Traces size     : {self._ths.samples.shape[1]}
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
