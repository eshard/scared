from scared import traces


class Container:
    """Provides a wrapper object around a trace header set and a frame targeted by an analysis.

    A container must be provided to :class:`Analysis` run method. This wrapper provides helpers
    for batch processing.

    Attributes:
        frame (:class:`scared.traces.Samples.SUPPORTED_INDICES_TYPES`, default=None): optional frame
            to be used by the analysis.

    """

    def __init__(self, ths, frame=None):
        """Initialize the container instance.

        Args:
            ths (:class:`estraces.TraceHeaderSet`): a :class:`TraceHeaderSet` instance.
            frame (:class:`estraces.Samples.SUPPORTED_INDICES_TYPES`, default=None): optional frame
            to be used by the analysis.
        """
        if not isinstance(ths, traces.TraceHeaderSet):
            raise TypeError(f'ths must be an instance of TraceHeaderSet, not {type(ths)}')
        self._ths = ths

        if frame is not None and not isinstance(frame, traces.Samples.SUPPORTED_INDICES_TYPES):
            raise TypeError(f'frame should be of type {traces.Samples.SUPPORTED_INDICES_TYPES}, not {type(frame)}.')
        self.frame = frame if frame is not None else ...
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
        """Provides a :class:`TraceHeaderSet` iterator of sub-ths containing `batch_size`.

        If `batch_size` is not provided, it is computed based on a simple mapping.

        Args:
            batch_size (int, default=None): size of sub ths provided.
        """
        if batch_size and batch_size < 0:
            raise ValueError(f'batch_size must be a positive integer, not {batch_size}.')
        batch_size = batch_size if batch_size else self._compute_batch_size(self.trace_size)
        return self._ths.split(batch_size)
