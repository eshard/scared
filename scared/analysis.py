from . import selection_functions as _sf, container as _container, models, distinguishers
import numpy as _np


class BaseAnalysis:
    """Base class for all analysis processing objects.

    This class must be subclassed and combined with a mixin inheriting from :class:`DistiguinsherMixin`.
    It provides the common processing method for a side-channel statistical analysis.

    The base class:
        - initialize the state before processing any traces
        - provides method, either to process traces and compute results manually, either to run a complete processing of a :class:`Container` instance
        - manage results attributes: dstinguisher method output (`results`), `scores` and `convergence_traces`

    Attributes:
        results (:class:`numpy.ndarray`): array containing the latest results obtained from the distinguisher computing.
        scores (:class:`numpy.ndarray`): array containing the latest scores obtained by processing `results` with `discriminant`.
        convergence_step (int, default=None): number of traces to process before each computation of results.
            If convergence_step is provided, all intermediate scores computed are kept in `convergence_traces`
        convergences_traces (:class:`numpy.ndarray`): array containing the `scores` values at each convergence step defined by `convergence_step`.

    Examples:
        First, you have to use either one the distinguisher mixin available, all create one which inherit from :class:`DistinguisherMixin`:

            class MyDistinguisherMixin(DistinguisherMixin):
            # implements the needed method.

        Create a new class by inheriting from :class:BaseAnalysis class and the distinguisher mixin:

            class MyAnalysis(BaseAnalysis, MyDistinguisherMixin):
                pass

        Create your analysis object and run it on a container:

            analysis = MyAnalysis(...)
            analysis.run(container)

    """

    def __init__(self, selection_function, model, discriminant, precision='float32', convergence_step=None):
        """Initialize analysis.

        Args:
            selection_function (:class:`SelectionFunction`): selection function to compute intermediate values. Must inherit from :class:`SelectionFunction`.
            model (:class:`Model`): model instance to compute leakage intermediate values. Must inherit from :class:`Model`.
            discriminant (function): a function to compute scores from a distinguisher results array.
                Must be decorated with :func:`scared.discriminants.disciminant`.
            precision (:class:`numpy.dtype`, default=`float32`): precision which will be used for computations.
            convergence_step (int, default=None): if provided, `run` method will compute and stores `scores` each time `convergence_step` traces are processed.

        """
        if not isinstance(self, distinguishers.DistinguisherMixin):
            raise NotImplementedError(f'BaseAnalysis class should be used in conjunction with a mixin class inheriting from DistinguisherMixin.')

        distinguishers._initialize_distinguisher(self, precision=precision, processed_traces=0)

        self._set_selection_function(selection_function)
        self._set_model(model)
        self._set_discriminant(discriminant)
        self._set_convergence(convergence_step)

        self.scores = None
        self.results = None

    def _set_convergence(self, convergence_step):
        if convergence_step is not None:
            if not isinstance(convergence_step, int):
                raise TypeError(f'convergence_step must be an integer, not {type(convergence_step)}.')
            if convergence_step <= 0:
                raise ValueError(f'convergence_step must be a strictly positive integer, not {type(convergence_step)}.')

        self.convergence_step = convergence_step
        self.convergence_traces = None
        self._batches_processed = [0]

    def _set_selection_function(self, selection_function):
        if not isinstance(selection_function, _sf.SelectionFunction):
            raise TypeError(f'selection function should be a SelectionFunction, not {type(selection_function)}.')
        self.selection_function = selection_function

    def _set_model(self, model):
        if not isinstance(model, models.Model):
            raise TypeError(f'model should be a Model instance, not {type(model)}.')
        self.model = model

    def _set_discriminant(self, discriminant):
        if not callable(discriminant):
            raise TypeError(f'discriminant should be a callable, not {type(discriminant)}.')
        self.discriminant = discriminant

    def _compute_batch_size(self, base_batch_size):
        if self.convergence_step:
            if base_batch_size >= self.convergence_step:
                return self.convergence_step
            else:
                return int(self.convergence_step / (self.convergence_step // base_batch_size))
        return base_batch_size

    def _final_compute(self):
        self.compute_results()
        if self.convergence_step and len(self._batches_processed) > 1:
            self._compute_convergence_traces()

    def run(self, container):
        """Process traces wrapped by `container` and compute results for this analysis.

        Starting from the current state of this instance, container is processed by batch.
        Batch size is determined from convergence step and container default batch size.

        Results are computed for each convergence step (if required), and after having processed all the container.

        Args:
            container (:class:`Container`): a :class:`TraceHeaderSet` wrapped by a :class:`Container` instance.

        """
        if not isinstance(container, _container.Container):
            raise TypeError(f'container should be a type Container, not {type(container)}.')

        batch_size = self._compute_batch_size(container.batch_size)
        for batch in container.batches(batch_size=batch_size):
            self.process(batch, frame=container.frame)
            self.compute_convergence()
        self._final_compute()

    def compute_intermediate_values(self, metadata):
        """Compute intermediate leakage values for this instance from metadata.

        Args:
            metadata (mapping): a dict-like object containing the data to be used with selection function.

        """
        return self.model(self.selection_function(**metadata))

    def process(self, traces_batch, frame=...):
        """Process and update the current state with traces batch.

        Intermediate leakage values are computed, and state is updated.
        This method is used internally by `run`, but can also be used to have a finer control on the process.

        Args:
            traces_batch (:class:`estraces.Trace`): collection of `estraces.Trace` instance?
            frame (default=...): frame of traces samples to use.

        """
        intermediate_values = self.compute_intermediate_values(traces_batch.metadatas)

        if self.convergence_traces is None and self.convergence_step:
            self.convergence_traces = _np.empty(intermediate_values.shape[1:] + (0, ), dtype=self.precision)

        self.update(
            data=intermediate_values,
            traces=traces_batch.samples[:, frame]
        )

    def compute_convergence(self):
        """Compute convergence results, if required, for the current state.

        This method is used internally by `run`, but can also be used to have a finer control on the process.

        """
        if self.convergence_step:
            self._batches_processed.append(self.processed_traces)
            if self._batches_processed[-1] - self._batches_processed[0] >= self.convergence_step:
                self._batches_processed = [self._batches_processed[-1]]
                self.compute_results()
                self._compute_convergence_traces()

    def _compute_convergence_traces(self):
        self.convergence_traces = _np.append(self.convergence_traces, self.scores[..., None], axis=-1)

    def compute_results(self):
        """Compute results for the current state.

        This method is used internally by `run`, but can also be used to have a finer control on the process.

        """
        self.results = self.compute()
        self.scores = self.discriminant(self.results)


class CPAAnalysis(BaseAnalysis, distinguishers.CPADistinguisherMixin):
    __doc__ = distinguishers.CPADistinguisherMixin.__doc__ + BaseAnalysis.__doc__


class DPAAnalysis(BaseAnalysis, distinguishers.DPADistinguisherMixin):
    __doc__ = distinguishers.DPADistinguisherMixin.__doc__ + BaseAnalysis.__doc__

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if not isinstance(self.model, models.Monobit):
            raise distinguishers.DistinguisherError(f'DPA analysis can be processed only with Monobit model, not {type(self.model)}.')
