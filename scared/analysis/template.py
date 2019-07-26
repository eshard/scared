from .base import BasePartitionedAttack, BasePartitionedReverse
from scared import distinguishers, selection_functions as _sf, container as _container
import logging

logger = logging.getLogger(__name__)


class _TemplateBuildAnalysis(BasePartitionedReverse, distinguishers._TemplateBuildDistinguisherMixin):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(self.selection_function.words, int):
            raise _sf.base.SelectionFunctionError(
                f'Selection function for template attack must return only 1 word of intermediate data.\
                `selection_function.words` attribute must thus be an integer, not {self.selection_function.words}.'
            )


class BaseTemplateAttack(BasePartitionedAttack):

    def __init__(self, container_building, selection_function, reverse_selection_function, model, convergence_step=None, partitions=None, precision='float32'):
        """Initialize template attack.

        Args:
            reverse_selection_function (:class:`SelectionFunction`): selection function to compute intermediate values for build phase.
            selection_function (:class:`SelectionFunction`): attack selection function to compute intermediate values for matching phase.
            model (:class:`Model`): model instance to compute leakage intermediate values. Must inherit from :class:`Model`.
            precision (:class:`numpy.dtype`, default=`float32`): precision which will be used for computations.
            convergence_step (int, default=None): if provided, `run` method will compute and stores `scores` each time `convergence_step` traces are processed.
            partitions (range, default=None): partitions to use for the build phase. If unset, partitions will be determinated by the model values.

        """
        super().__init__(
            selection_function=selection_function,
            model=model,
            precision=precision,
            discriminant=lambda scores: scores,
            convergence_step=convergence_step
        )
        self._init_template(partitions=partitions, reverse_selection_function=reverse_selection_function, container_building=container_building)

    def build(self):
        """Build templates for this attack."""
        logger.debug(f'Build analysis object: {self._build_analysis}.')
        self._build_analysis.run(self.container_building)
        self.partitions = self._build_analysis.partitions
        self.templates = self._build_analysis.results
        self.pooled_covariance_inv = self._build_analysis.pooled_covariance_inv
        self.pooled_covariance = self._build_analysis.pooled_covariance
        self.is_build = True

    def _init_template(self, partitions, reverse_selection_function, container_building):
        distinguishers.partitioned._set_partitions(self, partitions)

        if not isinstance(container_building, _container.Container):
            raise TypeError(f'TemplateAttack must be instantiated with a `Container` instance for building phase.')
        self.container_building = container_building

        self._build_analysis = _TemplateBuildAnalysis(
            selection_function=reverse_selection_function,
            model=self.model,
            partitions=self.partitions,
            precision=self.precision
        )

        self.is_build = False


class TemplateAttack(BaseTemplateAttack, distinguishers.TemplateAttackDistinguisherMixin):
    """Provides a high-level class to proceed to static template attacks.

    This class is first for the profiling phase. Once build, matching phase can be run on trace header sets container.

    Attributes:
        templates (:class:`numpy.ndarray`): templates array, available once analysis build phase is run.
        scores (:class:`numpy.ndarray`): scores resulting from the latest matching phase run on a trace header set.
        convergence_step (int, default=None): number of traces to process before each computation of results.
            If convergence_step is provided, all intermediate scores computed are kept in `convergence_traces`
        convergences_traces (:class:`numpy.ndarray`): array containing the `scores` values at each convergence step defined by `convergence_step`.

    Examples:
        Instantiate by passing a container with building trace header set, a reverse selection function and a model.

            template = scared.TemplateAttack(
                container_building=container,
                reverse_selection_function=sf,
                model=scared.Value()
            )

        Use `build` to build templates:

            template.build()

        Use `run` with a trace header set container to process matching:

            template.run(container)

    """

    def __init__(
        self, container_building, reverse_selection_function, model, convergence_step=None, partitions=None, precision='float32', selection_function=None
    ):

        @_sf.base.attack_selection_function
        def identity(**kwargs):
            return kwargs[kwargs.keys()[0]]

        super().__init__(
            selection_function=identity,
            model=model,
            precision=precision,
            convergence_step=convergence_step,
            container_building=container_building,
            reverse_selection_function=reverse_selection_function,
            partitions=partitions
        )

        self.selection_function = lambda **kwargs: kwargs[list(kwargs.keys())[0]]


class TemplateDPAAttack(BaseTemplateAttack, distinguishers.TemplateDPADistinguisherMixin):
    """Provides a high-level class to proceed to DPA template attacks.

    This class is first for the profiling phase. Once build, matching phase can be run on trace header sets container.

    Attributes:
        templates (:class:`numpy.ndarray`): templates array, available once analysis build phase is run.
        scores (:class:`numpy.ndarray`): scores resulting from the latest matching phase run on a trace header set.
        convergence_step (int, default=None): number of traces to process before each computation of results.
            If convergence_step is provided, all intermediate scores computed are kept in `convergence_traces`
        convergences_traces (:class:`numpy.ndarray`): array containing the `scores` values at each convergence step defined by `convergence_step`.

    Examples:
        Instantiate by passing a container with building trace header set, a reverse selection function, an attack selection function and a model

            template = scared.TemplateAttack(
                container_building=container,
                selection_function=asf,
                reverse_selection_function=sf,
                model=scared.Value(),
            )

        Use `build` to build templates:

            template.build()

        Use `run` with a trace header set container to process matching:

            template.run(container)

    """

    pass
