import warnings
import logging

import estraces as traces  # noqa: F401

from .selection_functions.base import selection_function, attack_selection_function, reverse_selection_function, SelectionFunctionError  # noqa: F401
from .models import HammingWeight, Value, Monobit, Model  # noqa: F401
from .discriminants import discriminant, nanmax, maxabs, opposite_min, nansum, abssum  # noqa: F401
from .distinguishers import (  # noqa: F401
    DistinguisherError, Distinguisher, DistinguisherMixin,
    DPADistinguisherMixin, DPADistinguisher,
    CPADistinguisherMixin, CPAAlternativeDistinguisherMixin, CPADistinguisher, CPAAlternativeDistinguisher,
    PartitionedDistinguisherMixin, PartitionedDistinguisher, ANOVADistinguisherMixin, ANOVADistinguisher,
    NICVDistinguisherMixin, NICVDistinguisher, SNRDistinguisherMixin, SNRDistinguisher,
    MIADistinguisher, TemplateAttackDistinguisherMixin
)
from .ttest import TTestThreadAccumulator, TTestAnalysis, TTestError, TTestContainer  # noqa:F401
from .analysis import (  # noqa:F401
    BaseAttack, CPAAttack, DPAAttack,
    ANOVAAttack, NICVAttack, SNRAttack,
    BasePartitionedAttack, MIAAttack,
    BaseReverse, CPAReverse, DPAReverse,
    ANOVAReverse, NICVReverse, SNRReverse,
    BasePartitionedReverse, MIAReverse,
    TemplateAttack, TemplateDPAAttack
)
from .preprocesses import preprocess, Preprocess, PreprocessError  # noqa:F401
from .synchronization import Synchronizer, ResynchroError, SynchronizerError  # noqa:F401
from . import preprocesses  # noqa: F401
from . import signal_processing  # noqa: F401
from . import aes  # noqa: F401
from . import des  # noqa: F401
from . import container as _container
from .container import set_batch_size  # noqa: F401
from . import utils as _utils  # noqa: F401  # for backwards compatibility


Container = _container.Container
# Set default logging handler to avoid "No handler found" warnings.
logging.getLogger(__name__).addHandler(logging.NullHandler())
# Always display DeprecationWarning by default.
warnings.simplefilter('default', category=DeprecationWarning)

# Get version from setuptools-scm
try:
    from ._version import __version__
except ImportError:
    # Fallback for development installations without setuptools-scm
    try:
        from importlib.metadata import version, PackageNotFoundError
        try:
            __version__ = version("scared")
        except PackageNotFoundError:
            __version__ = "unknown"
    except ImportError:
        __version__ = "unknown"

VERSION = __version__
