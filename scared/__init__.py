# -*- coding: utf-8 -*-
from .__version__ import __version__ as VERSION  # noqa: F401, N812
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
from .ttest import TTestAccumulator, TTestAnalysis, TTestError, TTestContainer  # noqa:F401
from .analysis import (  # noqa:F401
    BaseAttack, CPAAttack, DPAAttack,
    ANOVAAttack, NICVAttack, SNRAttack,
    BasePartitionedAttack, MIAAttack,
    _Attack,
    BaseReverse, CPAReverse, DPAReverse,
    ANOVAReverse, NICVReverse, SNRReverse,
    BasePartitionedReverse, MIAReverse,
    _Reverse, TemplateAttack, TemplateDPAAttack
)
from .preprocesses import preprocess, Preprocess, PreprocessError  # noqa:F401
from . import preprocesses  # noqa: F401
from . import aes  # noqa: F401
from . import des  # noqa: F401
from . import container as _container

Container = _container.Container
# Set default logging handler to avoid "No handler found" warnings.
logging.getLogger(__name__).addHandler(logging.NullHandler())
# Always display DeprecationWarning by default.
warnings.simplefilter('default', category=DeprecationWarning)
