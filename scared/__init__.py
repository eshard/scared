# -*- coding: utf-8 -*-
from .__version__ import __version__ as VERSION  # noqa: F401, N812
import warnings
import logging

import estraces as traces  # noqa: F401

from .selection_functions import selection_function, attack_selection_function, reverse_selection_function  # noqa: F401
from .models import HammingWeight, Value, Monobit, Model  # noqa: F401
from .discriminants import discriminant, nanmax, maxabs, opposite_min, nansum, abssum  # noqa: F401
from .distinguishers import (  # noqa: F401
    Distinguisher, DistinguisherError, DPADistinguisher,
    CPADistinguisher, DistinguisherMixin, DPADistinguisherMixin,
    CPADistinguisherMixin
)
from .analysis import BaseAnalysis, CPAAnalysis, DPAAnalysis  # noqa:F401
from . import container as _container

Container = _container.Container

# Set default logging handler to avoid "No handler found" warnings.
logging.getLogger(__name__).addHandler(logging.NullHandler())
# Always display DeprecationWarning by default.
warnings.simplefilter('default', category=DeprecationWarning)
