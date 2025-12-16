"""SCALib integration module for scared.

This module provides wrappers around SCALib's high-performance implementations
of side-channel analysis metrics, maintaining compatibility with scared's API.

SCALib (https://github.com/simple-crypto/SCALib) is an optional dependency.
"""

from .snr import SNRDistinguisherSCALib, SNRDistinguisherSCALibMixin, SCALIB_AVAILABLE
from .analysis import SNRAttackSCALib, SNRReverseSCALib
from .ttest import TTestAnalysisSCALib

__all__ = [
    'SNRDistinguisherSCALib',
    'SNRDistinguisherSCALibMixin',
    'SNRAttackSCALib',
    'SNRReverseSCALib',
    'TTestAnalysisSCALib',
    'SCALIB_AVAILABLE',
]
