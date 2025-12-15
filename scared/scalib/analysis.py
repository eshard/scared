"""SCALib-based analysis classes for scared."""

from ..analysis.base import BasePartitionedAttack as _BasePartitionedAttack, BasePartitionedReverse as _BasePartitionedReverse
from .snr import SNRDistinguisherSCALibMixin as _SNRDistinguisherSCALibMixin


class SNRAttackSCALib(_BasePartitionedAttack, _SNRDistinguisherSCALibMixin):
    """SNR attack using SCALib's high-performance implementation.

    This class provides the same interface as scared's SNRAttack but uses
    SCALib's optimized SNR metric internally for better performance.

    Examples:
        >>> from scared.scalib import SNRAttackSCALib
        >>> from scared import aes, Container, HammingWeight, maxabs
        >>>
        >>> # Create attack
        >>> attack = SNRAttackSCALib(
        ...     selection_function=aes.selection_functions.encrypt.FirstSubBytes(),
        ...     model=HammingWeight(),
        ...     discriminant=maxabs
        ... )
        >>>
        >>> # Run on container
        >>> attack.run(container)
        >>> key = attack.scores.argmax(axis=0)

    See Also:
        - :class:`scared.analysis.SNRAttack`: Standard scared SNR attack
        - :class:`SNRDistinguisherSCALib`: Underlying SCALib-based distinguisher

    """

    pass


class SNRReverseSCALib(_BasePartitionedReverse, _SNRDistinguisherSCALibMixin):
    """SNR reverse analysis using SCALib's high-performance implementation.

    This class provides the same interface as scared's SNRReverse but uses
    SCALib's optimized SNR metric internally for better performance.

    Examples:
        >>> from scared.scalib import SNRReverseSCALib
        >>> from scared import Container
        >>>
        >>> # Create reverse analysis
        >>> reverse = SNRReverseSCALib(
        ...     partitions=range(256),
        ...     precision='float32'
        ... )
        >>>
        >>> # Run on container
        >>> reverse.run(container)
        >>> snr_values = reverse.results

    See Also:
        - :class:`scared.analysis.SNRReverse`: Standard scared SNR reverse
        - :class:`SNRDistinguisherSCALib`: Underlying SCALib-based distinguisher

    """

    pass
