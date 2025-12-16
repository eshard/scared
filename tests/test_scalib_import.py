"""Tests for SCALib module import handling."""

import pytest
import numpy as np


def test_scalib_module_importable():
    """Test that the scalib module can be imported."""
    from scared import scalib
    assert scalib is not None


def test_scalib_available_flag_exists():
    """Test that SCALIB_AVAILABLE flag exists."""
    from scared.scalib import SCALIB_AVAILABLE
    assert isinstance(SCALIB_AVAILABLE, bool)


def test_snr_distinguisher_scalib_class_exists():
    """Test that SNRDistinguisherSCALib class is defined."""
    from scared.scalib import SNRDistinguisherSCALib
    assert SNRDistinguisherSCALib is not None


def test_snr_distinguisher_scalib_instantiation():
    """Test that SNRDistinguisherSCALib can be instantiated."""
    from scared.scalib import SNRDistinguisherSCALib, SCALIB_AVAILABLE

    # Should be able to create instance
    snr = SNRDistinguisherSCALib(partitions=range(16))
    assert snr is not None
    assert snr.partitions is not None

    if not SCALIB_AVAILABLE:
        # If SCALib not available, should get error on update
        traces = np.random.randn(10, 20).astype(np.float32)
        labels = np.random.randint(0, 16, (10, 1), dtype=np.uint8)

        with pytest.raises(ImportError, match="SCALib is not installed"):
            snr.update(traces, labels)


def test_graceful_import_error():
    """Test that ImportError is handled gracefully."""
    from scared.scalib import SCALIB_AVAILABLE

    if not SCALIB_AVAILABLE:
        # Should have gracefully handled the missing import
        from scared.scalib.snr import SNRDistinguisherSCALib
        snr = SNRDistinguisherSCALib(partitions=range(16))

        # Should provide helpful error message
        with pytest.raises(ImportError) as exc_info:
            traces = np.random.randn(10, 20).astype(np.float32)
            labels = np.random.randint(0, 16, (10, 1), dtype=np.uint8)
            snr.update(traces, labels)

        assert "pip install scalib" in str(exc_info.value).lower()
