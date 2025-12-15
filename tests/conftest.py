import platform
import pytest
from .context import scared  # noqa


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--force-scalib",
        action="store_true",
        default=False,
        help="Force running SCALib tests even if SCALib is not available (tests will fail if not installed)"
    )


def pytest_collection_modifyitems(config, items):
    """Skip SCALib tests on ARM architecture or when SCALib is not available."""
    # Check if user wants to force SCALib tests
    force_scalib = config.getoption("--force-scalib")

    # If forcing, don't skip any tests
    if force_scalib:
        return

    is_arm = platform.machine() in ('aarch64', 'arm64', 'armv7l', 'armv8')

    # Check if SCALib is actually available
    try:
        from scared.scalib import SCALIB_AVAILABLE
        scalib_available = SCALIB_AVAILABLE
    except ImportError:
        scalib_available = False

    # Skip SCALib tests if on ARM or if SCALib not available
    if is_arm or not scalib_available:
        reason = "SCALib not available on ARM architecture" if is_arm else "SCALib not installed"
        skip_scalib = pytest.mark.skip(reason=reason)
        for item in items:
            # Skip tests in scalib-related test files
            if 'scalib' in str(item.fspath).lower():
                item.add_marker(skip_scalib)
