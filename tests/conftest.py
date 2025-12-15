import platform
import pytest
from .context import scared  # noqa


def pytest_collection_modifyitems(config, items):
    """Skip SCALib tests on ARM architecture where SCALib may not be available."""
    is_arm = platform.machine() in ('aarch64', 'arm64', 'armv7l', 'armv8')

    if is_arm:
        skip_scalib = pytest.mark.skip(reason="SCALib not available on ARM architecture")
        for item in items:
            # Skip tests in scalib-related test files
            if 'scalib' in str(item.fspath).lower():
                item.add_marker(skip_scalib)
