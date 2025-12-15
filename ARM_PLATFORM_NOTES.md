# ARM Platform Support Notes

## SCALib on ARM

SCALib is not currently available (or not easily buildable) on ARM architectures. To handle this:

### Automatic Test Skipping

Tests requiring SCALib are automatically skipped on ARM platforms through pytest hooks.

**Detection Logic** (in `tests/conftest.py`):
- Detects ARM using: `platform.machine() in ('aarch64', 'arm64', 'armv7l', 'armv8')`
- Automatically skips all tests in files containing 'scalib' in the filename
- Skip reason: "SCALib not available on ARM architecture"

**Affected Test Files**:
- `tests/test_scalib_snr.py` (54 tests)
- `tests/test_scalib_import.py` (5 tests)
- `tests/test_scalib_performance.py` (12 tests)
- `tests/end_to_end/test_e2e_snr_scalib.py` (1 test)

**Total**: 72 tests skipped on ARM

### CI Configuration

**Non-ARM platforms** (`test:python313-intel`, `test:python313`, `test:python312`, `test:python311`):
- Use `.test:python-with-scalib` base
- Install dependencies with: `pip install .[test-scalib]`
- Run all tests including SCALib

**ARM platform** (`test:python313-arm`):
- Uses `.test:python` base (without scalib)
- Install dependencies with: `pip install .[test]`
- SCALib tests automatically skipped via pytest hooks
- All other tests run normally

### Dependency Configuration

**`pyproject.toml`**:
```toml
[project.optional-dependencies]
test = [
    "pycryptodome",
    "pytest",
    "pytest-cov",
    "pytest-mock",
]
test-scalib = [
    "pycryptodome",
    "pytest",
    "pytest-cov",
    "pytest-mock",
    "scalib",  # Added only for non-ARM platforms
]
```

**Conda build recipe** (`.recipe/prod/meta.yaml`):
```yaml
test:
  requires:
    - pytest
    - pytest-cov
    - pytest-mock
    - pycryptodome
    - scalib  # [not (aarch64 or arm64)]
```

The conda selector `# [not (aarch64 or arm64)]` ensures scalib is only installed on x86_64 builds.

### For Developers

**On ARM platforms**:
```bash
pip install .[test]  # Install without SCALib
pytest tests/        # SCALib tests will be automatically skipped
```

**On x86_64/Intel platforms**:
```bash
pip install .[test-scalib]  # Install with SCALib
pytest tests/                # All tests will run
```

### Testing the Skip Logic

Run `tests/test_arm_skip.py` to verify the platform detection:
```bash
pytest tests/test_arm_skip.py -v -s
```

This will print the current platform and whether SCALib tests would be skipped.

## Graceful Degradation

Even when SCALib is not available:
- The `scared.scalib` module can still be imported
- `SCALIB_AVAILABLE` flag is set to `False`
- `SNRDistinguisherSCALib` class can be instantiated
- Clear error message when trying to use it: "SCALib is not installed. Please install it with: pip install scalib"

This allows users on ARM to use the rest of scared without issues.
