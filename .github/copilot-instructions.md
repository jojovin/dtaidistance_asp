# DTAIDistance Development Guide

## Project Overview
This is a time series distance library (DTW, Edit Distance) with **dual Python/C implementations** for performance. The project implements Dynamic Time Warping with a recent enhancement for **asymmetric penalties** (`penalty_s1`, `penalty_s2`).

## Architecture

### Three-Layer Stack
1. **C Core** (`src/DTAIDistanceC/DTAIDistanceC/dd_dtw.{c,h}`) - Pure C implementation with OpenMP support
2. **Cython Bindings** (`src/dtaidistance/dtw_cc.pyx`, `dtaidistancec_dtw.pxd`) - Bridge C to Python
3. **Python API** (`src/dtaidistance/dtw.py`) - User-facing interface with `DTWSettings` class

### Settings Pattern
All algorithm parameters flow through `DTWSettings`:
- C struct: `src/DTAIDistanceC/DTAIDistanceC/dd_dtw.h` defines `struct DTWSettings_s`
- Cython: `dtw_cc.pyx` exposes `DTWSettings` class with C struct wrapper
- Python: `dtw.py` provides Pythonic API via `DTWSettings(**kwargs)` 

**Critical**: When adding parameters, update all three layers in this order: C → Cython → Python

### Asymmetric Penalty Implementation
The recent asymmetric penalty feature demonstrates the pattern:
- `penalty` (symmetric, backward compatible) vs `penalty_s1`/`penalty_s2` (asymmetric)
- C code at line ~106 in `dd_dtw.c`: `seq_t penalty_vertical = settings->penalty_s1;`
- Tests in `tests/test_asymmetric_penalty.py` validate both C and Python implementations

## Build & Test Workflow

### Building
```bash
# Rebuild C extensions (required after C/Cython changes)
python3 setup.py build_ext --inplace

# Or use convenience script
./rebuild_and_install.sh
```

### Testing
```bash
# Run all tests (includes C library tests)
make test

# Skip C library tests (Python-only)
make test-nolibs

# Run specific test
pytest tests/test_asymmetric_penalty.py -v
```

**Key**: Always test both `use_c=True` and `use_c=False` for feature parity (see `compare_c_python.py` pattern)

### OpenMP Considerations
- Setup.py detects compiler and sets OpenMP flags automatically
- On macOS: Uses `-Xpreprocessor -fopenmp` for Clang compatibility
- Custom flags: `--noopenmp`, `--forcellvm`, `--forcegnugcc` (see `setup.py:68-73`)

## Code Conventions

### Import Guards
```python
# Numpy is optional - always guard usage
try:
    import numpy as np
except ImportError:
    np = None
```

### C/Python Consistency
When implementing algorithms:
1. Python reference implementation in `dtw.py` (pure Python)
2. C optimized version in `dd_dtw.c`
3. Test equivalence: `assert dtw.distance(s1, s2, use_c=False) == dtw.distance(s1, s2, use_c=True)`

### Testing Pattern
```python
@numpyonly  # Skip if numpy unavailable
def test_feature():
    with util_numpy.test_uses_numpy() as np:
        # Test code using numpy
```

## Key Files

- [dtw.py](src/dtaidistance/dtw.py) - Main Python API, `distance()`, `distance_fast()`, `DTWSettings`
- [dd_dtw.c](src/DTAIDistanceC/DTAIDistanceC/dd_dtw.c) - C implementation of DTW algorithms
- [dtw_cc.pyx](src/dtaidistance/dtw_cc.pyx) - Cython bridge layer
- [setup.py](setup.py) - Complex build system with OpenMP detection
- [ASYMMETRIC_PENALTY_README.md](ASYMMETRIC_PENALTY_README.md) - Example of adding new features

## Common Tasks

### Adding New DTW Parameters
1. Add field to `DTWSettings_s` struct in `dd_dtw.h`
2. Initialize in `dtw_settings_default()` in `dd_dtw.c`
3. Add property to `DTWSettings` class in `dtw_cc.pyx`
4. Update `DTWSettings.__init__()` in `dtw.py`
5. Write tests comparing C and Python implementations

### Debugging C Code
- Use `printf()` in C code (outputs to stderr during build)
- Compile without optimization: modify `c_args` in `setup.py`
- Check Cython HTML annotation: `cython -a dtw_cc.pyx` generates `dtw_cc.html`

### Performance Testing
```bash
make benchmark  # Uses pytest-benchmark
```
