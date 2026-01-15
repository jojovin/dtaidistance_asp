# Asymmetric Penalty Feature for DTW

## Overview

This implementation adds support for asymmetric penalties in Dynamic Time Warping (DTW). Previously, the DTW implementation supported a single `penalty` parameter that applied the same cost when aligning one point to multiple points in either direction. With this enhancement, you can now specify different penalties for expansion/compression in each series independently.

## What is a Penalty in DTW?

In DTW, when aligning two time series, you can move:
- **Diagonally**: Match one point from series 1 to one point from series 2 (no penalty)
- **Horizontally**: Match one point from series 1 to multiple points from series 2 (compress s1, expand s2)
- **Vertically**: Match multiple points from series 1 to one point from series 2 (expand s1, compress s2)

The penalty parameter adds an extra cost when taking horizontal or vertical steps, discouraging excessive compression or expansion.

## Asymmetric Penalties

With asymmetric penalties, you can now control these costs independently:

- **`penalty_s1`**: Penalty for expanding series 1 (vertical moves in the DTW matrix)
  - Higher values make it more expensive to align multiple points from s1 to one point in s2
  
- **`penalty_s2`**: Penalty for expanding series 2 (horizontal moves in the DTW matrix)
  - Higher values make it more expensive to align one point from s1 to multiple points in s2

## Usage

### Python API

```python
from dtaidistance import dtw
import numpy as np

s1 = np.array([0., 1, 2, 3, 4])
s2 = np.array([0., 1, 2, 3, 4])

# Old way (symmetric penalty, still supported for backward compatibility)
distance = dtw.distance(s1, s2, penalty=1.0)

# New way (asymmetric penalties)
distance = dtw.distance(s1, s2, penalty_s1=2.0, penalty_s2=0.5)

# Using DTWSettings
settings = dtw.DTWSettings(penalty_s1=2.0, penalty_s2=0.5)
distance = dtw.distance(s1, s2, **settings.c_kwargs())
```

### C API

The C implementation has been updated with two new fields in the `DTWSettings` struct:

```c
struct DTWSettings_s {
    // ... other fields ...
    seq_t penalty;       // Original symmetric penalty (for backward compatibility)
    seq_t penalty_s1;    // Penalty for expanding series 1
    seq_t penalty_s2;    // Penalty for expanding series 2
    // ... other fields ...
};
```

The implementation maintains backward compatibility: if `penalty_s1` and `penalty_s2` are both 0 but `penalty` is set, the code will use the symmetric `penalty` value for both directions.

## Implementation Details

### Modified Files

1. **C Implementation**:
   - `src/DTAIDistanceC/DTAIDistanceC/dd_dtw.h`: Added `penalty_s1` and `penalty_s2` fields to `DTWSettings` and `DTWWps` structs
   - `src/DTAIDistanceC/DTAIDistanceC/dd_dtw.c`: Updated all DTW distance functions to use asymmetric penalties

2. **Cython Bindings**:
   - `src/dtaidistance/dtaidistancec_dtw.pxd`: Updated struct declarations
   - `src/dtaidistance/dtw_cc.pyx`: Added parameter handling and properties for new penalty fields

3. **Python Interface**:
   - `src/dtaidistance/dtw.py`: Added `penalty_s1` and `penalty_s2` parameters to `DTWSettings` class

4. **Tests**:
   - `tests/test_asymmetric_penalty.py`: Comprehensive test suite for the new feature

### Backward Compatibility

The implementation is fully backward compatible. Existing code using the `penalty` parameter will continue to work exactly as before. The new asymmetric penalties are only activated when explicitly set via `penalty_s1` and/or `penalty_s2`.

## Use Cases

Asymmetric penalties are useful when:

1. **Different temporal scales**: One series represents a process that naturally varies faster than the other
2. **Query-template matching**: You want to allow the template to stretch but keep the query fixed
3. **Signal processing**: One signal is a reference that should match tightly, while the other can have more variation
4. **Biological sequences**: Different costs for insertions vs deletions

## Example

See `example_asymmetric_penalty.py` for a complete working example.

## Building

After making these changes, you'll need to rebuild the C extensions:

```bash
python setup.py build_ext --inplace
```

Or install in development mode:

```bash
pip install -e .
```

## Testing

Run the asymmetric penalty tests:

```bash
pytest tests/test_asymmetric_penalty.py -v
```

Run all DTW tests to ensure backward compatibility:

```bash
pytest tests/test_penalty.py tests/test_dtw.py -v
```
