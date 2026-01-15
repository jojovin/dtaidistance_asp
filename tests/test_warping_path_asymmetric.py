"""
Test warping paths with asymmetric penalties.

This tests that warping_paths and best_path work correctly with
asymmetric penalties (penalty_s1 and penalty_s2).
Tests by: Johan Nygaard Vinther
"""
import math
import pytest
from dtaidistance import dtw, util_numpy

numpyonly = pytest.mark.skipif("util_numpy.test_without_numpy()")


@numpyonly
def test_warping_path_with_asymmetric_penalty():
    """Test that warping paths work correctly with asymmetric penalties."""
    with util_numpy.test_uses_numpy() as np:
        # Create two series with different lengths
        s1 = np.array([0., 1, 2, 3, 4])
        s2 = np.array([0., 1, 2, 3, 4, 5, 6, 7, 8])
        
        # Test with symmetric penalty (baseline)
        path_symmetric, d_symmetric = dtw.warping_path(s1, s2, penalty=1.0, include_distance=True)
        print(f"\nSymmetric penalty (1.0): distance={d_symmetric:.4f}, path length={len(path_symmetric)}")
        
        # Test with asymmetric penalty favoring expansion of s2
        # Low penalty_s2 = cheap to expand s2 (horizontal moves)
        # High penalty_s1 = expensive to expand s1 (vertical moves)
        path_favor_s2, d_favor_s2 = dtw.warping_path(
            s1, s2, penalty_s1=2.0, penalty_s2=0.5, include_distance=True
        )
        print(f"Asymmetric (s1=2.0, s2=0.5): distance={d_favor_s2:.4f}, path length={len(path_favor_s2)}")
        
        # Test with asymmetric penalty favoring expansion of s1
        # High penalty_s2 = expensive to expand s2 (horizontal moves)
        # Low penalty_s1 = cheap to expand s1 (vertical moves)
        path_favor_s1, d_favor_s1 = dtw.warping_path(
            s1, s2, penalty_s1=0.5, penalty_s2=2.0, include_distance=True
        )
        print(f"Asymmetric (s1=0.5, s2=2.0): distance={d_favor_s1:.4f}, path length={len(path_favor_s1)}")
        
        # All distances should be positive and finite
        assert d_symmetric > 0 and not math.isinf(d_symmetric)
        assert d_favor_s2 > 0 and not math.isinf(d_favor_s2)
        assert d_favor_s1 > 0 and not math.isinf(d_favor_s1)
        
        # All paths should be valid
        assert len(path_symmetric) > 0
        assert len(path_favor_s2) > 0
        assert len(path_favor_s1) > 0
        
        # The distances should differ when using different asymmetric penalties
        # The paths may be the same if there's only one optimal alignment,
        # but the accumulated costs should differ
        assert d_favor_s2 != d_symmetric or d_favor_s1 != d_symmetric, \
            "Asymmetric penalties should produce different distances than symmetric"
        
        # When favoring s2 expansion (low penalty_s2), distance should be lower than
        # when penalizing s2 expansion (high penalty_s2) for sequences where s2 is longer
        assert d_favor_s2 < d_favor_s1, \
            f"Lower penalty_s2 should give lower distance when s2 is longer: {d_favor_s2} vs {d_favor_s1}"


@numpyonly  
def test_warping_paths_with_asymmetric_penalty():
    """Test that warping_paths matrix computation works with asymmetric penalties."""
    with util_numpy.test_uses_numpy() as np:
        # Create two series
        s1 = np.array([0., 1, 2, 3, 4])
        s2 = np.array([0., 1, 2, 3, 4, 5, 6])
        
        # Test Python implementation
        d_python, paths_python = dtw.warping_paths(
            s1, s2, penalty_s1=2.0, penalty_s2=0.5, use_c=False
        )
        
        # Test C implementation  
        d_c, paths_c = dtw.warping_paths(
            s1, s2, penalty_s1=2.0, penalty_s2=0.5, use_c=True
        )
        
        print(f"\nPython distance: {d_python:.6f}")
        print(f"C distance:      {d_c:.6f}")
        print(f"Difference:      {abs(d_c - d_python):.6f}")
        
        # Distances should match between Python and C
        assert abs(d_c - d_python) < 1e-6, \
            f"Python and C implementations should produce same distance: {d_python} vs {d_c}"
        
        # Both should be positive and finite
        assert d_python > 0 and not math.isinf(d_python)
        assert d_c > 0 and not math.isinf(d_c)


@numpyonly
def test_warping_paths_fast_with_asymmetric_penalty():
    """Test that warping_paths_fast works with asymmetric penalties."""
    with util_numpy.test_uses_numpy() as np:
        s1 = np.array([0., 1, 2, 3, 4])
        s2 = np.array([0., 1, 2, 3, 4, 5, 6])
        
        # Test with warping_paths_fast (C implementation)
        d_fast, paths_fast = dtw.warping_paths_fast(
            s1, s2, penalty_s1=2.0, penalty_s2=0.5
        )
        
        # Compare with regular warping_paths using C
        d_regular, paths_regular = dtw.warping_paths(
            s1, s2, penalty_s1=2.0, penalty_s2=0.5, use_c=True
        )
        
        print(f"\nFast distance:    {d_fast:.6f}")
        print(f"Regular distance: {d_regular:.6f}")
        
        # Should produce same result
        assert abs(d_fast - d_regular) < 1e-6, \
            f"warping_paths_fast should match warping_paths: {d_fast} vs {d_regular}"


@numpyonly
def test_warping_path_backward_compatibility():
    """Test that single penalty parameter still works for backward compatibility."""
    with util_numpy.test_uses_numpy() as np:
        s1 = np.array([0., 1, 2, 3, 4])
        s2 = np.array([0., 1, 2, 3, 4, 5, 6])
        
        # Old way with single penalty
        path_old, d_old = dtw.warping_path(s1, s2, penalty=1.0, include_distance=True)
        
        # Should still work and produce valid results
        assert d_old > 0 and not math.isinf(d_old)
        assert len(path_old) > 0
        
        # Manually setting both penalties to same value should give same result
        path_manual, d_manual = dtw.warping_path(
            s1, s2, penalty_s1=1.0, penalty_s2=1.0, include_distance=True
        )
        
        # Should produce same distance (paths might differ slightly due to tie-breaking)
        assert abs(d_old - d_manual) < 1e-6, \
            f"Backward compatibility broken: {d_old} vs {d_manual}"


if __name__ == "__main__":
    test_warping_path_with_asymmetric_penalty()
    test_warping_paths_with_asymmetric_penalty()
    test_warping_paths_fast_with_asymmetric_penalty()
    test_warping_path_backward_compatibility()
    print("\nAll tests passed!")
