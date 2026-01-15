"""
Test asymmetric penalty feature for DTW.

This tests that penalty_s1 and penalty_s2 work independently and produce
different results when set to different values.
"""
import math
import pytest
from dtaidistance import dtw, util_numpy

numpyonly = pytest.mark.skipif("util_numpy.test_without_numpy()")


@numpyonly
def test_asymmetric_penalty():
    """Test that asymmetric penalties work correctly."""
    with util_numpy.test_uses_numpy() as np:
        np.set_printoptions(precision=2, linewidth=120)
        # Create two series of different lengths to ensure asymmetric behavior
        # s1 is longer, so the algorithm must either expand s2 or compress s1
        s1 = np.array([0., 1, 2, 3, 4, 5, 6, 7, 8])
        s2 = np.array([0., 2, 4, 6, 8])
        
        # Test symmetric penalty (baseline)
        d_symmetric = dtw.distance(s1, s2, penalty=1.0)
        
        # Test asymmetric penalty favoring expansion of s2 (cheap s2, expensive s1)
        # Low penalty_s2 = cheap to expand s2 (horizontal moves)
        # High penalty_s1 = expensive to expand s1 (vertical moves)
        d_favor_s2_expand = dtw.distance(s1, s2, penalty_s1=2.0, penalty_s2=0.5)
        
        # Test asymmetric penalty favoring expansion of s1 (cheap s1, expensive s2)
        # High penalty_s2 = expensive to expand s2 (horizontal moves)
        # Low penalty_s1 = cheap to expand s1 (vertical moves)
        d_favor_s1_expand = dtw.distance(s1, s2, penalty_s1=0.5, penalty_s2=2.0)
        
        # All distances should be positive and finite
        assert d_symmetric > 0 and not math.isinf(d_symmetric)
        assert d_favor_s2_expand > 0 and not math.isinf(d_favor_s2_expand)
        assert d_favor_s1_expand > 0 and not math.isinf(d_favor_s1_expand)
        
        # The asymmetric penalties should produce different results
        assert d_favor_s2_expand != d_favor_s1_expand, "Asymmetric penalties should produce different distances"
        
        # Print results for debugging
        print(f"Symmetric penalty (1.0): {d_symmetric}")
        print(f"Favor s2 expansion (s1=2.0, s2=0.5): {d_favor_s2_expand}")
        print(f"Favor s1 expansion (s1=0.5, s2=2.0): {d_favor_s1_expand}")


@numpyonly
def test_asymmetric_penalty_backwards_compatibility():
    """Test that penalty parameter still works (backwards compatibility)."""
    with util_numpy.test_uses_numpy() as np:
        s1 = np.array([0., 1, 2, 1, 0])
        s2 = np.array([2., 1, 0, 1, 2])
        
        # Old way: using penalty parameter
        d_old = dtw.distance(s1, s2, penalty=1.0)
        
        # New way: using penalty_s1 and penalty_s2 with same values
        d_new = dtw.distance(s1, s2, penalty_s1=1.0, penalty_s2=1.0)
        
        # Should produce the same result
        assert d_old == pytest.approx(d_new), "Backward compatibility broken: penalty should equal symmetric penalty_s1/s2"


@numpyonly
def test_asymmetric_penalty_zero():
    """Test that zero penalty works correctly."""
    with util_numpy.test_uses_numpy() as np:
        s1 = np.array([0., 1, 2, 3, 4])
        s2 = np.array([0., 1, 2, 3, 4])
        
        # No penalty
        d_no_penalty = dtw.distance(s1, s2)
        
        # Zero asymmetric penalties
        d_zero_asymmetric = dtw.distance(s1, s2, penalty_s1=0, penalty_s2=0)
        
        # Should be the same
        assert d_no_penalty == pytest.approx(d_zero_asymmetric)


@numpyonly
def test_asymmetric_penalty_only_s1():
    """Test setting only penalty_s1."""
    with util_numpy.test_uses_numpy() as np:
        s1 = np.array([0., 1, 2, 1, 0, 1, 2])
        s2 = np.array([2., 1, 0, 1, 2, 1, 0])
        
        # Only penalize expansion in s1
        d_s1_only = dtw.distance(s1, s2, penalty_s1=1.0, penalty_s2=0)
        
        # Should be different from no penalty
        d_no_penalty = dtw.distance(s1, s2)
        
        assert d_s1_only != d_no_penalty


@numpyonly
def test_asymmetric_penalty_only_s2():
    """Test setting only penalty_s2."""
    with util_numpy.test_uses_numpy() as np:
        s1 = np.array([0., 1, 2, 1, 0, 1, 2])
        s2 = np.array([2., 1, 0, 1, 2, 1, 0])
        
        # Only penalize expansion in s2
        d_s2_only = dtw.distance(s1, s2, penalty_s1=0, penalty_s2=1.0)
        
        # Should be different from no penalty
        d_no_penalty = dtw.distance(s1, s2)
        
        assert d_s2_only != d_no_penalty


if __name__ == "__main__":
    test_asymmetric_penalty()
    test_asymmetric_penalty_backwards_compatibility()
    test_asymmetric_penalty_zero()
    test_asymmetric_penalty_only_s1()
    test_asymmetric_penalty_only_s2()
    print("All tests passed!")
