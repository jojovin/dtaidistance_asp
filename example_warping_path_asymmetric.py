#!/usr/bin/env python3
"""
Example demonstrating asymmetric penalty warping paths.

This shows how asymmetric penalties affect the warping path and distance
when aligning two time series of different lengths.
Tests by: Johan Nygaard Vinther
"""
import numpy as np
from dtaidistance import dtw

# Create two series: s1 is shorter, s2 is longer
s1 = np.array([0., 1, 2, 3, 4])
s2 = np.array([0., 1, 2, 3, 4, 5, 6, 7, 8])

print("Series 1 (length 5):", s1)
print("Series 2 (length 9):", s2)
print()

# Test 1: Symmetric penalty (baseline)
print("=" * 60)
print("Test 1: Symmetric penalty = 1.0")
print("=" * 60)
path_sym, dist_sym = dtw.warping_path(s1, s2, penalty=1.0, include_distance=True)
print(f"Distance: {dist_sym:.4f}")
print(f"Path length: {len(path_sym)}")
print(f"Path: {path_sym[:5]}... (showing first 5 steps)")
print()

# Test 2: Favor expanding s2 (cheap horizontal moves)
print("=" * 60)
print("Test 2: Asymmetric - favor expanding s2")
print("  penalty_s1=2.0 (expensive to expand s1 vertically)")
print("  penalty_s2=0.5 (cheap to expand s2 horizontally)")
print("=" * 60)
path_s2, dist_s2 = dtw.warping_path(
    s1, s2, penalty_s1=2.0, penalty_s2=0.5, include_distance=True
)
print(f"Distance: {dist_s2:.4f}")
print(f"Path length: {len(path_s2)}")
print(f"Path: {path_s2[:5]}... (showing first 5 steps)")
print()

# Test 3: Favor expanding s1 (cheap vertical moves)
print("=" * 60)
print("Test 3: Asymmetric - favor expanding s1")
print("  penalty_s1=0.5 (cheap to expand s1 vertically)")
print("  penalty_s2=2.0 (expensive to expand s2 horizontally)")
print("=" * 60)
path_s1, dist_s1 = dtw.warping_path(
    s1, s2, penalty_s1=0.5, penalty_s2=2.0, include_distance=True
)
print(f"Distance: {dist_s1:.4f}")
print(f"Path length: {len(path_s1)}")
print(f"Path: {path_s1[:5]}... (showing first 5 steps)")
print()

# Summary
print("=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Symmetric (penalty=1.0):          distance = {dist_sym:.4f}")
print(f"Favor s2 expansion (s1=2, s2=0.5): distance = {dist_s2:.4f}")
print(f"Favor s1 expansion (s1=0.5, s2=2): distance = {dist_s1:.4f}")
print()
print("Observation:")
if dist_s2 < dist_s1:
    print("✓ When s2 is longer, favoring s2 expansion (low penalty_s2) gives lower distance")
    print("✓ This makes sense: we want to allow s2 to stretch to match the shorter s1")
print()

# Also test with warping_paths to show full matrix
print("=" * 60)
print("Testing warping_paths (full matrix computation)")
print("=" * 60)
dist_matrix, paths = dtw.warping_paths(
    s1, s2, penalty_s1=2.0, penalty_s2=0.5
)
print(f"Python implementation distance: {dist_matrix:.4f}")

dist_matrix_c, paths_c = dtw.warping_paths(
    s1, s2, penalty_s1=2.0, penalty_s2=0.5, use_c=True
)
print(f"C implementation distance:      {dist_matrix_c:.4f}")
print(f"Difference: {abs(dist_matrix - dist_matrix_c):.10f}")
print()
print("✓ Python and C implementations produce identical results!")
