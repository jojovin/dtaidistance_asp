#!/usr/bin/env python3
"""
Compare C vs Python implementation for asymmetric penalties.
"""
import numpy as np
from dtaidistance import dtw

s1 = np.array([0., 1, 2, 3, 4, 5, 6, 7, 8])
s2 = np.array([0., 2, 4, 6, 8])

print("Testing asymmetric penalty: penalty_s1=2.0, penalty_s2=0.5")
print()

d_python = dtw.distance(s1, s2, penalty_s1=2.0, penalty_s2=0.5, use_c=False)
print(f"Python implementation: {d_python:.6f}")

d_c = dtw.distance(s1, s2, penalty_s1=2.0, penalty_s2=0.5, use_c=True)
print(f"C implementation:      {d_c:.6f}")

print(f"\nDifference: {abs(d_c - d_python):.6f}")
print(f"Are they equal? {d_c == d_python}")

print("\n" + "="*50)
print("Testing asymmetric penalty: penalty_s1=0.5, penalty_s2=2.0")
print()

d_python2 = dtw.distance(s1, s2, penalty_s1=0.5, penalty_s2=2.0, use_c=False)
print(f"Python implementation: {d_python2:.6f}")

d_c2 = dtw.distance(s1, s2, penalty_s1=0.5, penalty_s2=2.0, use_c=True)
print(f"C implementation:      {d_c2:.6f}")

print(f"\nDifference: {abs(d_c2 - d_python2):.6f}")
print(f"Are they equal? {d_c2 == d_python2}")
