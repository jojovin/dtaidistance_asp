#!/bin/bash
# Script to rebuild and reinstall the package with the asymmetric penalty changes

echo "Rebuilding dtaidistance package..."

# Clean old build artifacts
echo "Cleaning old build artifacts..."
rm -rf build/
rm -rf dist/
rm -rf *.egg-info

# Rebuild the package
echo "Building package..."
python setup.py build_ext --inplace

# Install the package
echo "Installing package..."
python3 setup.py install

echo "You can test with: python example_asymmetric_penalty.py"
