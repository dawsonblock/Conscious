#!/usr/bin/env python3
"""
Quick script to verify Python environment is properly configured.
"""
import sys

print("Python version:", sys.version)
print("Python executable:", sys.executable)

try:
    import numpy as np
    print(f"✓ NumPy {np.__version__} found")
except ImportError as e:
    print(f"✗ NumPy not found: {e}")
    sys.exit(1)

try:
    import scipy
    print(f"✓ SciPy {scipy.__version__} found")
except ImportError as e:
    print(f"✗ SciPy not found: {e}")
    sys.exit(1)

try:
    import matplotlib
    print(f"✓ Matplotlib {matplotlib.__version__} found")
except ImportError as e:
    print(f"✗ Matplotlib not found: {e}")
    sys.exit(1)

print("\n✓ All dependencies are properly installed!")
