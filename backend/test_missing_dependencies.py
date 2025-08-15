#!/usr/bin/env python3
"""
Test for missing dependencies that might cause fallbacks
"""

import sys

def check_dependencies():
    """Check for missing dependencies"""
    
    missing_deps = []
    
    # Check scipy
    try:
        import scipy
        import scipy.signal
        import scipy.stats
        print(f"OK scipy {scipy.__version__} available")
    except ImportError as e:
        missing_deps.append(f"scipy: {e}")
        print(f"MISSING scipy: {e}")
    
    # Check numpy
    try:
        import numpy as np
        print(f"OK numpy {np.__version__} available")
    except ImportError as e:
        missing_deps.append(f"numpy: {e}")
        print(f"MISSING numpy: {e}")
    
    # Check sklearn
    try:
        from sklearn.feature_selection import mutual_info_regression
        print(f"OK sklearn.feature_selection available")
    except ImportError as e:
        missing_deps.append(f"sklearn.feature_selection: {e}")
        print(f"MISSING sklearn.feature_selection: {e}")
    
    # Check matplotlib
    try:
        import matplotlib
        print(f"OK matplotlib {matplotlib.__version__} available")
    except ImportError as e:
        missing_deps.append(f"matplotlib: {e}")
        print(f"MISSING matplotlib: {e}")
    
    # Test basic operations that might fail
    try:
        import numpy as np
        import scipy.signal
        import scipy.stats
        
        # Test signal processing
        signal = np.random.randn(1000)
        freqs, psd = scipy.signal.periodogram(signal)
        print("OK scipy.signal.periodogram works")
        
        # Test Hilbert transform
        analytic = scipy.signal.hilbert(signal)
        print("OK scipy.signal.hilbert works")
        
        # Test butter filter
        b, a = scipy.signal.butter(4, [0.1, 0.5], btype='band', fs=1.0)
        print("OK scipy.signal.butter works")
        
        # Test mutual info
        try:
            from sklearn.feature_selection import mutual_info_regression
            X = np.random.randn(100, 1)
            y = np.random.randn(100)
            mi = mutual_info_regression(X, y)
            print("OK mutual_info_regression works")
        except Exception as e:
            print(f"FAILED mutual_info_regression: {e}")
        
    except Exception as e:
        print(f"FAILED Basic operations: {e}")
    
    if missing_deps:
        print(f"\nMissing dependencies: {len(missing_deps)}")
        for dep in missing_deps:
            print(f"  - {dep}")
    else:
        print(f"\nOK All core dependencies available")
    
    return len(missing_deps) == 0

if __name__ == "__main__":
    success = check_dependencies()
    sys.exit(0 if success else 1)