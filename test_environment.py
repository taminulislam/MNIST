#!/usr/bin/env python3
"""
CS535 HW4 - Environment Test Script
This script verifies that all required packages are installed correctly.
"""

import sys

def test_imports():
    """Test all required package imports."""
    print("=" * 70)
    print("CS535 HW4 - Environment Verification")
    print("=" * 70)
    print()

    packages = {
        'tensorflow': 'TensorFlow',
        'numpy': 'NumPy',
        'matplotlib': 'Matplotlib',
        'sklearn': 'Scikit-learn',
        'pandas': 'Pandas',
        'seaborn': 'Seaborn',
        'jupyter': 'Jupyter'
    }

    results = []
    all_success = True

    for package, name in packages.items():
        try:
            if package == 'sklearn':
                import sklearn
                version = sklearn.__version__
            elif package == 'jupyter':
                import jupyter
                version = "✓"
            else:
                module = __import__(package)
                version = module.__version__

            print(f"✓ {name:15s} - Version {version}")
            results.append((name, True, version))
        except ImportError as e:
            print(f"✗ {name:15s} - NOT FOUND")
            results.append((name, False, str(e)))
            all_success = False
        except Exception as e:
            print(f"⚠ {name:15s} - ERROR: {str(e)}")
            results.append((name, False, str(e)))
            all_success = False

    print()
    print("=" * 70)

    if all_success:
        print("SUCCESS: All packages installed correctly!")
        print()
        print("Additional Information:")
        print("-" * 70)

        # Test TensorFlow specifics
        try:
            import tensorflow as tf
            print(f"Python Version: {sys.version.split()[0]}")
            print(f"TensorFlow Version: {tf.__version__}")

            # Check for GPU
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                print(f"GPU Available: Yes ({len(gpus)} GPU(s) detected)")
                for i, gpu in enumerate(gpus):
                    print(f"  GPU {i}: {gpu.name}")
            else:
                print("GPU Available: No (CPU only - this is fine for this assignment)")

            # Test Fashion MNIST loading
            print()
            print("Testing Fashion MNIST dataset loading...")
            fashion_mnist = tf.keras.datasets.fashion_mnist
            (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
            print(f"✓ Dataset loaded successfully!")
            print(f"  Training samples: {len(train_images):,}")
            print(f"  Test samples: {len(test_images):,}")
            print(f"  Image shape: {train_images[0].shape}")

        except Exception as e:
            print(f"⚠ Warning: Could not complete additional tests - {str(e)}")

        print()
        print("=" * 70)
        print("You're ready to start the assignment!")
        print("Run: jupyter notebook")
        print("=" * 70)
        return 0
    else:
        print("ERROR: Some packages are missing or not installed correctly.")
        print()
        print("To fix this, run:")
        print("  pip install -r requirements.txt")
        print()
        print("=" * 70)
        return 1

if __name__ == "__main__":
    sys.exit(test_imports())
