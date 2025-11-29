# Command Reference for CS535 HW4

## Environment Setup Commands

### Create Virtual Environment
```bash
python3 -m venv cs535_hw4_env
```

### Activate Virtual Environment
```bash
# Linux/Mac
source cs535_hw4_env/bin/activate

# Windows
cs535_hw4_env\Scripts\activate
```

### Deactivate Virtual Environment
```bash
deactivate
```

### Install Dependencies
```bash
# Upgrade pip first
pip install --upgrade pip

# Install from requirements file
pip install -r requirements.txt

# Or install individually
pip install tensorflow numpy matplotlib scikit-learn jupyter pandas seaborn
```

## Running Jupyter Notebook

### Launch Jupyter
```bash
# Make sure virtual environment is activated first
jupyter notebook
```

### Launch Jupyter Lab (Alternative)
```bash
jupyter lab
```

### Run Jupyter on Specific Port
```bash
jupyter notebook --port 8889
```

### Run Jupyter with No Browser (for remote servers)
```bash
jupyter notebook --no-browser --port=8888
```

## Working with the Notebook

### Convert Notebook to PDF (via LaTeX)
```bash
jupyter nbconvert --to pdf CS535_HW4_MNIST_Fashion_Classification.ipynb
```

### Convert Notebook to HTML
```bash
jupyter nbconvert --to html CS535_HW4_MNIST_Fashion_Classification.ipynb
```

### Execute Notebook from Command Line
```bash
jupyter nbconvert --to notebook --execute CS535_HW4_MNIST_Fashion_Classification.ipynb
```

## Verification Commands

### Check Python Version
```bash
python --version
# Should be 3.8 or higher
```

### Check TensorFlow Installation
```bash
python -c "import tensorflow as tf; print(tf.__version__)"
```

### Check GPU Availability (Optional)
```bash
python -c "import tensorflow as tf; print('GPU Available:', tf.config.list_physical_devices('GPU'))"
```

### List Installed Packages
```bash
pip list
```

### Check Package Versions
```bash
pip show tensorflow numpy matplotlib scikit-learn
```

## Google Colab Alternative

If you prefer to use Google Colab instead of local setup:

1. Go to [Google Colab](https://colab.research.google.com/)
2. Upload the notebook: File → Upload notebook
3. All required packages are pre-installed
4. Change runtime type for GPU: Runtime → Change runtime type → GPU (optional)
5. Run all cells: Runtime → Run all

## Troubleshooting Commands

### Update All Packages
```bash
pip install --upgrade tensorflow numpy matplotlib scikit-learn jupyter pandas seaborn
```

### Clear Jupyter Cache
```bash
jupyter notebook --clear-output
```

### Reinstall TensorFlow
```bash
pip uninstall tensorflow
pip install tensorflow
```

### Check Jupyter Kernels
```bash
jupyter kernelspec list
```

### Install Jupyter Kernel for Virtual Environment
```bash
python -m ipykernel install --user --name=cs535_hw4_env
```

## Git Commands (Optional - for version control)

### Initialize Repository
```bash
git init
```

### Add Files
```bash
git add CS535_HW4_MNIST_Fashion_Classification.ipynb
git add requirements.txt
git add README.md
```

### Commit Changes
```bash
git commit -m "Complete CS535 HW4 implementation"
```

## Useful Python Commands for Testing

### Test TensorFlow
```python
import tensorflow as tf
print("TensorFlow version:", tf.__version__)
print("GPU available:", tf.config.list_physical_devices('GPU'))
```

### Test All Imports
```python
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd
print("All imports successful!")
```

### Quick Fashion MNIST Test
```python
from tensorflow import keras
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
print(f"Dataset loaded: {train_images.shape}")
```

## Performance Tips

### Limit TensorFlow GPU Memory Growth (if using GPU)
```python
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
```

### Set Number of Threads for CPU
```python
import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(4)
```

## Export Commands

### Export with Outputs
```bash
jupyter nbconvert --to html --no-input CS535_HW4_MNIST_Fashion_Classification.ipynb
```

### Export to Multiple Formats
```bash
jupyter nbconvert --to html,pdf CS535_HW4_MNIST_Fashion_Classification.ipynb
```

## Clean Up

### Remove Virtual Environment
```bash
# Deactivate first
deactivate

# Remove directory
rm -rf cs535_hw4_env
```

### Clear Python Cache
```bash
find . -type d -name "__pycache__" -exec rm -r {} +
find . -type f -name "*.pyc" -delete
```
