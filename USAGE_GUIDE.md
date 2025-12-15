# Fashion MNIST Classification - Usage Guide

## Table of Contents
1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Detailed Usage](#detailed-usage)
4. [Module Documentation](#module-documentation)
5. [Examples](#examples)
6. [Tips & Best Practices](#tips--best-practices)

## Installation

### Step 1: Clone or Download Project
```bash
cd /path/to/MNIST
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Using venv
python -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation
```bash
python -c "import tensorflow as tf; print(tf.__version__)"
```

## Quick Start

### Run All Homework Tasks
```bash
python run_experiments.py
```

This executes all 4 tasks and saves results to `outputs/`.

### Train a Model
```bash
# Baseline model
python train.py --model baseline --epochs 10

# With model saving
python train.py --model baseline --epochs 10 --save-model
```

### Evaluate a Model
```bash
python evaluate.py --model-path models/baseline_model.keras
```

## Detailed Usage

### 1. Training Custom Models

#### Baseline Model (Simple)
```bash
python train.py \
    --model baseline \
    --epochs 10 \
    --batch-size 32 \
    --save-model
```

**Output:**
- Training progress logs
- Final accuracy metrics
- Training curve plot (`outputs/baseline_training_history.png`)
- Saved model (`models/baseline_model.keras`)

#### Improved Model (Better Performance)
```bash
python train.py \
    --model improved \
    --epochs 20 \
    --batch-size 64 \
    --save-model \
    --output-dir results_improved
```

**Features:**
- Dropout layers (reduces overfitting)
- Batch normalization (faster convergence)
- More hidden units (higher capacity)

#### CNN Model (Best Performance)
```bash
python train.py \
    --model cnn \
    --epochs 15 \
    --batch-size 32 \
    --save-model
```

**Features:**
- Convolutional layers (spatial feature learning)
- Max pooling (translation invariance)
- Better accuracy (~91-93%)

### 2. Evaluating Models

```bash
python evaluate.py \
    --model-path models/baseline_model.keras \
    --output-dir evaluation_results
```

**Generates:**
- Confusion matrix (`confusion_matrix.png`)
- Sample predictions with confidence (`prediction_samples.png`)
- Per-class accuracy analysis
- Common misclassification patterns

### 3. Running Complete Experiments

```bash
python run_experiments.py
```

**Executes:**
1. **Task 1**: Train baseline model (10 epochs)
2. **Task 2**: Generate confusion matrix
3. **Task 3**: Plot learning curve over epochs
4. **Task 4**: Analyze data volume impact (12 models, 5K-60K samples)

**Estimated Time:** 15-20 minutes

### 4. Using Single-File Script

```bash
python mnist_fashion_classification.py
```

This runs the complete pipeline in one script (useful for demos).

### 5. Using Jupyter Notebook

```bash
jupyter notebook CS535_HW4_MNIST_Fashion_Classification.ipynb
```

Interactive exploration with visualizations.

## Module Documentation

### `src/config.py`
Configuration parameters for the entire project.

```python
from src import config

# Access parameters
print(config.EPOCHS_BASELINE)  # 10
print(config.HIDDEN_UNITS)     # 128
print(config.CLASS_NAMES)      # ['T-shirt/top', ...]
```

### `src/data_loader.py`
Data loading and preprocessing.

```python
from src.data_loader import load_fashion_mnist, create_data_subset

# Load data
train_images, train_labels, test_images, test_labels = load_fashion_mnist()

# Create subset
subset_images, subset_labels = create_data_subset(
    train_images, train_labels, size=10000, seed=42
)
```

### `src/models.py`
Model architectures.

```python
from src.models import create_baseline_model, create_improved_model, create_cnn_model

# Create models
baseline = create_baseline_model()
improved = create_improved_model()
cnn = create_cnn_model()

# Customize parameters
custom_model = create_baseline_model(hidden_units=256, num_classes=10)
```

### `src/visualization.py`
Plotting utilities.

```python
from src.visualization import (
    plot_sample_images, plot_confusion_matrix,
    plot_training_history, plot_data_volume_curve
)

# Plot samples
plot_sample_images(train_images, train_labels, num_samples=25)

# Plot confusion matrix
cm = plot_confusion_matrix(true_labels, predicted_labels)

# Plot training history
plot_training_history(history, save_path='my_plot.png')
```

### `src/analysis.py`
Analysis functions.

```python
from src.analysis import (
    analyze_confusion_matrix, print_training_summary,
    evaluate_model_performance
)

# Analyze confusion matrix
results = analyze_confusion_matrix(cm)

# Print training summary
print_training_summary(history)

# Evaluate model
loss, acc = evaluate_model_performance(model, test_images, test_labels)
```

## Examples

### Example 1: Train Multiple Models and Compare

```python
import tensorflow as tf
from src.data_loader import load_fashion_mnist
from src.models import create_baseline_model, create_improved_model
from src.analysis import evaluate_model_performance

# Load data
train_images, train_labels, test_images, test_labels = load_fashion_mnist()

# Train baseline
baseline = create_baseline_model()
baseline.fit(train_images, train_labels, epochs=10, validation_split=0.1, verbose=1)

# Train improved
improved = create_improved_model()
improved.fit(train_images, train_labels, epochs=10, validation_split=0.1, verbose=1)

# Compare
print("Baseline Performance:")
evaluate_model_performance(baseline, test_images, test_labels)

print("\nImproved Performance:")
evaluate_model_performance(improved, test_images, test_labels)
```

### Example 2: Custom Data Volume Experiment

```python
import numpy as np
from src import config
from src.data_loader import load_fashion_mnist, create_data_subset
from src.models import create_baseline_model
from src.visualization import plot_data_volume_curve

# Load data
train_images, train_labels, test_images, test_labels = load_fashion_mnist()

# Custom data sizes
data_sizes = [1000, 5000, 10000, 20000, 40000]
train_accs = []
test_accs = []

for size in data_sizes:
    # Create subset
    subset_img, subset_lbl = create_data_subset(train_images, train_labels, size)

    # Train model
    model = create_baseline_model()
    history = model.fit(subset_img, subset_lbl, epochs=5,
                       validation_data=(test_images, test_labels), verbose=0)

    train_accs.append(history.history['accuracy'][-1])
    test_accs.append(history.history['val_accuracy'][-1])

# Plot results
plot_data_volume_curve(data_sizes, train_accs, test_accs, epochs=5)
```

### Example 3: Analyze Specific Classes

```python
import numpy as np
from sklearn.metrics import classification_report
from src.data_loader import load_fashion_mnist
from src import config

# Load data and model
_, _, test_images, test_labels = load_fashion_mnist()
model = tf.keras.models.load_model('models/baseline_model.keras')

# Make predictions
predictions = model.predict(test_images)
predicted_labels = np.argmax(predictions, axis=1)

# Detailed report
report = classification_report(
    test_labels, predicted_labels,
    target_names=config.CLASS_NAMES,
    digits=4
)
print(report)
```

### Example 4: Save and Load Models

```python
from src.models import create_baseline_model
from src.data_loader import load_fashion_mnist

# Train and save
train_images, train_labels, test_images, test_labels = load_fashion_mnist()
model = create_baseline_model()
model.fit(train_images, train_labels, epochs=10, verbose=1)
model.save('my_custom_model.keras')

# Load and use
loaded_model = tf.keras.models.load_model('my_custom_model.keras')
predictions = loaded_model.predict(test_images)
```

## Tips & Best Practices

### Performance Tips

1. **Use GPU acceleration** for faster training:
   ```bash
   # Check GPU availability
   python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
   ```

2. **Adjust batch size** based on memory:
   ```bash
   # Larger batch = faster, but more memory
   python train.py --batch-size 128  # If you have enough RAM/VRAM

   # Smaller batch = slower, but less memory
   python train.py --batch-size 16   # If you have limited RAM/VRAM
   ```

3. **Early stopping** for efficiency:
   ```python
   from tensorflow.keras.callbacks import EarlyStopping

   early_stop = EarlyStopping(monitor='val_accuracy', patience=3)
   model.fit(X, y, epochs=50, callbacks=[early_stop])
   ```

### Reproducibility

Always set random seeds:
```python
import numpy as np
import tensorflow as tf

np.random.seed(42)
tf.random.set_seed(42)
```

Or use the CLI option:
```bash
python train.py --seed 42
```

### Debugging

Enable verbose output:
```python
# In model.fit()
model.fit(X, y, verbose=2)  # Progress bar per epoch

# Or
model.fit(X, y, verbose=1)  # Detailed progress
```

### Saving Space

Don't save models during experiments:
```bash
python train.py --model baseline --epochs 10
# (no --save-model flag)
```

### Custom Output Directory

Organize results by experiment:
```bash
python train.py --model baseline --output-dir experiment1
python train.py --model improved --output-dir experiment2
```

## Troubleshooting

### Issue: "No module named 'src'"
**Solution:** Run from project root directory:
```bash
cd /path/to/MNIST
python train.py
```

### Issue: TensorFlow not found
**Solution:** Install dependencies:
```bash
pip install -r requirements.txt
```

### Issue: Out of memory
**Solution:** Reduce batch size:
```bash
python train.py --batch-size 16
```

### Issue: Slow training
**Solutions:**
1. Use fewer epochs: `--epochs 5`
2. Use smaller model: `--model baseline`
3. Use GPU if available
4. Reduce data size (for testing)

---

For more information, see `README_NEW.md` or the original Jupyter notebook.
